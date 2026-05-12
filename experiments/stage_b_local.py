from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from rotationquant.activation_capture import TinyLlamaLocalIOCapture
from rotationquant.metrics import tensor_metrics
from rotationquant.modeling import TINYLLAMA_BASE_DIR, load_causal_lm
from rotationquant.ppl import load_text_dataset, tokenize_texts
from rotationquant.run_metadata import build_run_metadata, create_run_output_dir, write_run_metadata
from rotationquant.stage_b import (
    STAGE_B_FFN_SPECS,
    STAGE_B_LINEAR_SPECS,
    STAGE_B_METHODS,
    fake_quant_ffn_from_weights,
    fake_quant_linear,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B B2 local Linear / FFN fake quant experiments.")
    parser.add_argument("--model-dir", default=TINYLLAMA_BASE_DIR)
    parser.add_argument("--output-dir", default="outputs/stage_b")
    parser.add_argument(
        "--linear-methods",
        nargs="+",
        default=[
            "direct_absmax_w4a4",
            "rot_absmax_w4a4",
            "rot_lm_w4a4",
            "rot_lm_w3a4",
            "rot_lm_w4a3",
            "rot_lm_w3a3",
            "rot_lm_w2a4",
        ],
    )
    parser.add_argument(
        "--ffn-methods",
        nargs="+",
        default=[
            "ffn_fp16",
            "ffn_direct_absmax_w4a4",
            "ffn_rot_absmax_w4a4",
            "ffn_rot_lm_w4a4",
            "ffn_rot_lm_w3a4",
            "ffn_rot_lm_w4a3",
            "ffn_rot_lm_w3a3",
        ],
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--mxfp4-group-size", type=int, default=32)
    parser.add_argument("--rotation-seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--layer-limit", type=int, default=None)
    return parser.parse_args()


def write_records(records: list[dict[str, object]], output_dir: Path, stem: str) -> None:
    with (output_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if records:
        fieldnames = sorted({key for record in records for key in record.keys()})
        with (output_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


def to_markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def summarize(linear_records: list[dict[str, object]], ffn_records: list[dict[str, object]], output_dir: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        return
    markdown = ["# Stage B B2 Local Summary", ""]
    if linear_records:
        linear_df = pd.DataFrame(linear_records)
        linear_summary = (
            linear_df.groupby(["method_key", "method", "bits"], dropna=False)[["relative_mse", "cosine", "sqnr_db"]]
            .mean()
            .reset_index()
        )
        linear_summary.to_csv(output_dir / "summary_linear_by_method.csv", index=False)
        markdown.extend(
            [
                "## Linear",
                "",
                to_markdown_table(
                    linear_summary.round(6).to_dict(orient="records"),
                    ["method_key", "method", "bits", "relative_mse", "cosine", "sqnr_db"],
                ),
                "",
            ]
        )
    if ffn_records:
        ffn_df = pd.DataFrame(ffn_records)
        ffn_summary = (
            ffn_df.groupby(["method_key", "method", "bits"], dropna=False)[["relative_mse", "cosine", "sqnr_db"]]
            .mean()
            .reset_index()
        )
        ffn_summary.to_csv(output_dir / "summary_ffn_by_method.csv", index=False)
        markdown.extend(
            [
                "## FFN",
                "",
                to_markdown_table(
                    ffn_summary.round(6).to_dict(orient="records"),
                    ["method_key", "method", "bits", "relative_mse", "cosine", "sqnr_db"],
                ),
                "",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    unknown_linear = sorted(set(args.linear_methods) - set(STAGE_B_LINEAR_SPECS))
    unknown_ffn = sorted(set(args.ffn_methods) - (set(STAGE_B_FFN_SPECS) | {"ffn_fp16"}))
    if unknown_linear:
        raise ValueError(f"Unknown Stage B linear methods: {unknown_linear}")
    if unknown_ffn:
        raise ValueError(f"Unknown Stage B FFN methods: {unknown_ffn}")

    output_dir, run_id, timestamp = create_run_output_dir(args.output_dir, "stage_b_local")
    texts = load_text_dataset(args.dataset, args.dataset_config, args.split, text_column=args.text_column)
    model, tokenizer = load_causal_lm(args.model_dir, dtype=args.dtype, device_map=args.device_map)
    if args.device is not None and args.device_map is None:
        model.to(args.device)
    input_ids = tokenize_texts(tokenizer, list(texts), max_samples=args.max_samples)[:, : args.sequence_length]
    input_ids = input_ids.to(args.device or next(model.parameters()).device)

    with TinyLlamaLocalIOCapture(model, layer_limit=args.layer_limit) as capture:
        with torch.no_grad():
            model(input_ids)

    linear_records: list[dict[str, object]] = []
    total_linear = len(capture.linear_records) * len(args.linear_methods)
    progress = tqdm(total=total_linear, desc="Stage B local Linear")
    for item in capture.linear_records:
        x = item.input.float()
        reference = item.output.float()
        weight = item.module.weight.detach().cpu().float()
        bias = getattr(item.module, "bias", None)
        bias_cpu = bias.detach().cpu().float() if bias is not None else None
        for method_key in args.linear_methods:
            spec = STAGE_B_LINEAR_SPECS[method_key]
            candidate, metadata = fake_quant_linear(
                x,
                weight,
                bias_cpu,
                spec,
                block_size=args.block_size,
                mxfp4_group_size=args.mxfp4_group_size,
                rotation_seed=args.rotation_seed,
            )
            linear_records.append(
                {
                    "layer": item.layer,
                    "site": "linear",
                    "method_key": method_key,
                    **metadata,
                    **tensor_metrics(reference, candidate),
                }
            )
            progress.update(1)
    progress.close()

    ffn_records: list[dict[str, object]] = []
    total_ffn = len(capture.ffn_records) * len(args.ffn_methods)
    progress = tqdm(total=total_ffn, desc="Stage B local FFN")
    for item in capture.ffn_records:
        x = item.input.float()
        reference = item.output.float()
        weights = {
            "gate": item.module.gate_proj.weight.detach().cpu().float(),
            "up": item.module.up_proj.weight.detach().cpu().float(),
            "down": item.module.down_proj.weight.detach().cpu().float(),
        }
        for method_key in args.ffn_methods:
            if method_key == "ffn_fp16":
                candidate = reference
                metadata = {
                    "method": "fp16",
                    "bits": "FP16",
                    "w_bits": 16,
                    "a_bits": 16,
                    "rotation": "none",
                    "compute_interpretation": "baseline",
                }
            else:
                spec = STAGE_B_FFN_SPECS[method_key]
                candidate, metadata = fake_quant_ffn_from_weights(
                    x,
                    weights["gate"],
                    weights["up"],
                    weights["down"],
                    spec,
                    block_size=args.block_size,
                    mxfp4_group_size=args.mxfp4_group_size,
                    rotation_seed=args.rotation_seed,
                    act_owner=None,
                )
            method = metadata["method"]
            if method in STAGE_B_METHODS:
                metadata["compute_interpretation"] = STAGE_B_METHODS[method].compute_interpretation
            ffn_records.append(
                {
                    "layer": item.layer,
                    "site": "ffn",
                    "method_key": method_key,
                    **metadata,
                    **tensor_metrics(reference, candidate),
                }
            )
            progress.update(1)
    progress.close()

    write_records(linear_records, output_dir, "linear_metrics")
    write_records(ffn_records, output_dir, "ffn_metrics")
    summarize(linear_records, ffn_records, output_dir)
    write_run_metadata(
        build_run_metadata(
            experiment="stage_b_local",
            args=args,
            output_dir=output_dir,
            run_id=run_id,
            timestamp=timestamp,
            extra={
                "linear_record_count": len(linear_records),
                "ffn_record_count": len(ffn_records),
                "captured_linear_count": len(capture.linear_records),
                "captured_ffn_count": len(capture.ffn_records),
                "duration_seconds": round(time.perf_counter() - start_time, 3),
                "output_files": [
                    "linear_metrics.jsonl",
                    "linear_metrics.csv",
                    "ffn_metrics.jsonl",
                    "ffn_metrics.csv",
                    "summary_linear_by_method.csv",
                    "summary_ffn_by_method.csv",
                    "summary.md",
                ],
            },
        ),
        output_dir,
        filename="run_metadata.json",
    )


if __name__ == "__main__":
    main()
