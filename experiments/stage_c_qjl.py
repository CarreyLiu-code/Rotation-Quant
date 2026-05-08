from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from rotationquant.attention_capture import TinyLlamaAttentionCapture
from rotationquant.modeling import TINYLLAMA_BASE_DIR, load_causal_lm
from rotationquant.ppl import load_text_dataset, tokenize_texts
from rotationquant.run_metadata import build_run_metadata, create_run_output_dir, write_run_metadata
from rotationquant.stage_c import STAGE_C_QJL_SPECS, evaluate_qjl_residual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage C C3 QJL residual sweep.")
    parser.add_argument("--model-dir", default=TINYLLAMA_BASE_DIR)
    parser.add_argument("--output-dir", default="outputs/stage_c")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["hadamard_lm_k3", "hadamard_lm_k2", "hadamard_lm_k2_qjl", "hadamard_lm_k3_qjl"],
    )
    parser.add_argument("--qjl-seed", type=int, default=0)
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
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    return parser.parse_args()


def write_records(records: list[dict[str, object]], output_dir: Path) -> None:
    with (output_dir / "qjl_metrics.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if records:
        fieldnames = sorted({key for record in records for key in record})
        with (output_dir / "qjl_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


def to_markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def summarize(records: list[dict[str, object]], output_dir: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        return
    df = pd.DataFrame(records)
    if df.empty:
        return
    metric_columns = [
        "ip_bias",
        "ip_variance",
        "score_relative_mse",
        "softmax_kl",
        "topk_overlap",
        "output_cosine",
    ]
    summary = df.groupby(["method_key", "method", "bits"], dropna=False)[metric_columns].mean().reset_index()
    summary.to_csv(output_dir / "summary_by_method.csv", index=False)
    markdown = [
        "# Stage C C3 QJL Summary",
        "",
        to_markdown_table(
            summary.round(6).to_dict(orient="records"),
            ["method_key", "method", "bits", "score_relative_mse", "softmax_kl", "topk_overlap", "output_cosine"],
        ),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    unknown = sorted(set(args.methods) - set(STAGE_C_QJL_SPECS))
    if unknown:
        raise ValueError(f"Unknown Stage C QJL methods: {unknown}")

    output_dir, run_id, timestamp = create_run_output_dir(args.output_dir, "stage_c_qjl")
    texts = load_text_dataset(args.dataset, args.dataset_config, args.split, text_column=args.text_column)
    model, tokenizer = load_causal_lm(args.model_dir, dtype=args.dtype, device_map=args.device_map)
    if args.device is not None and args.device_map is None:
        model.to(args.device)
    input_ids = tokenize_texts(tokenizer, list(texts), max_samples=args.max_samples)[:, : args.sequence_length]
    input_ids = input_ids.to(args.device or next(model.parameters()).device)

    with TinyLlamaAttentionCapture(model, layer_limit=args.layer_limit) as capture:
        with torch.no_grad():
            model(input_ids)

    selected_layers = set(args.layers) if args.layers is not None else None
    records: list[dict[str, object]] = []
    selected_records = [item for item in capture.records if selected_layers is None or item.layer_index in selected_layers]
    progress = tqdm(total=len(selected_records) * len(args.methods), desc="Stage C QJL")
    for item in selected_records:
        for method_key in args.methods:
            spec = STAGE_C_QJL_SPECS[method_key]
            metrics = evaluate_qjl_residual(
                item.q_rope.float(),
                item.k_rope.float(),
                item.v_proj_out.float(),
                item.attention_mask.float() if item.attention_mask is not None else None,
                item.scaling,
                item.num_key_value_groups,
                spec,
                seed=args.qjl_seed + item.layer_index,
            )
            records.append(
                {
                    "layer_index": item.layer_index,
                    "layer": item.layer,
                    "method_key": method_key,
                    "qjl_seed": args.qjl_seed + item.layer_index,
                    **metrics,
                }
            )
            progress.update(1)
    progress.close()

    write_records(records, output_dir)
    summarize(records, output_dir)
    write_run_metadata(
        build_run_metadata(
            experiment="stage_c_qjl",
            args=args,
            output_dir=output_dir,
            run_id=run_id,
            timestamp=timestamp,
            extra={
                "record_count": len(records),
                "captured_attention_count": len(capture.records),
                "selected_attention_count": len(selected_records),
                "duration_seconds": round(time.perf_counter() - start_time, 3),
                "output_files": ["qjl_metrics.jsonl", "qjl_metrics.csv", "summary_by_method.csv", "summary.md"],
            },
        ),
        output_dir,
    )


if __name__ == "__main__":
    main()
