from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from rotationquant.activation_capture import TinyLlamaActivationCapture
from rotationquant.metrics import distribution_metrics, tensor_metrics
from rotationquant.modeling import TINYLLAMA_BASE_DIR, load_causal_lm
from rotationquant.ppl import load_text_dataset, tokenize_texts
from rotationquant.run_metadata import build_run_metadata, create_run_output_dir, write_run_metadata
from rotationquant.stage_b import STAGE_B_METHODS, block_hadamard_last_dim, quantize_activation_for_b1, stage_b_method_supports_bits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B B1 activation rotation quantization sweep.")
    parser.add_argument("--model-dir", default=TINYLLAMA_BASE_DIR)
    parser.add_argument("--output-dir", default="outputs/stage_b")
    parser.add_argument("--methods", nargs="+", default=["direct_absmax", "rot_absmax", "rot_lm"])
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 3, 2])
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
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--layer-limit", type=int, default=None)
    parser.add_argument("--histogram-bins", type=int, default=64)
    parser.add_argument("--histogram-max-values", type=int, default=200_000)
    return parser.parse_args()


def to_markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def histogram_record(
    tensor: torch.Tensor,
    *,
    layer_index: int,
    site: str,
    domain: str,
    bins: int,
    max_values: int,
) -> dict[str, object]:
    values = tensor.float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() > max_values:
        indices = torch.linspace(0, values.numel() - 1, max_values).long()
        values = values[indices]
    if values.numel() == 0:
        return {"layer_index": layer_index, "site": site, "domain": domain, "bins": [], "counts": []}
    min_value = float(values.min().cpu())
    max_value = float(values.max().cpu())
    if min_value == max_value:
        return {
            "layer_index": layer_index,
            "site": site,
            "domain": domain,
            "bins": [min_value, max_value],
            "counts": [int(values.numel())],
        }
    counts = torch.histc(values, bins=bins, min=min_value, max=max_value)
    edges = torch.linspace(min_value, max_value, bins + 1)
    return {
        "layer_index": layer_index,
        "site": site,
        "domain": domain,
        "bins": [float(v) for v in edges.tolist()],
        "counts": [int(v) for v in counts.tolist()],
    }


def write_csv_jsonl(records: list[dict[str, object]], output_dir: Path) -> None:
    jsonl_path = output_dir / "activation_metrics.jsonl"
    csv_path = output_dir / "activation_metrics.csv"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if records:
        fieldnames = sorted({key for record in records for key in record.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


def summarize(records: list[dict[str, object]], output_dir: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        return
    df = pd.DataFrame(records)
    if df.empty:
        return
    grouped = df.groupby(["site", "method", "bits"], dropna=False)[["relative_mse", "cosine", "sqnr_db"]].mean()
    summary = grouped.reset_index()
    summary.to_csv(output_dir / "summary_by_tensor.csv", index=False)
    rows = summary.round(6).to_dict(orient="records")
    markdown = [
        "# Stage B B1 Activation Summary",
        "",
        to_markdown_table(rows, ["site", "method", "bits", "relative_mse", "cosine", "sqnr_db"]),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    unknown_methods = sorted(set(args.methods) - set(STAGE_B_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown Stage B B1 methods: {unknown_methods}")

    output_dir, run_id, timestamp = create_run_output_dir(args.output_dir, "stage_b_activation")
    texts = load_text_dataset(args.dataset, args.dataset_config, args.split, text_column=args.text_column)
    model, tokenizer = load_causal_lm(args.model_dir, dtype=args.dtype, device_map=args.device_map)
    if args.device is not None and args.device_map is None:
        model.to(args.device)
    input_ids = tokenize_texts(tokenizer, list(texts), max_samples=args.max_samples)[:, : args.sequence_length]
    input_ids = input_ids.to(args.device or next(model.parameters()).device)

    with TinyLlamaActivationCapture(model, layer_limit=args.layer_limit) as capture:
        with torch.no_grad():
            model(input_ids)

    records: list[dict[str, object]] = []
    histograms: list[dict[str, object]] = []
    method_bit_pairs = [(method, bits) for method in args.methods for bits in args.bits if stage_b_method_supports_bits(method, bits)]
    total = len(capture.records) * len(method_bit_pairs)
    progress = tqdm(total=total, desc="Stage B activation")
    for activation in capture.records:
        tensor = activation.tensor
        rotated = block_hadamard_last_dim(tensor, block_size=args.block_size)
        histograms.append(
            histogram_record(
                tensor,
                layer_index=activation.layer_index,
                site=activation.site,
                domain="original",
                bins=args.histogram_bins,
                max_values=args.histogram_max_values,
            )
        )
        histograms.append(
            histogram_record(
                rotated,
                layer_index=activation.layer_index,
                site=activation.site,
                domain="rotated",
                bins=args.histogram_bins,
                max_values=args.histogram_max_values,
            )
        )
        original_distribution = {f"original_{key}": value for key, value in distribution_metrics(tensor).items()}
        rotated_distribution = {f"rotated_{key}": value for key, value in distribution_metrics(rotated).items()}
        for method, bits in method_bit_pairs:
            candidate, metadata = quantize_activation_for_b1(
                tensor,
                method_name=method,
                bits=bits,
                block_size=args.block_size,
                mxfp4_group_size=args.mxfp4_group_size,
                rotation_seed=args.rotation_seed,
            )
            records.append(
                {
                    "layer_index": activation.layer_index,
                    "site": activation.site,
                    **metadata,
                    **tensor_metrics(tensor, candidate),
                    **original_distribution,
                    **rotated_distribution,
                }
            )
            progress.update(1)
    progress.close()

    write_csv_jsonl(records, output_dir)
    (output_dir / "histograms.json").write_text(json.dumps(histograms, ensure_ascii=False, indent=2), encoding="utf-8")
    summarize(records, output_dir)
    write_run_metadata(
        build_run_metadata(
            experiment="stage_b_activation",
            args=args,
            output_dir=output_dir,
            run_id=run_id,
            timestamp=timestamp,
            extra={
                "record_count": len(records),
                "captured_activation_count": len(capture.records),
                "duration_seconds": round(time.perf_counter() - start_time, 3),
                "output_files": [
                    "activation_metrics.jsonl",
                    "activation_metrics.csv",
                    "histograms.json",
                    "summary_by_tensor.csv",
                    "summary.md",
                ],
            },
        ),
        output_dir,
        filename="run_metadata.json",
    )


if __name__ == "__main__":
    main()
