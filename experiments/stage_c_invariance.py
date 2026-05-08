from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from rotationquant.attention_capture import TinyLlamaAttentionCapture
from rotationquant.modeling import TINYLLAMA_BASE_DIR, load_causal_lm
from rotationquant.ppl import load_text_dataset, tokenize_texts
from rotationquant.run_metadata import build_run_metadata, create_run_output_dir, write_run_metadata
from rotationquant.stage_c import invariance_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage C C1 post-RoPE attention invariance sanity.")
    parser.add_argument("--model-dir", default=TINYLLAMA_BASE_DIR)
    parser.add_argument("--output-dir", default="outputs/stage_c")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--layer-limit", type=int, default=1)
    parser.add_argument("--pass-threshold", type=float, default=1e-5)
    return parser.parse_args()


def write_records(records: list[dict[str, object]], output_dir: Path) -> None:
    with (output_dir / "invariance_metrics.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if records:
        fieldnames = sorted({key for record in records for key in record})
        with (output_dir / "invariance_metrics.csv").open("w", encoding="utf-8", newline="") as f:
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
    summary = df[["score_relative_mse", "max_score_abs_diff", "output_relative_mse", "output_cosine"]].mean()
    rows = df.round(8).to_dict(orient="records")
    markdown = [
        "# Stage C C1 Invariance Summary",
        "",
        to_markdown_table(
            rows,
            ["layer_index", "score_relative_mse", "max_score_abs_diff", "output_relative_mse", "output_cosine", "pass"],
        ),
        "",
        "## Mean",
        "",
        to_markdown_table([summary.round(8).to_dict()], list(summary.index)),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    output_dir, run_id, timestamp = create_run_output_dir(args.output_dir, "stage_c_invariance")
    texts = load_text_dataset(args.dataset, args.dataset_config, args.split, text_column=args.text_column)
    model, tokenizer = load_causal_lm(args.model_dir, dtype=args.dtype, device_map=args.device_map)
    if args.device is not None and args.device_map is None:
        model.to(args.device)
    input_ids = tokenize_texts(tokenizer, list(texts), max_samples=args.max_samples)[:, : args.sequence_length]
    input_ids = input_ids.to(args.device or next(model.parameters()).device)

    with TinyLlamaAttentionCapture(model, layer_limit=args.layer_limit) as capture:
        with torch.no_grad():
            model(input_ids)

    records: list[dict[str, object]] = []
    for item in capture.records:
        metrics = invariance_metrics(
            item.q_rope.float(),
            item.k_rope.float(),
            item.v_proj_out.float(),
            item.attention_mask.float() if item.attention_mask is not None else None,
            item.scaling,
            item.num_key_value_groups,
        )
        records.append(
            {
                "layer_index": item.layer_index,
                "layer": item.layer,
                "method": "no_quant_headwise_hadamard",
                "q_heads": int(item.q_rope.shape[1]),
                "kv_heads": int(item.k_rope.shape[1]),
                "sequence_length": int(item.q_rope.shape[2]),
                "head_dim": int(item.q_rope.shape[3]),
                "pass": bool(metrics["score_relative_mse"] <= args.pass_threshold),
                **metrics,
            }
        )

    write_records(records, output_dir)
    summarize(records, output_dir)
    write_run_metadata(
        build_run_metadata(
            experiment="stage_c_invariance",
            args=args,
            output_dir=output_dir,
            run_id=run_id,
            timestamp=timestamp,
            extra={
                "record_count": len(records),
                "captured_attention_count": len(capture.records),
                "duration_seconds": round(time.perf_counter() - start_time, 3),
                "output_files": ["invariance_metrics.jsonl", "invariance_metrics.csv", "summary.md"],
            },
        ),
        output_dir,
    )


if __name__ == "__main__":
    main()
