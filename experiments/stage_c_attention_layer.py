from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from rotationquant.attention_capture import TinyLlamaAttentionCapture
from rotationquant.metrics import cosine_similarity, relative_mse
from rotationquant.modeling import TINYLLAMA_BASE_DIR, load_causal_lm
from rotationquant.ppl import load_text_dataset, tokenize_texts
from rotationquant.run_metadata import build_run_metadata, create_run_output_dir, write_run_metadata
from rotationquant.stage_b import STAGE_B_METHODS
from rotationquant.stage_c import (
    STAGE_C_KV_SPECS,
    STAGE_C_STRUCTURED_ATTENTION_SPECS,
    AttentionComputation,
    attention_quality_metrics,
    reference_attention,
)
from rotationquant.stage_c_model import attention_layer_metrics_from_details, fake_quant_attention_from_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage C C4 local attention-layer experiments.")
    parser.add_argument("--model-dir", default=TINYLLAMA_BASE_DIR)
    parser.add_argument("--output-dir", default="outputs/stage_c")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "fp16",
            "attn_identity_fp16",
            "attn_kv_hlm_k4v4_reconstruct",
            "attn_kv_hlm_k4v4",
            "attn_kv_hlm_k3v4",
            "attn_kv_hlm_k4v3",
            "attn_rot_lm_w4a4_hlm_k4v4",
            "attn_rot_lm_w3a4_hlm_k3v4",
            "attn_rot_lm_w4a3_hlm_k4v3",
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


def write_records(records: list[dict[str, object]], output_dir: Path) -> None:
    with (output_dir / "attention_layer_metrics.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if records:
        fieldnames = sorted({key for record in records for key in record})
        with (output_dir / "attention_layer_metrics.csv").open("w", encoding="utf-8", newline="") as f:
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
        "projection_relative_mse",
        "score_relative_mse",
        "softmax_kl",
        "topk_overlap",
        "pre_o_output_relative_mse",
        "pre_o_output_cosine",
        "layer_output_relative_mse",
        "layer_output_cosine",
    ]
    group_columns = ["method_key", "linear_bits", "kv_bits", "value_path"]
    for optional_column in ["block_size", "kv_block_size", "o_proj_domain_block_size"]:
        if optional_column in df.columns:
            group_columns.append(optional_column)
    by_method = df.groupby(group_columns, dropna=False)[metric_columns].mean().reset_index()
    by_layer = df.groupby(["layer_index", "method_key"], dropna=False)[metric_columns].mean().reset_index()
    by_method.to_csv(output_dir / "summary_by_method.csv", index=False)
    by_layer.to_csv(output_dir / "summary_by_layer.csv", index=False)
    markdown = [
        "# Stage C C4 Attention-layer Summary",
        "",
        to_markdown_table(
            by_method.round(6).to_dict(orient="records"),
            [
                "method_key",
                "linear_bits",
                "kv_bits",
                "value_path",
                "block_size",
                "kv_block_size",
                "score_relative_mse",
                "softmax_kl",
                "pre_o_output_cosine",
                "layer_output_relative_mse",
                "layer_output_cosine",
            ],
        ),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")


def fp16_record() -> dict[str, object]:
    return {
        "method": "fp16",
        "linear_method": "fp16",
        "linear_bits": "FP16",
        "w_bits": 16,
        "a_bits": 16,
        "kv_method": "fp16",
        "kv_bits": "K16V16",
        "k_bits": 16,
        "v_bits": 16,
        "value_path": "reference",
        "compute_interpretation": "baseline",
        "projection_relative_mse": 0.0,
        "projection_cosine": 1.0,
        "q_proj_relative_mse": 0.0,
        "k_proj_relative_mse": 0.0,
        "v_proj_relative_mse": 0.0,
        "ip_bias": 0.0,
        "ip_variance": 0.0,
        "ip_relative_mse": 0.0,
        "score_mse": 0.0,
        "score_relative_mse": 0.0,
        "softmax_kl": 0.0,
        "topk_overlap": 1.0,
        "output_relative_mse": 0.0,
        "output_cosine": 1.0,
        "key_relative_mse": 0.0,
        "value_relative_mse": 0.0,
        "pre_o_output_relative_mse": 0.0,
        "pre_o_output_cosine": 1.0,
        "layer_output_relative_mse": 0.0,
        "layer_output_cosine": 1.0,
    }


def method_metadata(method_key: str) -> dict[str, object]:
    spec = STAGE_C_STRUCTURED_ATTENTION_SPECS[method_key]
    kv_spec = STAGE_C_KV_SPECS[spec.kv_spec_key]
    linear_method = STAGE_B_METHODS[spec.linear_spec.method].name if spec.linear_spec is not None else "fp16"
    return {
        "method": spec.name,
        "linear_method": linear_method,
        "linear_bits": spec.linear_spec.label if spec.linear_spec is not None else "FP16",
        "w_bits": spec.linear_spec.w_bits if spec.linear_spec is not None else 16,
        "a_bits": spec.linear_spec.a_bits if spec.linear_spec is not None else 16,
        "kv_method": kv_spec.method,
        "kv_bits": kv_spec.label,
        "k_bits": kv_spec.k_bits,
        "v_bits": kv_spec.v_bits,
        "kv_rotation_backend": kv_spec.rotation_backend,
        "kv_block_size": kv_spec.kv_block_size,
        "linear_rotation_backend": (
            STAGE_B_METHODS[spec.linear_spec.method].rotation_backend or "none"
            if spec.linear_spec is not None
            else "none"
        ),
        "value_path": spec.value_path,
        "o_proj_domain_block_size": kv_spec.kv_block_size if spec.value_path == "o_proj_absorb" else "",
        "compute_interpretation": spec.compute_interpretation,
    }


def pre_o_metrics(reference_heads: torch.Tensor, details: dict[str, torch.Tensor]) -> dict[str, float]:
    candidate = details.get("attn_output_heads_reference_domain", details["attn_output_heads"]).float()
    return {
        "pre_o_output_relative_mse": relative_mse(reference_heads, candidate),
        "pre_o_output_cosine": cosine_similarity(reference_heads, candidate),
    }


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    unknown = sorted(set(args.methods) - (set(STAGE_C_STRUCTURED_ATTENTION_SPECS) | {"fp16"}))
    if unknown:
        raise ValueError(f"Unknown Stage C attention methods: {unknown}")

    output_dir, run_id, timestamp = create_run_output_dir(args.output_dir, "stage_c_attention_layer")
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
    progress = tqdm(total=len(capture.records) * len(args.methods), desc="Stage C attention layer")
    for item in capture.records:
        attention_mask = item.attention_mask.float() if item.attention_mask is not None else None
        reference = reference_attention(
            item.q_rope.float(),
            item.k_rope.float(),
            item.v_proj_out.float(),
            attention_mask,
            item.scaling,
            item.num_key_value_groups,
        )
        for method_key in args.methods:
            if method_key == "fp16":
                metrics = fp16_record()
            else:
                output, details = fake_quant_attention_from_record(
                    item,
                    method_name=method_key,
                    block_size=args.block_size,
                    mxfp4_group_size=args.mxfp4_group_size,
                    rotation_seed=args.rotation_seed,
                )
                candidate = AttentionComputation(
                    raw_inner_product=details["raw_inner_product"].float(),
                    scores=details["scores"].float(),
                    probs=details["attn_probs"].float(),
                    output_heads=details.get("attn_output_heads_reference_domain", details["attn_output_heads"]).float(),
                    key_hat=details["key_hat"].float(),
                    value_hat=details["value_hat"].float(),
                    metadata={},
                )
                metrics = {
                    **method_metadata(method_key),
                    **attention_quality_metrics(reference, candidate, attention_mask),
                    **pre_o_metrics(item.attn_output_heads.float(), details),
                    **attention_layer_metrics_from_details(item, output, details),
                }
            records.append(
                {
                    "layer_index": item.layer_index,
                    "layer": item.layer,
                    "method_key": method_key,
                    "block_size": args.block_size if method_key != "fp16" else "",
                    "mxfp4_group_size": args.mxfp4_group_size if method_key != "fp16" else "",
                    **metrics,
                }
            )
            progress.update(1)
    progress.close()

    write_records(records, output_dir)
    summarize(records, output_dir)
    write_run_metadata(
        build_run_metadata(
            experiment="stage_c_attention_layer",
            args=args,
            output_dir=output_dir,
            run_id=run_id,
            timestamp=timestamp,
            extra={
                "record_count": len(records),
                "captured_attention_count": len(capture.records),
                "duration_seconds": round(time.perf_counter() - start_time, 3),
                "output_files": [
                    "attention_layer_metrics.jsonl",
                    "attention_layer_metrics.csv",
                    "summary_by_method.csv",
                    "summary_by_layer.csv",
                    "summary.md",
                ],
            },
        ),
        output_dir,
    )


if __name__ == "__main__":
    main()
