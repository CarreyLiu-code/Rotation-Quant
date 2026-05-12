from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from rotationquant.modeling import TINYLLAMA_BASE_DIR, load_causal_lm
from rotationquant.ppl import evaluate_causal_lm_ppl, load_text_dataset, tokenize_texts
from rotationquant.run_metadata import build_run_metadata, create_run_output_dir, write_run_metadata
from rotationquant.stage_b import STAGE_B_METHODS
from rotationquant.stage_c import STAGE_C_KV_SPECS, STAGE_C_STRUCTURED_ATTENTION_SPECS
from rotationquant.stage_c_model import apply_stage_c_attention_fake_quant_


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage C C5 Attention-only model-level PPL evaluation.")
    parser.add_argument("--model-dir", default=TINYLLAMA_BASE_DIR)
    parser.add_argument("--output-dir", default="outputs/stage_c")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "fp16",
            "attn_identity_fp16",
            "attn_kv_hlm_k4v4",
            "attn_kv_hlm_k3v4",
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
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    return parser.parse_args()


def method_metadata(method_name: str, quantized_layers: int) -> dict[str, object]:
    if method_name == "fp16":
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
            "kv_rotation_backend": "none",
            "kv_block_size": "",
            "linear_rotation_backend": "none",
            "value_path": "reference",
            "block_size": "",
            "o_proj_domain_block_size": "",
            "mxfp4_group_size": "",
            "compute_interpretation": "baseline",
            "quantized_attention_modules": 0,
        }
    spec = STAGE_C_STRUCTURED_ATTENTION_SPECS[method_name]
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
        "quantized_attention_modules": quantized_layers,
    }


def write_summary(records: list[dict[str, object]], output_dir: Path) -> None:
    columns = ["method_key", "linear_bits", "kv_bits", "block_size", "kv_block_size", "value_path", "ppl", "compute_interpretation"]
    lines = ["# Stage C C5 PPL Summary", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in records:
        lines.append("| " + " | ".join(str(record.get(column, "")) for column in columns) + " |")
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    unknown = sorted(set(args.methods) - (set(STAGE_C_STRUCTURED_ATTENTION_SPECS) | {"fp16"}))
    if unknown:
        raise ValueError(f"Unknown Stage C PPL methods: {unknown}")

    output_dir, run_id, timestamp = create_run_output_dir(args.output_dir, "stage_c_ppl")
    texts = load_text_dataset(args.dataset, args.dataset_config, args.split, text_column=args.text_column)
    records: list[dict[str, object]] = []

    for method_name in args.methods:
        model, tokenizer = load_causal_lm(args.model_dir, dtype=args.dtype, device_map=args.device_map)
        quant_metadata: list[dict[str, object]] = []
        if method_name != "fp16":
            quant_metadata = apply_stage_c_attention_fake_quant_(
                model,
                method_name,
                block_size=args.block_size,
                mxfp4_group_size=args.mxfp4_group_size,
                rotation_seed=args.rotation_seed,
            )
        if args.device is not None and args.device_map is None:
            model.to(args.device)
        input_ids = tokenize_texts(tokenizer, list(texts), max_samples=args.max_samples)
        ppl = evaluate_causal_lm_ppl(
            model,
            input_ids,
            sequence_length=args.sequence_length,
            stride=args.stride,
            device=args.device,
        )
        record = {
            "method_key": method_name,
            **method_metadata(method_name, quantized_layers=len(quant_metadata)),
            "block_size": args.block_size if method_name != "fp16" else "",
            "mxfp4_group_size": args.mxfp4_group_size if method_name != "fp16" else "",
            "ppl": ppl,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "max_samples": args.max_samples,
            "sequence_length": args.sequence_length,
            "stride": args.stride,
        }
        records.append(record)
        with (output_dir / "ppl_runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if records:
        with (output_dir / "ppl.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    write_summary(records, output_dir)
    write_run_metadata(
        build_run_metadata(
            experiment="stage_c_ppl",
            args=args,
            output_dir=output_dir,
            run_id=run_id,
            timestamp=timestamp,
            extra={
                "record_count": len(records),
                "duration_seconds": round(time.perf_counter() - start_time, 3),
                "output_files": ["ppl_runs.jsonl", "ppl.csv", "summary.md"],
            },
        ),
        output_dir,
    )


if __name__ == "__main__":
    main()
