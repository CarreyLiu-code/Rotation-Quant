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
from rotationquant.stage_b import STAGE_B_MODEL_METHODS, STAGE_B_METHODS, apply_stage_b_ffn_fake_quant_


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B B3 FFN-only model-level PPL evaluation.")
    parser.add_argument("--model-dir", default=TINYLLAMA_BASE_DIR)
    parser.add_argument("--output-dir", default="outputs/stage_b")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "fp16",
            "ffn_direct_absmax_w4a4",
            "ffn_rot_absmax_w4a4",
            "ffn_rot_lm_w4a4",
            "ffn_rot_lm_w3a4",
            "ffn_rot_lm_w4a3",
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
            "bits": "FP16",
            "w_bits": 16,
            "a_bits": 16,
            "rotation": "none",
            "rotation_backend": "none",
            "block_size": "",
            "mxfp4_group_size": "",
            "compute_interpretation": "baseline",
            "quantized_ffn_modules": 0,
        }
    spec = STAGE_B_MODEL_METHODS[method_name]
    method = STAGE_B_METHODS[spec.method]
    return {
        "method": method.name,
        "bits": spec.label,
        "w_bits": spec.w_bits,
        "a_bits": spec.a_bits,
        "rotation": method.rotation,
        "rotation_backend": method.rotation_backend or "none",
        "block_size": "",
        "mxfp4_group_size": "",
        "compute_interpretation": method.compute_interpretation,
        "quantized_ffn_modules": quantized_layers,
    }


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    unknown_methods = sorted(set(args.methods) - (set(STAGE_B_MODEL_METHODS) | {"fp16"}))
    if unknown_methods:
        raise ValueError(f"Unknown Stage B PPL methods: {unknown_methods}")

    output_dir, run_id, timestamp = create_run_output_dir(args.output_dir, "stage_b_ppl")
    texts = load_text_dataset(args.dataset, args.dataset_config, args.split, text_column=args.text_column)
    records: list[dict[str, object]] = []

    for method_name in args.methods:
        model, tokenizer = load_causal_lm(args.model_dir, dtype=args.dtype, device_map=args.device_map)
        quant_metadata: list[dict[str, object]] = []
        if method_name != "fp16":
            # Build fake-quant FFN wrappers before moving the model to MPS.
            # This keeps one-time weight codebook mapping on CPU and avoids
            # paying that cost inside every forward pass.
            quant_metadata = apply_stage_b_ffn_fake_quant_(
                model,
                method_name=method_name,
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
            "ppl": ppl,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "max_samples": args.max_samples,
            "sequence_length": args.sequence_length,
            "stride": args.stride,
            "block_size": args.block_size if method_name != "fp16" else "",
            "mxfp4_group_size": args.mxfp4_group_size if method_name != "fp16" and STAGE_B_METHODS[STAGE_B_MODEL_METHODS[method_name].method].quantizer == "mxfp4_e2m1" else "",
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
    write_run_metadata(
        build_run_metadata(
            experiment="stage_b_ppl",
            args=args,
            output_dir=output_dir,
            run_id=run_id,
            timestamp=timestamp,
            extra={
                "record_count": len(records),
                "duration_seconds": round(time.perf_counter() - start_time, 3),
                "output_files": ["ppl_runs.jsonl", "ppl.csv"],
            },
        ),
        output_dir,
        filename="run_metadata.json",
    )


if __name__ == "__main__":
    main()
