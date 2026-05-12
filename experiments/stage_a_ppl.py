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
from rotationquant.stage_a import STAGE_A_METHODS, stage_a_method_supports_bits
from rotationquant.stage_a_model import apply_stage_a_weight_quant_


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A A16Wb model-level PPL evaluation.")
    parser.add_argument("--model-dir", default=TINYLLAMA_BASE_DIR)
    parser.add_argument("--output-dir", default="outputs/stage_a")
    parser.add_argument("--methods", nargs="+", default=["fp16", "direct_absmax", "hadamard_absmax", "hadamard_lm"])
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
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    output_dir, run_id, timestamp = create_run_output_dir(args.output_dir, "stage_a_ppl")

    texts = load_text_dataset(args.dataset, args.dataset_config, args.split, text_column=args.text_column)
    records: list[dict[str, object]] = []

    for method in args.methods:
        if method != "fp16" and method not in STAGE_A_METHODS:
            raise ValueError(f"Unknown Stage A method: {method}")
        bit_list = [16] if method == "fp16" else [bits for bits in args.bits if stage_a_method_supports_bits(method, bits)]
        for bits in bit_list:
            # Reload per run so each quantized model starts from the same FP checkpoint.
            model, tokenizer = load_causal_lm(args.model_dir, dtype=args.dtype, device_map=args.device_map)
            if args.device is not None and args.device_map is None:
                model.to(args.device)
            quant_metadata: list[dict[str, object]] = []
            if method != "fp16":
                quant_metadata = apply_stage_a_weight_quant_(
                    model,
                    bits=bits,
                    method_name=method,
                    block_size=args.block_size,
                    mxfp4_group_size=args.mxfp4_group_size,
                    rotation_seed=args.rotation_seed,
                )
            input_ids = tokenize_texts(tokenizer, list(texts), max_samples=args.max_samples)
            ppl = evaluate_causal_lm_ppl(
                model,
                input_ids,
                sequence_length=args.sequence_length,
                stride=args.stride,
                device=args.device,
            )
            record = {
                "method": method,
                "bits": bits,
                "ppl": ppl,
                "block_size": args.block_size if method != "fp16" else "",
                "mxfp4_group_size": args.mxfp4_group_size if method != "fp16" and STAGE_A_METHODS[method].quantizer == "mxfp4_e2m1" else "",
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "split": args.split,
                "max_samples": args.max_samples,
                "sequence_length": args.sequence_length,
                "stride": args.stride,
                "quantized_layers": len(quant_metadata),
            }
            records.append(record)
            with (output_dir / "ppl_runs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with (output_dir / "ppl.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    write_run_metadata(
        build_run_metadata(
            experiment="stage_a_ppl",
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
