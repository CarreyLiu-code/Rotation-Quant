from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from tqdm import tqdm

from rotationquant.modeling import TINYLLAMA_BASE_DIR, iter_llama_target_linears, load_causal_lm
from rotationquant.run_metadata import build_run_metadata, create_run_output_dir, write_run_metadata
from rotationquant.stage_a import STAGE_A_METHODS, stage_a_method_supports_bits, stage_a_tensor_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A weight-only tensor-level quantization sweep.")
    parser.add_argument("--model-dir", default=TINYLLAMA_BASE_DIR)
    parser.add_argument("--output-dir", default="outputs/stage_a")
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 3, 2])
    parser.add_argument("--methods", nargs="+", default=["direct_absmax", "hadamard_absmax", "hadamard_lm"])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--mxfp4-group-size", type=int, default=32)
    parser.add_argument("--rotation-seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--layer-limit", type=int, default=None)
    return parser.parse_args()


def write_outputs(records: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "tensor_metrics.jsonl"
    csv_path = output_dir / "tensor_metrics.csv"

    # JSONL preserves all metadata for later scripted analysis; CSV is the
    # convenient first-pass table for comparing A1/A2/A3 across layers.
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if records:
        fieldnames = sorted({key for record in records for key in record.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    unknown_methods = sorted(set(args.methods) - set(STAGE_A_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")

    output_dir, run_id, timestamp = create_run_output_dir(args.output_dir, "stage_a_tensor_sweep")
    model, _ = load_causal_lm(args.model_dir, dtype=args.dtype, device_map=args.device_map)
    layers = list(iter_llama_target_linears(model))
    if args.layer_limit is not None:
        layers = layers[: args.layer_limit]

    records: list[dict[str, object]] = []
    method_bit_pairs = [(method, bits) for method in args.methods for bits in args.bits if stage_a_method_supports_bits(method, bits)]
    total = len(layers) * len(method_bit_pairs)
    progress = tqdm(total=total, desc="Stage A tensor sweep")
    for layer_name, layer in layers:
        # Move one layer at a time to CPU so this tensor-level sweep works on
        # machines without enough unified memory for many extra model copies.
        weight = layer.weight.detach().cpu()
        for method, bits in method_bit_pairs:
            records.append(
                stage_a_tensor_record(
                    layer_name=layer_name,
                    weight=weight,
                    bits=bits,
                    method_name=method,
                    block_size=args.block_size,
                    mxfp4_group_size=args.mxfp4_group_size,
                    rotation_seed=args.rotation_seed,
                )
            )
            progress.update(1)
    progress.close()

    write_outputs(records, output_dir)
    write_run_metadata(
        build_run_metadata(
            experiment="stage_a_tensor_sweep",
            args=args,
            output_dir=output_dir,
            run_id=run_id,
            timestamp=timestamp,
            extra={
                "record_count": len(records),
                "target_layer_count": len(layers),
                "duration_seconds": round(time.perf_counter() - start_time, 3),
                "output_files": ["tensor_metrics.jsonl", "tensor_metrics.csv"],
            },
        ),
        output_dir,
        filename="run_metadata.json",
    )


if __name__ == "__main__":
    main()
