from __future__ import annotations

import argparse
import csv
import json
import time

from rotationquant.run_metadata import build_run_metadata, create_run_output_dir, write_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reserved Stage C C5 accuracy entrypoint.")
    parser.add_argument("--output-dir", default="outputs/stage_c")
    parser.add_argument("--benchmark", default="piqa")
    return parser.parse_args()


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    output_dir, run_id, timestamp = create_run_output_dir(args.output_dir, "stage_c_accuracy")
    record = {
        "benchmark": args.benchmark,
        "status": "reserved_not_run",
        "note": "Stage C first implementation only formalizes WikiText2 PPL; accuracy is intentionally deferred.",
    }
    with (output_dir / "accuracy_runs.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "accuracy.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        writer.writeheader()
        writer.writerow(record)
    (output_dir / "summary.md").write_text(
        "# Stage C C5 Accuracy\n\nAccuracy evaluation is reserved for a later run. No benchmark was executed.\n",
        encoding="utf-8",
    )
    write_run_metadata(
        build_run_metadata(
            experiment="stage_c_accuracy",
            args=args,
            output_dir=output_dir,
            run_id=run_id,
            timestamp=timestamp,
            extra={
                "record_count": 1,
                "status": "reserved_not_run",
                "duration_seconds": round(time.perf_counter() - start_time, 3),
                "output_files": ["accuracy_runs.jsonl", "accuracy.csv", "summary.md"],
            },
        ),
        output_dir,
    )


if __name__ == "__main__":
    main()
