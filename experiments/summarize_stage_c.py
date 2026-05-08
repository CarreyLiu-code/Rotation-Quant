from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage C run outputs.")
    parser.add_argument("run_dir", help="Stage C output directory.")
    return parser.parse_args()


def to_markdown_table(df: pd.DataFrame) -> str:
    columns = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in df.columns) + " |")
    return "\n".join(lines)


def summarize_metrics(path: Path, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    available_metrics = [metric for metric in metrics if metric in df.columns]
    return df.groupby(keys, dropna=False)[available_metrics].mean().reset_index().sort_values(keys)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    markdown = [f"# Stage C Summary: {run_dir.name}", ""]

    if (run_dir / "invariance_metrics.csv").exists():
        summary = summarize_metrics(
            run_dir / "invariance_metrics.csv",
            ["method"],
            ["score_relative_mse", "max_score_abs_diff", "output_relative_mse", "output_cosine"],
        )
        markdown.extend(["## Invariance", "", to_markdown_table(summary.round(8)), ""])

    if (run_dir / "kv_metrics.csv").exists():
        summary = summarize_metrics(
            run_dir / "kv_metrics.csv",
            ["method_key", "method", "bits"],
            ["score_relative_mse", "softmax_kl", "topk_overlap", "value_relative_mse", "output_cosine"],
        )
        summary.to_csv(run_dir / "summary_by_method.csv", index=False)
        markdown.extend(["## KV Local", "", to_markdown_table(summary.round(6)), ""])

    if (run_dir / "qjl_metrics.csv").exists():
        summary = summarize_metrics(
            run_dir / "qjl_metrics.csv",
            ["method_key", "method", "bits"],
            ["score_relative_mse", "softmax_kl", "topk_overlap", "output_cosine"],
        )
        summary.to_csv(run_dir / "summary_by_method.csv", index=False)
        markdown.extend(["## QJL", "", to_markdown_table(summary.round(6)), ""])

    if (run_dir / "attention_layer_metrics.csv").exists():
        summary = summarize_metrics(
            run_dir / "attention_layer_metrics.csv",
            ["method_key", "linear_bits", "kv_bits"],
            ["projection_relative_mse", "score_relative_mse", "softmax_kl", "layer_output_relative_mse", "layer_output_cosine"],
        )
        summary.to_csv(run_dir / "summary_by_method.csv", index=False)
        markdown.extend(["## Attention Layer", "", to_markdown_table(summary.round(6)), ""])

    if (run_dir / "ppl.csv").exists():
        ppl = pd.read_csv(run_dir / "ppl.csv")
        markdown.extend(["## PPL", "", to_markdown_table(ppl.round(6)), ""])

    (run_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")


if __name__ == "__main__":
    main()
