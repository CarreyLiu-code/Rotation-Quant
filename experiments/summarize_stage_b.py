from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage B run outputs.")
    parser.add_argument("run_dir", help="Stage B output directory.")
    return parser.parse_args()


def to_markdown_table(df: pd.DataFrame) -> str:
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in df.columns) + " |")
    return "\n".join(lines)


def summarize_metrics(path: Path, keys: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    return (
        df.groupby(keys, dropna=False)[["relative_mse", "cosine", "sqnr_db"]]
        .mean()
        .reset_index()
        .sort_values(keys)
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    markdown = [f"# Stage B Summary: {run_dir.name}", ""]

    if (run_dir / "activation_metrics.csv").exists():
        summary = summarize_metrics(run_dir / "activation_metrics.csv", ["site", "method", "bits"])
        summary.to_csv(run_dir / "summary_by_tensor.csv", index=False)
        markdown.extend(["## Activation", "", to_markdown_table(summary.round(6)), ""])

    if (run_dir / "linear_metrics.csv").exists():
        summary = summarize_metrics(run_dir / "linear_metrics.csv", ["method_key", "method", "bits"])
        summary.to_csv(run_dir / "summary_linear_by_method.csv", index=False)
        markdown.extend(["## Linear", "", to_markdown_table(summary.round(6)), ""])

    if (run_dir / "ffn_metrics.csv").exists():
        summary = summarize_metrics(run_dir / "ffn_metrics.csv", ["method_key", "method", "bits"])
        summary.to_csv(run_dir / "summary_ffn_by_method.csv", index=False)
        markdown.extend(["## FFN", "", to_markdown_table(summary.round(6)), ""])

    if (run_dir / "ppl.csv").exists():
        ppl = pd.read_csv(run_dir / "ppl.csv")
        markdown.extend(["## PPL", "", to_markdown_table(ppl.round(6)), ""])

    (run_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")


if __name__ == "__main__":
    main()
