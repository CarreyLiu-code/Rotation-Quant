from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage A tensor-level metrics.")
    parser.add_argument("run_dir", help="Stage A tensor sweep output directory.")
    return parser.parse_args()


def layer_group(layer: str) -> str:
    if ".self_attn." in layer:
        return "attention"
    if ".mlp." in layer:
        return "ffn"
    return "other"


def projection_name(layer: str) -> str:
    return layer.rsplit(".", 1)[-1]


def aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metrics = ["relative_mse", "cosine", "sqnr_db"]
    grouped = df.groupby(keys, dropna=False)[metrics]
    summary = grouped.agg(["mean", "median", "min", "max"])
    p90 = grouped.quantile(0.9).rename(columns={metric: f"{metric}_p90" for metric in metrics})
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index().merge(p90.reset_index(), on=keys)


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([str(row[column]) for column in df.columns])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    df = pd.read_csv(run_dir / "tensor_metrics.csv")
    df["layer_group"] = df["layer"].map(layer_group)
    df["projection"] = df["layer"].map(projection_name)

    method_summary = aggregate(df, ["method", "bits"])
    group_summary = aggregate(df, ["layer_group", "method", "bits"])
    projection_summary = aggregate(df, ["projection", "method", "bits"])

    method_summary.to_csv(run_dir / "summary_by_method.csv", index=False)
    group_summary.to_csv(run_dir / "summary_by_group.csv", index=False)
    projection_summary.to_csv(run_dir / "summary_by_projection.csv", index=False)

    markdown = [
        "# Stage A Tensor Summary",
        "",
        "## By Method",
        "",
        to_markdown_table(method_summary),
        "",
        "## By Layer Group",
        "",
        to_markdown_table(group_summary),
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")


if __name__ == "__main__":
    main()
