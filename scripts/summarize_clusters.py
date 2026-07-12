#!/usr/bin/env python3
"""Create a compact Markdown summary from batch clustering outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def representative_titles(df: pd.DataFrame, limit: int = 3) -> list[str]:
    if "representative_rank" in df.columns:
        df = df.sort_values("representative_rank")
    return [str(title) for title in df["title"].head(limit)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/clustering_results.csv")
    parser.add_argument("--keyword-summary", default="outputs/keyword_summary.csv")
    parser.add_argument("--output", default="outputs/clustering_summary.md")
    args = parser.parse_args()

    results = pd.read_csv(args.input).fillna("")
    keyword_summary = pd.read_csv(args.keyword_summary).fillna("")
    lines = ["# Clustering Results Summary", ""]

    lines.append("## Keyword Status")
    for _, row in keyword_summary.iterrows():
        lines.append(f"- {row['keyword']}: {row['paper_count']} papers, {row['status']}")

    clustered = results[results.get("cluster_status", "") == "clustered"].copy()
    if not clustered.empty:
        lines.append("")
        lines.append("## Clustered Groups")
        group_cols = ["focus_keyword", "cluster_method", "cluster"]
        for (keyword, method, cluster), subset in clustered.groupby(group_cols):
            label = str(subset.iloc[0].get("cluster_label_candidate", ""))
            summary = str(subset.iloc[0].get("cluster_summary_candidate", ""))
            lines.append("")
            lines.append(f"### {keyword} / {method} / Cluster {cluster}")
            lines.append(f"- Papers: {len(subset)}")
            if label:
                lines.append(f"- Label candidate: {label}")
            if summary:
                lines.append(f"- Summary candidate: {summary}")
            lines.append("- Representative papers:")
            for title in representative_titles(subset):
                lines.append(f"  - {title}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
