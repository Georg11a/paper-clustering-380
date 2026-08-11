#!/usr/bin/env python3
"""Refresh index controls and run metadata after the 459-paper rebuild."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--index", default="docs/index.html")
    parser.add_argument("--global-manifest", default="docs/explorer/global_comparison_neutral/manifest.json")
    args = parser.parse_args()

    assignments = pd.read_csv(args.assignments)
    paper_count = assignments["paper_id"].nunique()
    grouped = (
        assignments.groupby("keyword", sort=False)
        .agg(papers=("paper_id", "nunique"), clusters=("cluster_id", "nunique"))
        .reset_index()
        .sort_values(["papers", "keyword"], ascending=[False, True])
    )
    total_clusters = assignments["cluster_id"].nunique()
    path = Path(args.index)
    html = path.read_text(encoding="utf-8")

    options = [f'<option value="All keyword groups" selected>All keyword groups ({paper_count})</option>']
    options += [
        f'<option value="{row.keyword}">{row.keyword} ({row.papers})</option>'
        for row in grouped.itertuples()
    ]
    html = re.sub(
        r'<select id="keywordSelect">.*?</select>',
        '<select id="keywordSelect">' + "\n".join(options) + '</select>',
        html,
        count=1,
        flags=re.S,
    )
    html = re.sub(
        r'<div class="current" id="currentLabel">.*?</div>',
        f'<div class="current" id="currentLabel">{paper_count} papers · {total_clusters} interpretations</div>',
        html,
        count=1,
    )
    html = html.replace(
        "Batch 1–3 are reflected here: 282 retained publications",
        f"Batch 1–3 are reflected here: {paper_count} retained English publications",
    ).replace(
        "Cleaned 282-paper class-based TF-IDF interpretation",
        f"Cleaned {paper_count}-paper class-based TF-IDF interpretation",
    ).replace(
        "<strong>Batch 1:</strong> 282 retained English publications",
        f"<strong>Batch 1:</strong> {paper_count} retained English publications",
    ).replace(
        "All 282 papers are held constant.",
        f"All {paper_count} papers are held constant.",
    )

    path1_rows = [
        f'      {{ id: "all_statistical_bottom_up", keyword: "All keyword groups", method: "statistical_bottom_up", text_view: "context", papers: "{paper_count}", params: "{total_clusters} clusters", explorer: "explorer/statistical_bottom_up/all/paper_explorer.html", summary: "explorer/statistical_bottom_up/all/cluster_summary.md" }}'
    ]
    for row in grouped.itertuples():
        slug = slugify(row.keyword)
        path1_rows.append(
            f'      {{ id: "{slug}_statistical_bottom_up", keyword: "{row.keyword}", method: "statistical_bottom_up", text_view: "context", papers: "{row.papers}", params: "{row.clusters} clusters", explorer: "explorer/statistical_bottom_up/{slug}/paper_explorer.html", summary: "explorer/statistical_bottom_up/{slug}/cluster_summary.md" }}'
        )
    path1_block = "    const path1Runs = [\n" + ",\n".join(path1_rows) + "\n    ];"
    html = re.sub(r"    const path1Runs = \[.*?\n    \];", path1_block, html, count=1, flags=re.S)

    manifest = json.loads(Path(args.global_manifest).read_text(encoding="utf-8"))
    global_rows = []
    for view in manifest["views"]:
        relative = view["path"]
        if relative.startswith("raw/"):
            scope, method, space = "global_raw", relative.split("/")[-1], None
        elif relative.startswith("umap"):
            space, method = relative.split("/")
            scope = "global_umap"
        else:
            scope, method, space = "zhicheng_umap_hdbscan", "hdbscan", None
        run_id = "neutral_" + relative.replace("/", "_")
        space_attr = f', space: "{space}"' if space else ""
        stability = f'{float(view["stability_ari"]):.3f}'.lstrip("0")
        params = f'{view["cluster_count"]} clusters · {view["noise_count"]} noise · stability {stability}'
        global_rows.append(
            f'      {{ id: "{run_id}", view: "{scope}"{space_attr}, method: "{method}", label: "{view["title"]}", papers: "{paper_count}", params: "{params}", explorer: "explorer/global_comparison_neutral/{relative}/paper_explorer.html", summary: "explorer/global_comparison_neutral/{relative}/cluster_summary.md" }}'
        )
    global_block = "    const globalRuns = [\n" + ",\n".join(global_rows) + "\n    ];"
    html = re.sub(r"    const globalRuns = \[.*?\n    \];", global_block, html, count=1, flags=re.S)
    html = re.sub(r'const buildId = "[^"]+";', f'const buildId = "{paper_count}-full-rebuild-20260811";', html, count=1)
    path.write_text(html, encoding="utf-8")
    print(f"Updated {path}: {paper_count} papers, {total_clusters} keyword clusters")


if __name__ == "__main__":
    main()
