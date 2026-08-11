#!/usr/bin/env python3
"""Export shareable Page 1/2/3 assignment CSVs for the expanded corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "docs/explorer/global_comparison_neutral"
OUTPUT = REPO / "outputs/expanded_457_20260811"

RUNS = [
    ("page1", "raw", "kmeans", "raw/kmeans"),
    ("page1", "raw", "dbscan", "raw/dbscan"),
    ("page1", "raw", "hdbscan", "raw/hdbscan"),
    ("page2", "umap5", "kmeans", "umap5/kmeans"),
    ("page2", "umap5", "dbscan", "umap5/dbscan"),
    ("page2", "umap5", "hdbscan", "umap5/hdbscan"),
    ("page2", "umap10", "kmeans", "umap10/kmeans"),
    ("page2", "umap10", "dbscan", "umap10/dbscan"),
    ("page2", "umap10", "hdbscan", "umap10/hdbscan"),
    ("page3", "umap10_fixed", "hdbscan", "zhicheng_umap_hdbscan"),
]


def main() -> None:
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    config_by_path = {row["path"]: row["config"] for row in manifest["views"]}
    frames = []
    for page, space, algorithm, relative in RUNS:
        path = SOURCE / relative / "clustered_papers.csv"
        frame = pd.read_csv(path).fillna("")
        if len(frame) != 457 or frame["paper_id"].nunique() != 457:
            raise RuntimeError(f"Unexpected assignment size for {relative}: {frame.shape}")
        frame.insert(0, "analysis_page", page)
        frame.insert(1, "clustering_space", space)
        frame.insert(2, "algorithm", algorithm)
        frame.insert(3, "configuration", config_by_path[relative])
        frames.append(frame)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_views = pd.concat(frames, ignore_index=True)
    page1 = pd.concat([frame for frame in frames if frame.iloc[0]["analysis_page"] == "page1"], ignore_index=True)
    page2 = pd.concat([frame for frame in frames if frame.iloc[0]["analysis_page"] == "page2"], ignore_index=True)
    page3 = frames[-1]

    all_views.to_csv(OUTPUT / "clustering_assignments_all_pages.csv", index=False)
    page1.to_csv(OUTPUT / "page1_raw_assignments.csv", index=False)
    page2.to_csv(OUTPUT / "page2_umap_assignments.csv", index=False)
    page3.to_csv(OUTPUT / "page3_zhicheng_umap_hdbscan_assignments.csv", index=False)
    pd.DataFrame(manifest["views"]).to_csv(OUTPUT / "clustering_run_summary.csv", index=False)

    print(f"Wrote {OUTPUT / 'clustering_assignments_all_pages.csv'}: {len(all_views)} rows")
    print(f"Page 1 rows: {len(page1)}; Page 2 rows: {len(page2)}; Page 3 rows: {len(page3)}")


if __name__ == "__main__":
    main()
