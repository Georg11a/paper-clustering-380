#!/usr/bin/env python3
"""Build only Page 3 (fixed UMAP-10D + HDBSCAN) without touching Pages 1-2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import umap
from sklearn.preprocessing import normalize

import build_global_comparison_explorers as global_views
from refine_page3_cluster_summaries import (
    PAYLOAD_RE,
    refine_payload,
    update_csv,
    write_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-paper-count", type=int, required=True)
    parser.add_argument("--discussion-metadata", default="")
    args = parser.parse_args()

    global_views.METADATA_PATH = Path(args.metadata)
    papers = global_views.load_neutral_papers(Path(args.input)).fillna("")
    vectors = normalize(np.load(args.embeddings), norm="l2")
    if (
        len(papers) != args.expected_paper_count
        or papers["paper_id"].nunique() != args.expected_paper_count
        or vectors.shape != (args.expected_paper_count, 1024)
    ):
        raise ValueError("Page 3 papers and BGE-M3 vectors are not aligned")

    display_coordinates = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.08,
        metric="cosine",
        random_state=global_views.SEED,
        n_jobs=1,
    ).fit_transform(vectors)
    reduced10 = umap.UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=global_views.SEED,
        n_jobs=1,
    ).fit_transform(vectors)

    config = "hdbscan_mcs8_ms1"
    labels = global_views.fit_selected(config, reduced10, "euclidean")
    metrics = global_views.partition_metrics(vectors, labels)
    metrics["stability_ari"] = global_views.subsample_stability(
        config, reduced10, "euclidean"
    )
    discussion = global_views.load_discussion_metadata(
        Path(args.discussion_metadata) if args.discussion_metadata else None
    )
    out = Path(args.out)
    view = global_views.export_view(
        out,
        "zhicheng_umap_hdbscan",
        "Zhicheng workflow · UMAP 10D + HDBSCAN",
        config,
        metrics,
        labels,
        papers,
        vectors,
        display_coordinates,
        discussion,
    )
    explorer_dir = out / "zhicheng_umap_hdbscan"
    explorer_path = explorer_dir / "paper_explorer.html"
    page = explorer_path.read_text(encoding="utf-8")
    payload_match = PAYLOAD_RE.search(page)
    if not payload_match:
        raise ValueError(f"No explorer payload found in {explorer_path}")
    payload = refine_payload(json.loads(payload_match.group(2)))
    explorer_path.write_text(
        page[: payload_match.start(2)]
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + page[payload_match.end(2) :],
        encoding="utf-8",
    )
    update_csv(explorer_dir / "clustered_papers.csv", payload)
    write_markdown(explorer_dir / "cluster_summary.md", payload)
    manifest = {
        "scope": "Page 3 only; Page 1 and Page 2 remain frozen at 459 papers",
        "paper_count": len(papers),
        "input": "Drive-synced R_cent neutral chunks, assembled as 459 frozen + 49 new",
        "embeddings": "BGE-M3 1024D paper vectors, assembled as 459 frozen + 49 new",
        "metadata": "508-paper Page 3 metadata join; display-only fields added after clustering",
        "configuration": {
            "representation": "R_cent; 13 chunks per paper; BGE-M3 mean pooling",
            "umap_n_components": 10,
            "umap_n_neighbors": 15,
            "umap_min_dist": 0.0,
            "umap_metric": "cosine",
            "hdbscan_min_cluster_size": 8,
            "hdbscan_min_samples": 1,
            "seed": global_views.SEED,
            "noise_policy": "retain label -1; never reassign",
        },
        "view": view,
    }
    manifest_path = out / "zhicheng_umap_hdbscan" / "page3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(view, indent=2))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
