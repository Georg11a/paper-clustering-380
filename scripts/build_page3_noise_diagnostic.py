#!/usr/bin/env python3
"""Compare the original 282-paper Page 3 noise with the expanded corpus."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OLD_COMMIT = "ce39ca3"
RELATIVE = "docs/explorer/global_comparison_neutral/zhicheng_umap_hdbscan/clustered_papers.csv"


def main() -> None:
    old_bytes = subprocess.check_output(["git", "show", f"{OLD_COMMIT}:{RELATIVE}"], cwd=REPO)
    old = pd.read_csv(io.BytesIO(old_bytes))
    current = pd.read_csv(REPO / RELATIVE)
    old_status = old.set_index("paper_id")["cluster"].map(
        lambda value: "noise" if int(value) == -1 else "clustered"
    )
    current_status = current["cluster"].map(
        lambda value: "noise" if int(value) == -1 else "clustered"
    )
    diagnostic = current[["paper_id", "title", "keyword", "cluster"]].copy()
    diagnostic["old_282_status"] = diagnostic["paper_id"].map(old_status).fillna("new_paper")
    diagnostic["expanded_459_status"] = current_status
    diagnostic["transition"] = diagnostic["old_282_status"] + " -> " + diagnostic["expanded_459_status"]
    diagnostic["diagnostic_interpretation"] = diagnostic["transition"].map(
        {
            "noise -> noise": "persistent weak affinity across corpus sizes",
            "noise -> clustered": "gained a density-connected neighborhood after expansion",
            "clustered -> noise": "became peripheral after global UMAP refit and density re-estimation",
            "clustered -> clustered": "retained density-connected membership (cluster number may change)",
            "new_paper -> noise": "new paper lies in a sparse, niche, or boundary region under fixed HDBSCAN settings",
            "new_paper -> clustered": "new paper joined a density-connected region",
        }
    )
    output = REPO / "output/page3_noise_diagnostic_459.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostic.to_csv(output, index=False)

    summary = diagnostic.groupby("transition").size().rename("paper_count").reset_index()
    summary.to_csv(REPO / "output/page3_noise_transition_summary_459.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Old: {len(old)} papers, {(old.cluster == -1).sum()} noise, {old.loc[old.cluster != -1, 'cluster'].nunique()} clusters")
    print(f"New: {len(current)} papers, {(current.cluster == -1).sum()} noise, {current.loc[current.cluster != -1, 'cluster'].nunique()} clusters")


if __name__ == "__main__":
    main()
