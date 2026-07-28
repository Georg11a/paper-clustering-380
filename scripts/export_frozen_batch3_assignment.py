#!/usr/bin/env python3
"""Export one confirmed Batch 3 configuration as the frozen assignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--decision-note", required=True)
    parser.add_argument("--excluded-paper-ids", default="")
    args = parser.parse_args()

    assignment_path = Path(args.assignments)
    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata)

    assignments = pd.read_csv(assignment_path)
    papers = pd.read_csv(input_path)
    if args.config not in assignments:
        raise ValueError(f"Missing assignment column {args.config!r}")
    if assignments["paper_id"].astype(str).duplicated().any():
        raise ValueError("Assignments contain duplicate paper IDs.")
    if papers["paper_id"].astype(str).duplicated().any():
        raise ValueError("Input contains duplicate paper IDs.")

    metadata_columns = [
        column
        for column in ["paper_id", "keyword", "title", "authors", "year", "doi"]
        if column in papers
    ]
    merged = assignments[["paper_id", args.config]].merge(
        papers[metadata_columns],
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    if merged["title"].isna().any():
        raise ValueError("Some assignments have no matching metadata.")
    merged = merged.rename(columns={args.config: "cluster_index"})
    merged["cluster_index"] = merged["cluster_index"].astype(int)
    if (merged["cluster_index"] < 0).any():
        raise ValueError("Frozen complete-assignment configuration contains noise.")
    merged["cluster_id"] = merged["cluster_index"].map(
        lambda value: f"DT-C{value + 1:02d}"
    )
    merged["assignment_method"] = args.config
    output_columns = [
        "paper_id",
        "title",
        *[
            column
            for column in ["authors", "year", "doi", "keyword"]
            if column in merged
        ],
        "cluster_id",
        "cluster_index",
        "assignment_method",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged[output_columns].sort_values(
        ["cluster_index", "paper_id"]
    ).to_csv(output_path, index=False, encoding="utf-8-sig")

    sizes = (
        merged.groupby(["cluster_id", "cluster_index"])
        .size()
        .reset_index(name="paper_count")
        .to_dict("records")
    )
    metadata = {
        "status": "frozen",
        "frozen_date": "2026-07-28",
        "configuration": args.config,
        "paper_count": len(merged),
        "cluster_count": int(merged["cluster_index"].nunique()),
        "cluster_sizes": sizes,
        "decision_note": args.decision_note,
        "excluded_paper_ids": [
            value.strip()
            for value in args.excluded_paper_ids.split(",")
            if value.strip()
        ],
        "source_assignments": str(assignment_path),
        "source_assignments_sha256": sha256(assignment_path),
        "source_input": str(input_path),
        "source_input_sha256": sha256(input_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Froze {len(merged)} papers in {len(sizes)} clusters at "
        f"{output_path}"
    )
    print(pd.DataFrame(sizes).to_string(index=False))


if __name__ == "__main__":
    main()
