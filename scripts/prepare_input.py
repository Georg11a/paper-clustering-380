#!/usr/bin/env python3
"""Validate and prepare the final advancing-paper CSV for clustering."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["paper_id", "title", "abstract", "keyword"]
RECOMMENDED_COLUMNS = ["authors", "doi", "year", "venue", "url", "fulltext_flag", "primary_reason", "val_reason"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/final_advancing_list.csv")
    parser.add_argument("--output", default="data/final_advancing_list_prepared.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input).fillna("")
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["keyword"] = df["keyword"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df = df.drop_duplicates(subset=["paper_id"]).copy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Wrote {output}: {len(df)} papers")
    print("Keyword counts:")
    print(df["keyword"].value_counts().to_string())
    missing_recommended = [column for column in RECOMMENDED_COLUMNS if column not in df.columns]
    if missing_recommended:
        print(f"Missing recommended metadata columns: {missing_recommended}")


if __name__ == "__main__":
    main()
