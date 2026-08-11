#!/usr/bin/env python3
"""Create provisional statistical descriptors for expanded keyword clusters."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    papers = pd.read_csv(args.input).fillna("")
    assignments = pd.read_csv(args.assignments).fillna("")
    frame = papers.merge(assignments, on=["paper_id", "keyword", "title"], validate="one_to_one")
    rows: list[dict[str, object]] = []
    for keyword, group in frame.groupby("keyword", sort=True):
        text = (
            group["title"].astype(str)
            + " [SEP] "
            + group["abstract"].astype(str)
            + " [SEP] "
            + group["subdocument"].astype(str)
        )
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=6000)
        matrix = vectorizer.fit_transform(text)
        terms = vectorizer.get_feature_names_out()
        for cluster_id, subset in group.groupby("cluster_id", sort=True):
            positions = [group.index.get_loc(index) for index in subset.index]
            scores = np.asarray(matrix[positions].mean(axis=0)).ravel()
            top_terms = terms[np.argsort(-scores)[:15]]
            rows.append(
                {
                    "keyword": keyword,
                    "cluster_id": cluster_id,
                    "paper_count": len(subset),
                    "assignment_status": subset.iloc[0]["assignment_status"],
                    "interpretation_status": "provisional_pending_cluster_review",
                    "statistical_descriptor_not_final_label": "; ".join(top_terms[:3]),
                    "top_terms": " | ".join(top_terms),
                }
            )
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} provisional cluster descriptors to {args.out}")


if __name__ == "__main__":
    main()
