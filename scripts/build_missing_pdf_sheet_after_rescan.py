#!/usr/bin/env python3
"""Create a deduplicated missing-PDF sheet after the 2026-08-11 Downloads audit."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ACCEPTED = Path("/Users/baiyixin/Downloads/accepted_papers.csv")
CURRENT = REPO / "data/final_advancing_list_expanded_459.csv"
OUTPUT = REPO / "output/missing_pdf_after_rescan_20260811.csv"
AUDIT = REPO / "output/pdf_inventory_status_after_rescan_20260811.csv"
ID_RE = re.compile(r"^[0-9a-f]{12}$", re.I)

# Title variants confirmed by title/author/DOI review. Exact normalized-title
# duplicates are unioned automatically below.
CONFIRMED_SAME_PAPER = [
    ("0050f73e4373", "4c8bdfbd1309"),
    ("94bc88b88ae6", "b437b859b693"),
    ("4a9c96b6c482", "132626dd3ef8"),
    ("1aecb7940a63", "7e16109e6183"),
    ("b8507dbe97ca", "a0d1057f0acb"),
    ("8abe5ed83bac", "74beefd3ed50"),
    ("9eda131748eb", "a14565cde772"),
    ("0a373ca87142", "b1e2f4329820"),
]

LOCAL_NEW_COMPLETE = {
    "1aecb7940a63",
    "38b608e72e2d",
    "781628bf9a48",
    "a285258b5545",
    "55dc4ef842a6",
}
LOCAL_INCOMPLETE = {"23d63cb34b3e"}


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def main() -> None:
    raw = pd.read_csv(ACCEPTED, dtype=str).fillna("")
    raw = raw[raw["paper_id"].map(lambda value: bool(ID_RE.fullmatch(value)))].copy()
    raw = raw.drop_duplicates("paper_id", keep="first").reset_index(drop=True)
    ids = raw["paper_id"].astype(str).tolist()
    parent = {paper_id: paper_id for paper_id in ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        if left not in parent or right not in parent:
            raise RuntimeError(f"Unknown dedup ID: {left}, {right}")
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    raw["normalized_title"] = raw["title"].map(normalize_title)
    for _, group in raw.groupby("normalized_title", sort=False):
        group_ids = group["paper_id"].astype(str).tolist()
        for paper_id in group_ids[1:]:
            union(group_ids[0], paper_id)
    for left, right in CONFIRMED_SAME_PAPER:
        union(left, right)

    raw["paper_group"] = raw["paper_id"].map(find)
    current_ids = set(pd.read_csv(CURRENT, dtype=str)["paper_id"].astype(str))
    acquired_ids = current_ids | LOCAL_NEW_COMPLETE

    audit_rows = []
    missing_rows = []
    for _, group in raw.groupby("paper_group", sort=False):
        group_ids = group["paper_id"].astype(str).tolist()
        acquired = sorted(set(group_ids) & acquired_ids)
        incomplete = sorted(set(group_ids) & LOCAL_INCOMPLETE)
        if acquired:
            status = "downloaded"
        elif incomplete:
            status = "missing_full_pdf_link_stub_only"
        else:
            status = "missing_pdf"

        scores = group.apply(
            lambda row: sum(bool(str(value).strip()) for value in row), axis=1
        )
        representative = group.loc[scores.idxmax()].copy()
        audit_rows.append(
            {
                "paper_group": find(group_ids[0]),
                "representative_paper_id": representative["paper_id"],
                "representative_title": representative["title"],
                "status": status,
                "accepted_record_ids": ";".join(group_ids),
                "acquired_record_ids": ";".join(acquired),
                "local_incomplete_record_ids": ";".join(incomplete),
                "accepted_record_count": len(group_ids),
            }
        )
        if not acquired:
            record = representative.drop(labels=["normalized_title", "paper_group"]).to_dict()
            record["status"] = "missing_pdf"
            missing_rows.append(record)

    audit = pd.DataFrame(audit_rows).sort_values(
        ["status", "representative_title"], kind="stable"
    )
    missing = pd.DataFrame(missing_rows)
    source_columns = [column for column in raw.columns if column not in {"normalized_title", "paper_group"}]
    output_columns = source_columns[:2] + ["status"] + source_columns[2:]
    missing = missing[output_columns].sort_values(["keyword", "title"], kind="stable")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(OUTPUT, index=False, encoding="utf-8")
    audit.to_csv(AUDIT, index=False, encoding="utf-8")
    print(f"Accepted record IDs: {len(raw)}")
    print(f"Deduplicated paper groups: {len(audit)}")
    print(audit["status"].value_counts().to_string())
    print(f"Wrote {OUTPUT}: {len(missing)} missing independent papers")
    print(f"Wrote {AUDIT}")


if __name__ == "__main__":
    main()
