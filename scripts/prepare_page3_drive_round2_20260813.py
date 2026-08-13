#!/usr/bin/env python3
"""Stage the verified 15-PDF Page 3 increment and write an upload manifest."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


PAPER_IDS = [
    "75ffc50a1708",
    "f0eab2f851e8",
    "1e3836bc80cb",
    "f652405808cb",
    "6d339b25d458",
    "19c286f4e44f",
    "a60240423604",
    "24bfa2182839",
    "a0bac1579403",
    "721f009574ce",
    "4934044e5959",
    "49c09c48cbf2",
    "a325f90a8913",
    "4cce12210dba",
    "a8a02df6e7cc",
]

DRIVE_FILE_IDS = {
    "75ffc50a1708": "1xg-fEXa_NRcODFkXF1kiPnjeC6FriMaO",
    "f0eab2f851e8": "1Y8hn0bYrn5NCxZhiq6NxnmvkobQJc6dT",
    "1e3836bc80cb": "1DXOyuHOu-BmHejJrceXXUapCGhkrTapk",
    "f652405808cb": "1oCqszG0tbdCCpRbl8GEhWyc_1L0E-HKS",
    "6d339b25d458": "11gBBYeSfGQH6wDNym1A04IaQdrLpy_-c",
    "19c286f4e44f": "1Zgy9hYHfb7IpnYuMAXsF6LkLQYrpSUNM",
    "a60240423604": "1e5qWXVUXaPO2nX9d9bDHiy38KToSMno1",
    "24bfa2182839": "1APWA0fTlTHsxGc5M0SdBp8mTr2VSSutz",
    "a0bac1579403": "1pmWRTb6EQZ1IJkyHNlcg95J_bFRW_BD_",
    "721f009574ce": "1JsWP97hvPkxBNLrPukAu0yZdZ4bjTZl9",
    "4934044e5959": "1YWUrF0sGoabxlWl-N6mpuisGkG3KGvCm",
    "49c09c48cbf2": "11Fq8Mf_-79q6MJF2dVx7LLhiqNBkMsZN",
    "a325f90a8913": "1yEmTHWTvw2GUeTBGe_i62QClNb4oxQjb",
    "4cce12210dba": "1JlWozyJBEY_c91lkBV8CKzno_V7JaBnk",
    "a8a02df6e7cc": "1CEo_gKp640BMlfDedC6P-DtjIlPp9iIv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--downloads", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    tracker = pd.read_csv(args.tracker).fillna("")
    selected = tracker[tracker["paper_id"].astype(str).isin(PAPER_IDS)].copy()
    if len(selected) != 15 or selected["paper_id"].nunique() != 15:
        raise ValueError("Expected exactly 15 unique tracker records")
    selected = selected.set_index(selected["paper_id"].astype(str)).loc[PAPER_IDS]

    downloads = Path(args.downloads)
    outdir = Path(args.outdir)
    upload_dir = outdir / "upload_ready"
    upload_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for paper_id, row in selected.iterrows():
        source = downloads / f"{paper_id}.pdf"
        if not source.is_file():
            raise FileNotFoundError(source)
        target = upload_dir / source.name
        shutil.copy2(source, target)
        reader = PdfReader(str(target))
        rows.append(
            {
                "paper_id": paper_id,
                "title": row.get("title", ""),
                "found_status": row.get("Found", ""),
                "source_path": str(source),
                "upload_filename": target.name,
                "bytes": target.stat().st_size,
                "page_count": len(reader.pages),
                "sha256": sha256(target),
                "drive_file_id": DRIVE_FILE_IDS[paper_id],
                "drive_view_url": f"https://drive.google.com/file/d/{DRIVE_FILE_IDS[paper_id]}/view",
                "drive_download_url": f"https://drive.usercontent.google.com/download?id={DRIVE_FILE_IDS[paper_id]}&export=download&confirm=t",
                "drive_cache_path": "",
                "drive_sync_status": "pending",
                "drive_cache_sha256": "",
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "drive_upload_manifest.csv", index=False)
    selected.reset_index(drop=True).to_csv(outdir / "page3_analysis_candidates.csv", index=False)
    print(f"Staged {len(rows)} PDFs in {upload_dir}")
    print(f"Wrote {outdir / 'drive_upload_manifest.csv'}")


if __name__ == "__main__":
    main()
