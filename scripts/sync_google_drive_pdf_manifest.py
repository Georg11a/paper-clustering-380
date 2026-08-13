#!/usr/bin/env python3
"""Download paper PDFs from Google Drive using a recorded file-ID manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd


URL = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload-manifest", required=True)
    parser.add_argument("--file-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolved-manifest", required=True)
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        help="Optional corpus_role to download; repeat for multiple roles.",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.upload_manifest).fillna("")
    if args.role:
        manifest = manifest[manifest["corpus_role"].isin(args.role)].copy()
    file_ids = {
        str(row["name"]).removesuffix(".pdf"): str(row["id"])
        for row in json.loads(Path(args.file_ids).read_text(encoding="utf-8"))
    }
    missing = sorted(set(manifest["paper_id"].astype(str)) - set(file_ids))
    if missing:
        raise ValueError(f"Missing Drive file IDs: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = []
    for _, row in manifest.iterrows():
        record = row.to_dict()
        paper_id = str(record["paper_id"])
        file_id = file_ids[paper_id]
        url = URL.format(file_id=file_id)
        destination = output_dir / f"{paper_id}.pdf"
        expected_hash = str(record.get("sha256", ""))
        status = "cached"
        if not destination.exists() or sha256(destination) != expected_hash:
            temporary = destination.with_suffix(".pdf.part")
            with urllib.request.urlopen(url, timeout=180) as response, temporary.open(
                "wb"
            ) as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
            if temporary.stat().st_size < 5 or temporary.read_bytes()[:4] != b"%PDF":
                raise ValueError(f"Drive response for {paper_id} is not a PDF")
            if expected_hash and sha256(temporary) != expected_hash:
                raise ValueError(f"SHA-256 mismatch for {paper_id}")
            os.replace(temporary, destination)
            status = "downloaded"
        record.update(
            {
                "drive_file_id": file_id,
                "drive_download_url": url,
                "drive_cache_path": str(destination),
                "drive_sync_status": status,
                "drive_cache_sha256": sha256(destination),
            }
        )
        resolved.append(record)
        print(f"{paper_id}: {status}", flush=True)

    resolved_frame = pd.DataFrame(resolved)
    Path(args.resolved_manifest).parent.mkdir(parents=True, exist_ok=True)
    resolved_frame.to_csv(args.resolved_manifest, index=False)
    print(f"Wrote {args.resolved_manifest}")


if __name__ == "__main__":
    main()
