#!/usr/bin/env python3
"""Create a strict Page 3 PDF snapshot whose only source is Google Drive.

This script uses an authenticated rclone remote. It never falls back to Downloads
or any historical local corpus. A snapshot is created only when every Page 3
paper ID has a matching ``<paper_id>.pdf`` file in the requested Drive folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def run(command: list[str], capture: bool = False) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout if capture else ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remote",
        required=True,
        help="Configured rclone folder, for example gdrive:Design knowledge/Final_Advancing_PDFs",
    )
    parser.add_argument("--scope-metadata", required=True)
    parser.add_argument("--drive-exclusions", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--expected-paper-count", type=int, required=True)
    args = parser.parse_args()

    if not shutil.which("rclone"):
        raise RuntimeError("rclone is required but was not found")

    scope = pd.read_csv(args.scope_metadata, dtype=str).fillna("")
    if scope["paper_id"].duplicated().any():
        raise ValueError("Scope metadata contains duplicate paper IDs")
    scope_ids = set(scope["paper_id"])
    if len(scope_ids) != args.expected_paper_count:
        raise ValueError(
            f"Expected {args.expected_paper_count} scope IDs, found {len(scope_ids)}"
        )

    exclusions = pd.read_csv(args.drive_exclusions, dtype=str).fillna("")
    if exclusions["paper_id"].duplicated().any():
        raise ValueError("Drive exclusion table contains duplicate paper IDs")
    exclusion_ids = set(exclusions["paper_id"])

    listing = json.loads(
        run(
            [
                "rclone",
                "lsjson",
                "--files-only",
                "--include",
                "*.pdf",
                args.remote,
            ],
            capture=True,
        )
    )
    records = []
    for record in listing:
        filename = str(record.get("Name") or record.get("Path") or "")
        records.append(
            {
                "paper_id": Path(filename).stem,
                "name": filename,
                "drive_file_id": record.get("ID", ""),
                "bytes": record.get("Size", ""),
                "modified_time": record.get("ModTime", ""),
            }
        )
    inventory = pd.DataFrame(records)
    if inventory.empty:
        raise RuntimeError("No PDFs were returned by the Drive folder")
    if inventory["paper_id"].duplicated().any():
        duplicates = inventory.loc[inventory["paper_id"].duplicated(), "paper_id"].tolist()
        raise ValueError(f"Duplicate paper-ID filenames in Drive: {duplicates[:20]}")

    drive_ids = set(inventory["paper_id"])
    missing = sorted(scope_ids - drive_ids)
    extras = sorted(drive_ids - scope_ids)
    unknown_extras = sorted(set(extras) - exclusion_ids)
    snapshot = Path(args.snapshot_dir)
    snapshot.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(snapshot / "drive_inventory.csv", index=False)
    (snapshot / "scope_comparison.json").write_text(
        json.dumps(
            {
                "scope_count": len(scope_ids),
                "drive_pdf_count": len(drive_ids),
                "matched_count": len(scope_ids & drive_ids),
                "missing_from_drive": missing,
                "drive_only": extras,
                "unknown_drive_only": unknown_extras,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if missing:
        raise RuntimeError(
            f"Drive is not complete: {len(missing)} Page 3 PDFs are missing. "
            f"See {snapshot / 'scope_comparison.json'}"
        )
    if unknown_extras:
        raise RuntimeError(
            f"Drive contains {len(unknown_extras)} undocumented Page 3 exclusions. "
            f"See {snapshot / 'scope_comparison.json'}"
        )

    all_pdfs = snapshot / "all_drive_pdfs"
    analysis_pdfs = snapshot / "page3_analysis_pdfs"
    if all_pdfs.exists() or analysis_pdfs.exists():
        raise FileExistsError(
            "Snapshot PDF directories already exist; choose a new snapshot directory"
        )
    all_pdfs.mkdir()
    analysis_pdfs.mkdir()
    run(
        [
            "rclone",
            "copy",
            "--include",
            "*.pdf",
            "--checksum",
            args.remote,
            str(all_pdfs),
        ]
    )

    manifest_rows = []
    by_id = inventory.set_index("paper_id").to_dict("index")
    for paper_id in scope["paper_id"]:
        source = all_pdfs / f"{paper_id}.pdf"
        if not source.is_file() or source.read_bytes()[:4] != b"%PDF":
            raise RuntimeError(f"Drive download is missing or invalid: {source}")
        destination = analysis_pdfs / source.name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        manifest_rows.append(
            {
                "paper_id": paper_id,
                "title": scope.loc[scope["paper_id"].eq(paper_id), "title"].iloc[0],
                "drive_file_id": by_id[paper_id].get("drive_file_id", ""),
                "drive_view_url": (
                    "https://drive.google.com/file/d/"
                    f"{by_id[paper_id].get('drive_file_id', '')}/view"
                ),
                "snapshot_pdf_path": str(destination),
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(snapshot / "page3_drive_source_manifest.csv", index=False)
    file_ids = dict(zip(manifest["paper_id"], manifest["drive_file_id"]))
    (snapshot / "page3_drive_file_ids.json").write_text(
        json.dumps(dict(sorted(file_ids.items())), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Created Drive-only snapshot with {len(manifest)} Page 3 PDFs at "
        f"{analysis_pdfs}"
    )
    print(f"Drive-only files retained outside Page 3: {len(extras)}")


if __name__ == "__main__":
    main()
