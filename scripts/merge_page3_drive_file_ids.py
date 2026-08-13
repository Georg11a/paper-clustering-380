#!/usr/bin/env python3
"""Merge verified Drive file IDs into the Page 3 lookup map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    existing_path = Path(args.existing)
    values = (
        json.loads(existing_path.read_text(encoding="utf-8"))
        if existing_path.exists()
        else {}
    )
    manifest = pd.read_csv(args.manifest).fillna("")
    verified = manifest[manifest["drive_sync_status"].eq("verified_sha256_match")]
    values.update(
        dict(zip(verified["paper_id"].astype(str), verified["drive_file_id"].astype(str)))
    )
    Path(args.output).write_text(
        json.dumps(dict(sorted(values.items())), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(values)} Drive file IDs to {args.output}")


if __name__ == "__main__":
    main()
