#!/usr/bin/env python3
"""Install a verified Page 3 build without modifying archived Page 1–2 views."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--drive-file-ids", required=True)
    parser.add_argument("--drive-file-ids-target", required=True)
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    target_dir = Path(args.target_dir)
    required = [
        "paper_explorer.html",
        "clustered_papers.csv",
        "cluster_summary.md",
        "page3_manifest.json",
    ]
    for filename in required:
        source = build_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in required:
        shutil.copy2(build_dir / filename, target_dir / filename)

    shutil.copy2(args.drive_file_ids, args.drive_file_ids_target)
    manifest = json.loads((target_dir / "page3_manifest.json").read_text())
    print(
        f"Installed Page 3: {manifest['paper_count']} papers at {target_dir}; "
        "archived Page 1–2 views unchanged"
    )


if __name__ == "__main__":
    main()
