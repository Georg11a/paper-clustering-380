#!/usr/bin/env python3
"""Install verified Drive file IDs and refresh links in the deployed Page 3 HTML."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1KyP7NCwToMY-mGPLCWa4PnKM5e2ZIAfw"
)
LINKS_RE = re.compile(
    r"      const doi = .*?\n"
    r"      const url = .*?\n"
    r"      const driveFileIds = \{.*?\};\n"
    r"      const driveFileId = .*?\n"
    r"      const drive = .*?;",
    flags=re.DOTALL,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def inventory_lookup(rows: list[dict[str, str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in rows:
        filename = (row.get("name") or row.get("Name") or "").strip()
        file_id = (row.get("id") or row.get("ID") or row.get("drive_file_id") or "").strip()
        paper_id = (row.get("paper_id") or Path(filename).stem).strip()
        if not paper_id or not file_id:
            continue
        if paper_id in lookup:
            raise ValueError(f"Duplicate Drive paper ID: {paper_id}")
        lookup[paper_id] = file_id
    return lookup


def link_renderer(file_ids: dict[str, str]) -> str:
    compact_ids = json.dumps(file_ids, separators=(",", ":"), sort_keys=True)
    return (
        '      const doi = p.doi ? `<a class="paper-link" '
        'href="https://doi.org/${escapeAttr(p.doi)}" target="_blank" '
        'rel="noopener noreferrer">DOI: ${escapeHtml(p.doi)}</a>` : \'\';\n'
        '      const url = p.url ? `<a class="paper-link" href="${escapeAttr(p.url)}" '
        'target="_blank" rel="noopener noreferrer">Open paper URL</a>` : \'\';\n'
        f"      const driveFileIds = {compact_ids};\n"
        "      const driveFileId = driveFileIds[p.paper_id];\n"
        '      const drive = !p.doi && !p.url ? `<a class="paper-link" '
        'href="${driveFileId ? `https://drive.google.com/file/d/${escapeAttr(driveFileId)}/view` : '
        "`"
        + DRIVE_FOLDER_URL
        + '`}" target="_blank" rel="noopener noreferrer">'
        '${driveFileId ? `Open ${escapeHtml(p.paper_id)}.pdf in Google Drive` : '
        "`Open shared Google Drive folder`}</a>` : '';"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--exclusions", required=True)
    parser.add_argument("--file-ids-output", required=True)
    parser.add_argument("--html", required=True)
    parser.add_argument("--expected-paper-count", type=int, required=True)
    args = parser.parse_args()

    inventory = inventory_lookup(read_rows(Path(args.inventory)))
    metadata = read_rows(Path(args.metadata))
    exclusions = read_rows(Path(args.exclusions))
    scope_ids = {row["paper_id"].strip() for row in metadata}
    exclusion_ids = {row["paper_id"].strip() for row in exclusions}
    if len(metadata) != args.expected_paper_count or len(scope_ids) != args.expected_paper_count:
        raise ValueError("Page 3 metadata count or uniqueness check failed")
    missing = sorted(scope_ids - set(inventory))
    drive_only = sorted(set(inventory) - scope_ids)
    undocumented = sorted(set(drive_only) - exclusion_ids)
    if missing:
        raise ValueError(f"Drive inventory is missing {len(missing)} Page 3 PDFs: {missing[:10]}")
    if undocumented:
        raise ValueError(f"Drive has undocumented Page 3 exclusions: {undocumented}")

    page3_ids = {paper_id: inventory[paper_id] for paper_id in sorted(scope_ids)}
    output = Path(args.file_ids_output)
    output.write_text(json.dumps(page3_ids, indent=2) + "\n", encoding="utf-8")

    html_path = Path(args.html)
    page = html_path.read_text(encoding="utf-8")
    updated, replacements = LINKS_RE.subn(link_renderer(page3_ids), page, count=1)
    if replacements != 1:
        raise ValueError("Could not locate exactly one Page 3 link-renderer block")
    if "${doi || url || drive}" not in updated:
        raise ValueError("Page 3 does not use the required DOI → URL → Drive priority")
    html_path.write_text(updated, encoding="utf-8")
    print(
        f"Installed {len(page3_ids)} direct Drive links; "
        f"{len(drive_only)} documented Drive-only exclusions remain outside Page 3"
    )


if __name__ == "__main__":
    main()
