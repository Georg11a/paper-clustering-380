# Page 3 Google Drive source of truth

Page 3 can use Google Drive as its sole PDF input at build time. The public
GitHub Pages site remains static: it cannot securely enumerate a shared Drive
folder on every page load. Instead, every Page 3 rebuild begins with a fresh,
authenticated Drive snapshot and refuses to use historical local corpora as a
fallback.

## One-time authentication

`rclone` is already installed on the development machine. Configure one
read-only Google Drive remote:

```bash
rclone config
```

Use a descriptive remote name such as `final-advancing-drive` and select the
Google Drive provider. Browser authorization is required once; the resulting
credential remains local to the development machine.

## Create a strict Drive snapshot

```bash
.venv/bin/python scripts/sync_page3_from_drive.py \
  --remote "final-advancing-drive:Design knowledge/Final_Advancing_PDFs" \
  --scope-metadata ../page3_drive_increment_20260813_round2/neutral_523/page3_metadata_523.csv \
  --drive-exclusions data/page3_drive_exclusions.csv \
  --snapshot-dir ../page3_drive_snapshot_YYYYMMDD \
  --expected-paper-count 523
```

The command first lists the live Drive folder and writes a scope comparison.
It stops before downloading if even one of the 523 Page 3 IDs is absent. When
the check passes, it downloads the folder into a new immutable snapshot and
materializes exactly the 523 eligible PDFs in `page3_analysis_pdfs/`. Drive-only
archive or excluded records remain outside the Page 3 input.

## Continue the existing pipeline

Use the snapshot's `page3_analysis_pdfs/` directory as `--pdf-dir` for
`build_global_canonical_text.py`. Then run the existing R-centroid passage
selection, BGE-M3 embedding, UMAP-10D, and HDBSCAN scripts. Pass the snapshot's
`page3_drive_file_ids.json` to `build_page3_zhicheng_only.py` so records without
a DOI or publisher URL link directly to the corresponding Drive PDF.

The deployed detail panel uses one link per paper in this order: DOI first,
publisher URL second, and the verified direct Drive PDF URL only when both
metadata fields are empty. The checked-in `data/page3_drive_file_ids.json`
contains all 523 eligible Page 3 paper IDs; folder-level fallback links are not
expected in a verified build.

This changes the source boundary without changing the clustering method:

```text
live Google Drive folder
-> strict 523-ID snapshot
-> PDF text extraction and paragraph splitting
-> 12 centroid-selected passages plus title/abstract
-> BGE-M3 paper embeddings
-> UMAP 10D
-> HDBSCAN (minimum cluster size 8, minimum samples 1)
-> static Page 3 build
```
