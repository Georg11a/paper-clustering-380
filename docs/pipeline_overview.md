# Pipeline Overview

This repository is a clean, reproducible version of the paper clustering pipeline for the design-knowledge review.

## Input

The current input is `data/final_advancing_list.csv`, produced by the screening pipeline. The clustering scripts require `paper_id`, `title`, `abstract`, and `keyword`; DOI, URL, year, venue, and full-text flags are preserved when available.

## Processing

1. Load the final advancing-paper CSV.
2. Group records by the `keyword` column.
3. Skip keyword groups below the small-sample threshold, default `n < 10`.
4. For each remaining keyword group, build a text representation from title, abstract, keyword metadata, and selected full-text context when available.
5. Run k-means and HDBSCAN as the primary clustering methods.
6. Keep DBSCAN available as an optional baseline because earlier tests showed it often forms overly broad clusters.
7. Generate per-paper cluster assignments, representative-paper rankings, cluster theme terms, summaries, and HTML explorers.

## Output

The main batch runner writes:

- `outputs/clustering_results.csv`: combined per-paper results across keyword groups and clustering methods.
- `outputs/keyword_summary.csv`: per-keyword paper counts, skip/cluster status, and output directories.
- `outputs/run_metadata.json`: parameters and keyword counts for the run.
- `outputs/by_keyword/<keyword>/<method>/`: method-specific CSV, UMAP, HTML explorer, and cluster summary.

## Human Checkpoints

The pipeline is intentionally human-in-the-loop:

- Confirm the input CSV version.
- Review whether small keyword groups should be clustered or only listed.
- Inspect representative papers and cluster label candidates.
- Decide which method output is most interpretable for each keyword group.
- Align output columns with downstream citation graph and PageRank work.
