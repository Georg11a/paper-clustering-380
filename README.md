# Design-Knowledge Paper Clustering

Clean, reproducible clustering pipeline for the final advancing-paper set from the design-knowledge literature review.

This repository is the public-facing version of the clustering work: it reads the final screening CSV, clusters papers separately within each design-knowledge keyword, flags small keyword groups, and writes reviewable outputs for downstream citation-graph and reading-prioritization work.

## What This Pipeline Does

- Loads `data/final_advancing_list.csv`.
- Groups papers by the `keyword` column.
- Runs clustering per keyword instead of mixing all papers into one global run.
- Skips keyword groups with too few papers for meaningful clustering, default `n < 10`.
- Uses k-means and HDBSCAN as the main clustering methods.
- Keeps DBSCAN available as an optional baseline because earlier experiments showed it can produce overly broad clusters.
- Generates per-paper cluster assignments, representative-paper rankings, cluster labels, summaries, and HTML explorers.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── final_advancing_list.csv
├── scripts/
│   ├── run_clustering.py
│   ├── cluster_papers.py
│   ├── prepare_input.py
│   └── summarize_clusters.py
├── prompts/
│   ├── context_classification.md
│   ├── cluster_labeling.md
│   └── paragraph_selection.md
├── outputs/
└── docs/
    ├── pipeline_overview.md
    └── meeting_action_items.md
```

## Input CSV

Required columns:

```text
paper_id, title, abstract, keyword
```

Recommended metadata columns:

```text
authors, doi, year, venue, url, fulltext_flag, primary_reason, val_reason
```

The current input is:

```text
data/final_advancing_list.csv
```

This file is treated as the final working version for the current clustering pass.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run The Main Pipeline

```bash
.venv/bin/python scripts/run_clustering.py \
  --input data/final_advancing_list.csv \
  --output outputs/clustering_results.csv
```

By default, this runs:

- `kmeans`
- `hdbscan`

for every keyword group with at least 10 papers.

Small keyword groups are included in the combined output with:

```text
cluster_status = too_few_papers
```

Optional input validation step:

```bash
.venv/bin/python scripts/prepare_input.py \
  --input data/final_advancing_list.csv \
  --output data/final_advancing_list_prepared.csv
```

## Outputs

Main batch outputs:

```text
outputs/clustering_results.csv
outputs/keyword_summary.csv
outputs/run_metadata.json
```

Method-specific outputs:

```text
outputs/by_keyword/<keyword>/<method>/clustered_papers.csv
outputs/by_keyword/<keyword>/<method>/cluster_summary.md
outputs/by_keyword/<keyword>/<method>/paper_explorer.html
outputs/by_keyword/<keyword>/<method>/umap_clusters.html
```

Create a compact Markdown summary:

```bash
.venv/bin/python scripts/summarize_clusters.py \
  --input outputs/clustering_results.csv \
  --keyword-summary outputs/keyword_summary.csv \
  --output outputs/clustering_summary.md
```

`outputs/clustering_results.csv` is the main file to share with downstream code. It preserves metadata and adds:

```text
focus_keyword
cluster_method
cluster_status
cluster
cluster_label_candidate
cluster_summary_candidate
representative_rank
is_representative_top3
```

## Per-Keyword Logic

The pipeline follows the meeting decision that clustering all design-knowledge papers together is too broad. Instead:

```text
Design knowledge -> clustered separately
Design theory -> clustered separately
Design Patterns -> clustered separately
...
```

Default small-sample rule:

```text
if number of papers for a keyword < 10:
    do not cluster
    keep the papers in the output as too_few_papers
else:
    run k-means and HDBSCAN
```

Change the threshold:

```bash
.venv/bin/python scripts/run_clustering.py \
  --input data/final_advancing_list.csv \
  --output outputs/clustering_results.csv \
  --min-papers-to-cluster 6
```

## Optional DBSCAN Baseline

DBSCAN is available for comparison, but it is not the default main result.

```bash
.venv/bin/python scripts/run_clustering.py \
  --input data/final_advancing_list.csv \
  --output outputs/clustering_results_with_dbscan.csv \
  --methods kmeans hdbscan dbscan
```

## Embeddings

The default is a reproducible local baseline:

```text
--embedding tfidf-svd
```

If Ollama is installed and `nomic-embed-text` is available, semantic embeddings can be used:

```bash
.venv/bin/python scripts/run_clustering.py \
  --input data/final_advancing_list.csv \
  --output outputs/clustering_results_ollama.csv \
  --embedding ollama \
  --ollama-model nomic-embed-text
```

## Single-Keyword Debug Run

For inspecting one keyword in detail:

```bash
.venv/bin/python scripts/cluster_papers.py \
  --input data/final_advancing_list.csv \
  --outdir outputs/debug_design_rationale_kmeans \
  --focus-keyword "Design rationale" \
  --method kmeans \
  --embedding tfidf-svd \
  --select-k \
  --k-min 2 \
  --k-max 8
```

K-means uses a dynamic paper-count cap before silhouette selection. This avoids
forcing small keyword groups into too many clusters. For example, a 14-paper
keyword group is only allowed to choose between 2 and 3 clusters, even if the
requested maximum is 8.

## Prompts

Prompt templates used around the pipeline are in `prompts/`:

- `paragraph_selection.md`
- `context_classification.md`
- `cluster_labeling.md`

These prompts document the human/LLM-assisted parts of the workflow without requiring full chat histories.

## Technical Notes

See:

- `docs/pipeline_overview.md`
- `docs/meeting_action_items.md`
