# Global neutral clustering comparison — 282 papers

This is the meeting-requested Path 1 comparison. All papers are analyzed
together. The clustering input contains no retrieval keyword, previous cluster
assignment, or ground-truth label.

```text
R_cent neutral chunks (3,597 chunks / 282 papers)
→ frozen mean-pooled BGE-M3 1024D vectors
→ raw 1024D / UMAP 5D / UMAP 10D
→ K-Means / DBSCAN / HDBSCAN
→ the same 2D UMAP coordinates for visualization only
```

Run:

```bash
NUMBA_CACHE_DIR=/tmp/design_knowledge_numba_cache \
.venv/bin/python scripts/build_global_comparison_explorers.py \
  --input outputs/neutral/stage01/neutral_inputs/R_cent_neutral_chunks.csv \
  --embeddings outputs/neutral/stage01/emb/R_cent.npy \
  --out docs/explorer/global_comparison_neutral \
  --expected-paper-count 282
```

Open `docs/index.html`. It retains the existing explorer UI and exposes three
analysis choices in the first dropdown:

1. all papers together in raw 1024D;
2. raw versus UMAP-5D versus UMAP-10D for all three algorithms;
3. dedicated UMAP + HDBSCAN inspection for Zhicheng's shared configuration.

The third tab includes 216 fixed-seed sensitivity configurations spanning
UMAP dimensions 5/10, `n_neighbors` 5/10/15/30, HDBSCAN
`min_cluster_size` 5/8/10/12/15/20, and valid `min_samples` values. ARI against
the shared configuration is a sensitivity diagnostic, not accuracy.

Density noise remains label `-1`. It is displayed as weak affinity and is never
assigned to the nearest cluster. Silhouette is always measured in the frozen
raw 1024D cosine space after excluding noise. Stability is mean adjusted Rand
index over the overlapping papers in repeated 80% subsamples, in the stated
clustering space.

The explorer also joins the previously extracted Discussion metadata from
`data/fulltext_context_confirmed_284_only.csv` after clustering. This provides
an extractive `Discussion Summary` card for 45 of the current 282 papers. The
join is display-only: Discussion text is not used to build embeddings, fit
UMAP, select configurations, or assign clusters. Use `--discussion-metadata ""`
to build the views without these cards.

The selected configurations are representative views, not ground-truth
winners. No external gold labels are assumed. Retrieval keywords can be joined
back after assignment to diagnose whether a result simply reproduces the
sampling frame, but keyword agreement is not clustering accuracy.
