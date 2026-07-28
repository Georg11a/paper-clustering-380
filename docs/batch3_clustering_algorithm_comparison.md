# Batch 3 clustering-algorithm comparison

> **Superseded corpus note (2026-07-28):** The earlier 59-paper confirmation
> below is retained as an audit trail. After excluding editorial
> `c1db52f69aae` by corpus-scope decision, the active Design Theory corpus has
> 58 papers. The rerun froze `spectral_knn10_k3` with cluster sizes
> `17/23/18`. The active decision record is
> `outputs/batch3/design_theory_58_confirmatory_20260728/HUMAN_REVIEW_DECISION.md`,
> and the active assignment is
> `outputs/batch3/design_theory_58_confirmatory_20260728/design_theory_frozen_assignment.csv`.

## Prospectively frozen confirmation rule — 2026-07-28

The following decision rule was frozen after exploratory diagnostics had
already been inspected. It is therefore a **prospectively frozen confirmatory
rule**, not a preregistration:

> A configuration must assign at least 95% of papers, contain no cluster with
> fewer than eight papers, obtain a permutation-adjusted Cohesion Ratio
> \(z_\rho > 2\), and obtain mean pairwise ARI across overlapping 80%
> subsamples \(\geq 0.60\).
> Among passing values, choose the smallest \(k\). Human review has veto power
> over a numerically passing solution but cannot promote a numerically failing
> solution.

The provisional default is **Ward HAC on L2-normalized BGE-M3 embeddings,
cut at \(k=3\)**. Ward is preferred provisionally because it provides a nested
dendrogram suitable for survey-level and subsection-level organization,
assigns every paper for downstream per-cluster citation graphs, and is
computationally trivial at \(n=59\). It is not assumed to be more accurate
before the confirmatory comparison.

At the same \(k\), average-linkage cosine HAC, K-means, and two Spectral
graph configurations are robustness controls. Raw HDBSCAN is run once as a
documented density-based negative control.

The Cohesion Ratio is treated as evidence that a partition contains more
within-cluster semantic similarity than a fixed-size random-label null. It is
not used alone to select \(k\), because finer partitions can increase
within-cluster similarity mechanically. Results are therefore reported with a
1,000-permutation null distribution, cosine silhouette, cluster sizes, and
bootstrap stability.

Bootstrap stability is operationalized consistently with the earlier
exploratory runs: fit the same configuration to repeated 80% subsamples and
compute ARI over the overlap of every pair of subsamples. The confirmation
report additionally gives ARI relative to the full-data partition and
best-matching Jaccard stability for each individual cluster. These additional
statistics diagnose the source of instability but do not replace the frozen
pairwise-ARI threshold.

### Confirmation result

The confirmation run used 1,000 fixed-size label permutations and 100 repeated
80% subsamples on the frozen `59 × 1024` BGE-M3 matrix.

- **Ward \(k=3\) did not pass the frozen rule.** It assigned all papers,
  produced sizes `22/26/11`, and had \(z_\rho=8.98\), but its pairwise
  bootstrap ARI was only `0.388`.
- Average-cosine HAC degenerated to sizes `57/1/1` at \(k=3\).
- K-means \(k=3\) produced a three-paper cluster and pairwise bootstrap ARI
  `0.371`; it did not pass.
- Spectral `knn=10, k=3` produced sizes `17/24/18`, \(z_\rho=8.58\), and
  pairwise bootstrap ARI `0.659`; it passed.
- Spectral `knn=15, k=3` produced sizes `20/25/14`, \(z_\rho=7.49\), and
  pairwise bootstrap ARI `0.624`; it passed.
- The two passing \(k=3\) Spectral partitions had pairwise ARI `0.824`.
- Spectral `knn=15, k=2` also narrowly passed with pairwise bootstrap ARI
  `0.601`. Under the frozen smallest-\(k\) rule it must enter human review;
  reviewers may veto it if the two-cluster solution is substantively too
  coarse.
- Raw HDBSCAN assigned no papers to a cluster in the fixed negative-control
  run.

These results reject the provisional assumption that algorithm choice would
have little effect. Ward remains useful for visualizing a hierarchy, but it
cannot be frozen as the assignment method under the stated stability rule.
The eligible assignment candidates are therefore Spectral `knn=15, k=2`,
Spectral `knn=10, k=3`, and Spectral `knn=15, k=3`. Human review is used only
to veto an eligible candidate, not to rescue Ward, average-cosine HAC, or
K-means.

## Experimental scope

Two related experiments answer different questions:

1. **Design Theory group (59 papers):** Which subtopics emerge inside one
   predefined search-keyword group?
2. **Global corpus (284 papers):** What broader semantic structure emerges
   across the complete design-knowledge corpus?

Both experiments use BGE-M3 representations built from:

- one `Title [SEP] Abstract` chunk; and
- up to 12 `Title [SEP] keyword-conditioned PDF passage` chunks.

The L2-normalized chunk embeddings are mean-pooled and L2-normalized again.
The global corpus produced 3,612 chunks and a `284 × 1024` embedding matrix.
Only one over-context chunk required model-level truncation.

## Why multiple clustering families are required

The algorithms encode different assumptions:

- **K-means:** centroid-based, complete assignment, approximately compact
  clusters.
- **Agglomerative:** hierarchical merging under cosine distance.
- **DBSCAN / OPTICS / HDBSCAN:** density-based clusters with explicit noise.
- **UMAP–HDBSCAN:** density clustering after nonlinear manifold reduction,
  following the common BERTopic architecture.
- **Spectral:** communities in a nearest-neighbor similarity graph; does not
  require clusters to be well represented by Euclidean centroids.

No algorithm should be selected in advance. Configurations are compared using
cosine silhouette, random-seed agreement, 80% subsample bootstrap ARI, cluster
size, noise rate, and later human coherence and granularity.

## Prior research

- Grootendorst (2022), **BERTopic**, proposes transformer document embeddings
  followed by clustering and c-TF-IDF topic representation; the commonly used
  pipeline applies UMAP and HDBSCAN:
  <https://arxiv.org/abs/2203.05794>
- Eklund, Forsman, and Drewes (2023) empirically compare BERT/Doc2Vec,
  PCA/UMAP, and K-means/HDBSCAN. Their results emphasize evaluating the
  complete representation–reduction–clustering configuration; BERT + UMAP was
  strongest in their setting, rather than one clustering algorithm being
  universally best:
  <https://doi.org/10.3384/nejlt.2000-1533.2023.4396>
- Ogunleye et al. (2023) compare K-means, Spectral, Agglomerative, MeanShift,
  HDBSCAN, DBSCAN, BIRCH, and OPTICS inside embedding-based topic pipelines.
  Their best configuration was corpus-specific, and they explicitly argue that
  coherence alone must be complemented by topic quality and interpretability:
  <https://doi.org/10.3390/app13020797>
- Murugesan, Cho, and Tortora (2021) benchmark Spectral, DBSCAN, and K-means.
  They find no universally best method; DBSCAN is most appropriate for
  well-separated non-convex structure or substantial outliers, while Spectral
  performs strongly overall but may be less stable:
  <https://doi.org/10.1007/978-3-030-60104-1_20>
- von Luxburg (2007) explains why Spectral clustering can outperform
  centroid-based K-means when the relevant structure is expressed in a
  similarity graph:
  <https://doi.org/10.1007/s11222-007-9033-z>
- Ben-Hur, Elisseeff, and Guyon (2002) motivate selecting cluster structure by
  agreement across subsamples, which is the basis for the bootstrap-ARI test:
  <https://psb.stanford.edu/psb-online/proceedings/psb02/benhur.pdf>
- Zhang et al. (2022) show that direct clustering of high-quality contextual
  embeddings can produce coherent and diverse topics, supporting the
  embedding → clustering → topic-representation design:
  <https://aclanthology.org/2022.naacl-main.285/>

## Results: 59-paper Design Theory experiment

- Raw-space HDBSCAN rejected approximately 86% as noise.
- DBSCAN and OPTICS also failed the predefined cluster-size/noise criteria.
- Agglomerative clustering produced singleton-dominated partitions.
- Spectral clustering produced the strongest alternatives:
  - `knn=15, k=3`: cluster sizes `20/14/25`, bootstrap ARI `0.631`;
  - `knn=15, k=4`: cluster sizes `24/11/12/12`, bootstrap ARI `0.555`.

The 59-paper result therefore does not support freezing K-means `k=2` without
comparing the more interpretable Spectral `k=3` and `k=4` solutions.

## Results: global 284-paper experiment

### Raw embedding space

- K-means `k=2`: silhouette `0.0947`, bootstrap ARI `0.8316`.
- K-means `k=3`: silhouette `0.0933`, bootstrap ARI `0.7809`.
- Raw HDBSCAN produced high apparent silhouette only by rejecting roughly
  65–80% of papers as noise, so these configurations were not eligible.

### Alternative algorithms

- Spectral `knn=10, k=2`: bootstrap ARI `0.8798`, but one cluster contains
  233/284 papers and is too coarse for topic interpretation.
- UMAP–HDBSCAN's most stable configurations also collapse to two highly
  unequal macro-clusters.
- Spectral offers usable fine-grained candidates:
  - `knn=15, k=10`: sizes `9–66`, silhouette `0.0490`,
    bootstrap ARI `0.5428`;
  - `knn=15, k=12`: sizes `7–45`, silhouette `0.0644`,
    bootstrap ARI `0.5029`.

The `k=10` and `k=12` partitions have only modest agreement with the predefined
keyword groups (`NMI=0.350` and `0.367`). They partly recover recognizable
forms such as patterns, rationale, heuristics, and design theory, while also
forming cross-keyword clusters around AI, education, HCI, games, and
design-knowledge production. This indicates that the clusters are not merely
copies of the retrieval labels.

## Current recommendation

Do not freeze the global `k=2` solution merely because it maximizes stability.
It is useful as a macro-level diagnostic but too coarse for the research goal.

Carry these candidates into blinded human audit:

1. **Spectral `knn=15, k=10`** — more stable;
2. **Spectral `knn=15, k=12`** — finer and more balanced;
3. **UMAP–HDBSCAN `n_neighbors=5, min_cluster_size=15`** — density-based
   comparison, including explicit outliers;
4. **K-means `k=5` or `k=6`** — centroid baseline at comparable granularity.

Human comparison should rate paper membership coherence, coverage,
distinctiveness, appropriate granularity, and usefulness. Only after this audit
should the paper–cluster assignment be frozen.
