# Batch 3: Keyword-conditioned clustering

## Scope

Batch 3 assigns each of the 283 retained English-language publications to a
cluster **within its predefined search-keyword group**. It does not cluster the
entire corpus as one undifferentiated collection, and it does not use a
predefined category codebook.

The representation is fixed before clustering:

1. title;
2. abstract;
3. up to 12 keyword-conditioned full-text passages;
4. BGE-M3 embedding;
5. L2 normalization.

For each keyword group, the number of clusters is estimated from the K-means
inertia elbow using maximum distance from the endpoint chord. That estimated
`k` is passed to Spectral clustering on a cosine 10-nearest-neighbor graph.
K-means is therefore used to estimate `k`, not to produce the final paper
membership.

## Group-relative validation rule

The earlier Design Theory requirement of eight papers per cluster must not be
applied as an absolute cutoff to smaller keyword groups. The current minimum is:

```text
max(3, ceil(0.15 * keyword_group_size))
```

A candidate passes automatic validation only when:

- its smallest cluster meets the group-relative minimum;
- permutation-adjusted cohesion has `z_rho > 2`;
- mean 80%-subsample pairwise bootstrap ARI is at least `0.60`.

Passing these checks does not establish conceptual validity. It makes the
candidate eligible for a human granularity veto. A failed check does not prove
that the cluster structure is meaningless; it means the assignment must not be
described as automatically validated.

## Current 283-paper candidate

| Keyword | n | k | Cluster sizes | Status |
|---|---:|---:|---:|---|
| Design knowledge | 68 | 3 | 23/35/10 | Numeric review required |
| Design theory | 58 | 3 | 17/23/18 | Frozen from the completed pilot |
| Design Patterns | 51 | 2 | 35/16 | Numeric pass; human veto pending |
| Design methods | 27 | 3 | 14/9/4 | Numeric review required |
| Design Guidelines | 22 | 4 | 3/4/6/9 | Numeric review required |
| Design principles | 15 | 2 | 10/5 | Stability review required |
| Design rationale | 14 | 2 | 12/2 | Size and stability review required |
| Design Rules | 11 | 2 | 4/7 | Stability review required |
| Design Heuristics | 10 | 2 | 7/3 | Numeric pass; human veto pending |
| Design frameworks | 6 | 2 | 2/4 | Size and stability review required |
| Design Procedures | 1 | 1 | 1 | Cannot be subdivided |

The complete candidate is
`outputs/batch3/all_283_keyword_conditioned_20260728/all_283_keyword_conditioned_assignments.csv`.
It contains 283 rows, 283 unique paper IDs, and no missing cluster IDs.

## Human review

Two files make the remaining decision auditable:

- `keyword_cluster_profiles_for_review.csv` provides cluster sizes,
  representative titles, preliminary descriptive TF-IDF terms, centroid
  cohesion, and the smallest assignment margin. The terms are diagnostic and
  are **not** final topics.
- `human_review_shortlist.csv` contains 95 representative or boundary papers
  from all groups that still need a decision.

For each displayed paper, the reviewer records whether the membership is
acceptable and, if not, a suggested cluster. For each cluster, the reviewer
also records whether it is coherent and whether its granularity is too coarse,
acceptable, or too fine.

The review has veto/exception authority; it must not be used to invent a
post-hoc category codebook. An accepted candidate that failed the numeric rule
must be explicitly recorded as a human-accepted exception with the reason.

## Topic-modeling boundary

Topic modeling must not change paper membership. After Batch 3 is frozen,
statistical topic interpretation will treat each fixed cluster as a class and
use class-based TF-IDF to extract contrastive terms and phrases. This adapts the
representation stage of BERTopic without rerunning its UMAP-HDBSCAN clustering
stage.
