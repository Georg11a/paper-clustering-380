# Global Clustering Comparison — Revised Executable Plan

Date: 2026-07-29
Revised: 2026-07-31
Corpus: 282 retained papers

Meeting request: run all papers together, compare K-Means, DBSCAN, and HDBSCAN,
add UMAP before clustering, and test the UMAP + HDBSCAN workflow shared by
Zhicheng.

This revision adds two controls that are necessary for a valid decision:

1. the incumbent Spectral method must be included; and
2. the experiment must include a keyword-neutral representation.

## 1. Meeting-critical statement

Do **not** copy the Vizuara article's HDBSCAN parameters.

The article clusters 44,949 papers. This project clusters 282 papers, roughly
160 times fewer. Its `min_cluster_size=50` would allow at most five clusters in
this corpus and would probably produce one or two large clusters plus extensive
noise.

The earlier observation that “HDBSCAN creates too much noise” may therefore be
a parameter artifact rather than a property of HDBSCAN.

The meeting-safe interpretation is:

> Noise can be useful weak-affinity information, as Leilani suggested, but only
> after showing that the noise is not mainly caused by an inappropriate
> `min_cluster_size`, `min_samples`, UMAP neighborhood, or representation.

## 2. Revised experimental question

Do not replace the current keyword-conditioned Spectral assignments yet.
Treat the global experiment as a parallel **method-selection branch**.

```text
Frozen 282-paper corpus
        |
        +-------------------------------+
        |                               |
R_kw: keyword-conditioned         R_neutral: keyword-neutral
title + abstract + passages       title + abstract only
        |                               |
        +---------------+---------------+
                        |
              L2-normalized BGE-M3
                        |
        +---------------+---------------+
        |                               |
     Raw 1024D               Shared frozen UMAP 5D / 10D
        |                               |
 Spectral / K-Means / DBSCAN / HDBSCAN
                        |
 original-space metrics + stability + sanity checks
                        |
                 blinded human review
                        |
                 final method decision
```

This design separates four questions:

1. Is the current Spectral method better or worse than the three challengers?
2. Does UMAP improve a method, or only change the geometry?
3. Does keyword-conditioned passage selection leak the original keyword groups
   into the “naturally emerging” clusters?
4. Are the resulting clusters stable and substantively useful?

## 3. Structural issues that must be resolved first

### 3.1 Include Spectral as the incumbent baseline

The current Batch 3 method is not K-Means, DBSCAN, or HDBSCAN.

For each keyword group, K-Means inertia estimates `k`; the final memberships
come from Spectral clustering on a cosine 10-nearest-neighbor graph. The
existing 282-paper assignments, Path 1 coding, and Path 2 frozen inputs are
therefore downstream of Spectral.

The comparison must contain:

| Role | Method |
|---|---|
| Incumbent | Spectral on a cosine kNN graph |
| Challenger | K-Means |
| Challenger | DBSCAN |
| Challenger | HDBSCAN |

Without Spectral, the meeting could select a challenger without any controlled
evidence about whether it improves on the method already used downstream.

The group-level K-Means elbow rule must not be reused for the global Spectral
run. Global `k` is selected by stability over a separately defined range.

### 3.2 Add a keyword-neutral representation

The current representation is:

```text
title + abstract + up to 12 passages selected using the paper's keyword group
→ BGE-M3 → L2 normalization
```

This is appropriate for comparison with the frozen Batch 3 work, but not
sufficient for the meeting's request that clusters “emerge naturally.”
The passage-selection stage already contains keyword information. Agreement
between global clusters and keyword groups can therefore reflect representation
leakage rather than discovered structure.

Treat representation as an experimental factor:

| ID | Construction | Purpose |
|---|---|---|
| `R_kw` | Current title + abstract + up to 12 keyword-conditioned passages | Continuity with the frozen pipeline |
| `R_neutral` | Title + abstract only, embedded with the same BGE-M3 model | Valid test of naturally emerging global structure |

If a future neutral passage selector is added, it must use one global rule that
does not receive the original keyword label.

For every configuration, report ARI with the original keyword groups under both
representations. The difference
`ARI(R_kw, keyword) - ARI(R_neutral, keyword)` is a leakage diagnostic, not a
formal causal estimate.

The runner must assert that both representations are BGE-M3, 1024-dimensional,
row-aligned, and L2-normalized before raw-space clustering.

### 3.3 Keep the global experiment separate from the frozen branch

The Batch 3 topic-modeling boundary remains valid for the frozen branch:
class-based TF-IDF may interpret the fixed memberships but must not recluster
them.

The global comparison is a separate method-selection experiment. It may run
UMAP-HDBSCAN and other clustering methods, but its memberships do not overwrite
Batch 3 or invalidate downstream coding until a documented final decision is
made.

If a challenger is adopted later, the decision record must list which Path 1
and Path 2 artifacts require regeneration.

### 3.4 Define document independence consistently

The 283-to-282 deduplication established that a superseded conference version
is not an independent analytic record. Similar cases remain:

| Paper ID(s) | Relationship | Current handling |
|---|---|---|
| `cf2ee8097a06`, `f941f0424e1a` | Chapters 4 and 6 of Baldwin & Clark, *Design Rules Vol. 2* | Two records |
| `2a22a314dcfb` | Lane report described as a dissertation summary; companion report exists | One observed record |

Before confirmation, freeze a written independence rule for conference/journal
versions, book chapters, reports, and dissertation derivatives.

Do not silently remove records during the exploratory run. Instead:

- flag related records in metadata;
- report results with all 282 records;
- run a sensitivity analysis with non-independent pairs grouped or excluded;
- prevent a persistent two-document pair from being counted as standalone
  evidence of cluster stability.

This matters because near-duplicate or same-source documents can create
artificial local density and inflate bootstrap stability for methods that
produce small clusters.

## 4. Freeze and audit the experimental inputs

### 4.1 Current keyword-conditioned input

```text
Input:
outputs/batch2/global_282_bge_m3_contextual_final2/global_contextual_input_282.csv

R_kw embeddings:
outputs/batch2/global_282_bge_m3_contextual_final2/embeddings_bge_m3_global_282_k12.npy

Audit metadata:
outputs/batch2/global_282_bge_m3_contextual_final2/refresh_metadata.json
```

### 4.2 Required neutral input

Create and freeze:

```text
outputs/batch2/global_282_bge_m3_neutral_title_abstract_20260731/
├── global_neutral_input_282.csv
├── embeddings_bge_m3_global_282_title_abstract.npy
├── embedding_metadata.json
└── SHA256SUMS.txt
```

The neutral text must contain title and abstract only. It must not contain the
original keyword, keyword-conditioned passages, cluster labels, codebook terms,
or generated summaries.

### 4.3 Input assertions

```python
assert len(papers) == 282
assert papers["paper_id"].nunique() == 282
assert kw_vectors.shape == (282, 1024)
assert neutral_vectors.shape == (282, 1024)
assert papers["paper_id"].tolist() == neutral_papers["paper_id"].tolist()
assert np.allclose(np.linalg.norm(kw_vectors, axis=1), 1.0, atol=1e-5)
assert np.allclose(np.linalg.norm(neutral_vectors, axis=1), 1.0, atol=1e-5)
```

### Completion criteria

- [ ] Exactly 282 rows and 282 unique paper IDs.
- [ ] `R_kw` and `R_neutral` use identical row order.
- [ ] Both embedding matrices are `282 × 1024` BGE-M3 vectors.
- [ ] SHA-256 values are saved before clustering.
- [ ] The exact neutral text construction is recorded.
- [ ] No record is added, removed, or merged during a run.

### Prior research

- BGE-M3 supports multilingual and multi-granularity text representation, but
  its retrieval performance does not itself validate cluster quality:
  [Chen et al., *BGE M3-Embedding*
  (2024)](https://arxiv.org/abs/2402.03216).
- Document-clustering outcomes depend on the complete representation,
  reduction, and algorithm configuration:
  [Eklund, Forsman, and Drewes, *An Empirical Configuration Study of a Common
  Document Clustering Pipeline*
  (2023)](https://nejlt.ep.liu.se/article/view/4396).

## 5. Build one controlled comparison runner

Create:

```text
scripts/run_global_282_clustering_comparison.py
```

Prefer extracting and reusing tested metric helpers from the existing Batch 3
scripts.

### Required CLI

```text
--input
--representation name=embedding_path
--out
--seeds
--umap-components
--umap-neighbors
--k-values
--bootstrap-repeats
--bootstrap-fraction
--min-eligible-cluster-size
--sanity-check-spec
```

### Required matrix

| Space | Spectral | K-Means | DBSCAN | HDBSCAN |
|---|---:|---:|---:|---:|
| Raw normalized 1024D | required | required | required diagnostic | required |
| Shared UMAP 5D | required | required | required | required |
| Shared UMAP 10D | optional comparison | optional comparison | optional comparison | required Zhicheng comparison |

Run the matrix for both `R_kw` and `R_neutral`.

Raw 1024D DBSCAN is expected to perform poorly because of high-dimensional
density concentration. Retain the result as evidence about why reduction may
be needed; do not hide it as a failed run.

## 6. UMAP control: fit once, save, hash, and share

For a fixed representation, dimension, neighborhood, and seed, fit UMAP once.
All four methods must read the same saved matrix.

```python
reducer = UMAP(
    n_components=5,
    n_neighbors=15,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)
X_umap = reducer.fit_transform(emb_bge_m3_l2)
np.save("data/umap/R_neutral_umap_5d_nn15_seed42.npy", X_umap)
```

Save each matrix's SHA-256 in `data/umap/SHA256SUMS.txt` and record it in run
metadata. The comparison runner must load, not independently regenerate, the
matrix for each algorithm.

The phrase “fit once” applies within each controlled comparison cell:

```text
representation × n_components × n_neighbors × seed
```

During bootstrap, UMAP must be refit on each subsample to measure end-to-end
pipeline stability. It is still fit only once per bootstrap cell and then
shared by all four algorithms on that subsample. Reusing the full-data UMAP in
every bootstrap would overstate stability.

Visualization UMAP is separate:

- clustering uses saved 5D or 10D coordinates;
- visualization uses a separate 2D or 3D UMAP;
- visualization coordinates never become clustering input;
- UMAP axes are not interpreted as variables.

### UMAP grid

```yaml
n_components: [5, 10]
n_neighbors: [5, 15, 30]
min_dist: [0.0]
metric: [cosine]
exploratory_seeds: [11, 23, 37, 53, 71]
meeting_reference_seed: [42]
```

`n_neighbors=15` is 5.3% of this 282-paper corpus, compared with approximately
0.03% of the Vizuara corpus. It must therefore be tested, not treated as a
scale-free default. `n_neighbors=30` is intentionally a more global
sensitivity condition.

### Prior research

- UMAP is stochastic and supports non-visualization output dimensions:
  [McInnes, Healy, and Melville, *UMAP*
  (2018)](https://arxiv.org/abs/1802.03426).
- Standard UMAP does not preserve original local density, which is important
  when a density-based method follows it:
  [Narayan, Berger, and Cho, *Assessing Single-Cell Transcriptomic Variability
  through Density-Preserving Data Visualization*
  (2021)](https://doi.org/10.1038/s41587-020-00801-7).

## 7. Correct parameter ranges

### 7.1 Spectral — incumbent

```yaml
affinity: nearest_neighbors
n_neighbors: [10]
k: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
random_state: [42]
```

- Keep cosine kNN graph construction consistent with Batch 3.
- Do not use the keyword-group elbow rule globally.
- Select `k` primarily by bootstrap stability, then inspect granularity.

### 7.2 K-Means — complete-assignment baseline

```yaml
k: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
n_init: [50]
random_state: [42]
```

- Raw-space K-Means requires L2-normalized embeddings.
- K-Means has no noise concept and forces every paper into a cluster.
- Report the full cluster-size distribution because the method favors compact,
  approximately spherical clusters of comparable scale.
- Select `k` by stability over 50 80%-subsample runs, not by maximum
  silhouette.

### 7.3 DBSCAN — global-density diagnostic

```yaml
min_samples: [3, 5, 10]
eps: k-distance knee plus a documented sweep around the knee
```

- Derive the initial `eps` from the sorted k-distance curve using a documented
  knee detector.
- Export the complete `eps × cluster_count × noise_fraction` curve.
- Do not tune `eps` until a visually preferred result appears.
- A single global `eps` assumes comparable density across clusters. If DBSCAN
  is dominated by HDBSCAN on this heterogeneous corpus, report that as a
  methodological finding rather than a software failure.
- `min_samples` constrains the smallest detectable dense theme. Report that
  consequence explicitly.

### 7.4 HDBSCAN — variable-density challenger

```yaml
min_cluster_size: [2, 3, 5, 8, 12, 15]
min_samples: [1, 3, 5, 10]
cluster_selection_method: [eom, leaf]
prediction_data: [true]
```

- `eom` tends toward fewer, larger clusters; `leaf` may expose finer topics.
- Export `noise_fraction`, `cluster_count`, and cluster sizes across
  `min_cluster_size`. This plot is itself a meeting deliverable.
- If microclusters are excessive, test a documented
  `cluster_selection_epsilon` merge as a secondary sensitivity analysis. Do
  not immediately raise `min_cluster_size` and erase all small themes.
- Use all-points membership vectors to report, for every noise paper, its
  nearest cluster and membership strength. Keep the hard label as `-1`.

#### Implementation dependency

The repository currently uses `sklearn.cluster.HDBSCAN`. In the current
environment, that implementation does not expose `prediction_data` or
`all_points_membership_vectors()`, and the separate
`scikit-learn-contrib/hdbscan` package is not installed.

The new runner must therefore do one of the following and record the choice:

1. add and pin the `hdbscan` package, import `hdbscan.HDBSCAN`, set
   `prediction_data=True`, and use
   `hdbscan.all_points_membership_vectors(model)`; or
2. omit the full per-cluster soft-membership claim and report only the hard
   noise label plus the capabilities actually exposed by the chosen library.

Option 1 is required for the proposed “nearest cluster + membership strength”
deliverable. Do not silently mix scikit-learn HDBSCAN assignments with soft
memberships produced by a different fitted implementation.

### HDBSCAN noise policy

Noise is neither automatically a defect nor automatically meaningful.

For every configuration:

- preserve hard noise label `-1`;
- never reassign noise for primary metrics;
- report coverage;
- report the strongest soft membership and its strength;
- inspect whether noise is caused by weak affinity, missing text, language,
  representation leakage, duplicate handling, or parameter scale.

### Prior research

- DBSCAN defines density-connected clusters and explicit noise with a single
  neighborhood scale:
  [Ester et al. (1996)](https://file.biolab.si/papers/1996-DBSCAN-KDD.pdf).
- HDBSCAN extracts clusters from a hierarchy of density levels and supports
  outlier interpretation:
  [Campello, Moulavi, and Sander
  (2013)](https://doi.org/10.1007/978-3-642-37456-2_14);
  [Campello et al.
  (2015)](https://doi.org/10.1145/2733381).
- Zhicheng's shared article is a practical implementation guide, not evidence
  that parameters from 44,949 papers transfer to 282:
  [Vizuara, *From Text to Insights*
  (2024)](https://vizuara.substack.com/p/from-text-to-insights-hands-on-text).

## 8. Evaluation protocol

### 8.1 Use one metric reference space

Compute all geometry-based internal metrics in the original normalized 1024D
BGE-M3 space using cosine distance, regardless of where labels were produced.

Do not compare a raw-space silhouette with a reduced-space silhouette. UMAP
actively changes separation and density.

Silhouette remains descriptive. It is not the final selection criterion because
it changes with cluster count and excludes noise when calculated only on
assigned points.

### 8.2 Inherit the stronger Batch 3 validation

| Rule | Global adaptation |
|---|---|
| Group minimum `max(3, ceil(0.15 × group_size))` | Replace with absolute exploratory floor `max(3, ceil(0.01 × 282)) = 3`; always report actual sizes |
| Permutation-adjusted cohesion | Retain `z_rho > 2` |
| 80% subsample pairwise bootstrap ARI | Retain mean ARI `≥ 0.60` as the primary automatic stability threshold |

For K-Means and Spectral, select `k` using stability and human granularity, not
the highest silhouette.

Do not use a fixed maximum-noise gate to declare density methods invalid.
Coverage is a separate usability dimension. A low-coverage configuration may
be a useful map of high-confidence cores but cannot serve as a complete
downstream partition.

### 8.3 Required numeric outputs

| Metric | Interpretation |
|---|---|
| Number of non-noise clusters | Coarse/fine structure |
| Full cluster-size distribution | Tiny or dominant clusters |
| Noise fraction and coverage | Assigned vs weak-affinity evidence |
| Original-space cosine silhouette | Descriptive separation |
| Permutation-adjusted cohesion `z_rho` | Separation beyond a null assignment |
| Mean 80%-subsample pairwise ARI | Primary stability evidence |
| Seed-to-seed ARI | UMAP/K-Means stochastic sensitivity |
| ARI with original keyword | Leakage/discovery diagnostic, not ground truth |
| Sanity-check results | Known-paper behavior |
| Representative and boundary papers | Human interpretation |

### 8.4 Compare methods fairly when noise differs

K-Means and Spectral assign all papers. DBSCAN and HDBSCAN may retain noise.
A single full-corpus ARI between them mixes membership disagreement with
coverage policy.

Report both:

1. pairwise ARI on the intersection of papers assigned by both methods; and
2. full-corpus coverage/noise separately.

For a four-method table, also report ARI on the subset assigned by all four
methods, but include the subset size so a very small intersection cannot look
misleadingly authoritative.

### 8.5 Interpret keyword agreement in the correct direction

Original keyword labels are not ground truth. The meeting asks whether themes
emerge beyond those groups.

- high keyword ARI under `R_kw` may indicate representation leakage;
- low keyword ARI with low stability indicates an unreliable partition;
- low keyword ARI with high stability and human coherence is evidence of a
  reproducible structure that differs from the search taxonomy.

Always interpret keyword ARI together with the `R_kw` versus `R_neutral`
comparison.

### Prior research

- Silhouette measures cohesion and separation but is not a complete model
  selection criterion:
  [Rousseeuw (1987)](https://doi.org/10.1016/0377-0427(87)90125-7).
- Clustering stability under subsampling tests whether structure persists
  after perturbation:
  [Ben-Hur, Elisseeff, and Guyon
  (2002)](https://psb.stanford.edu/psb-online/proceedings/psb02/benhur.pdf).

## 9. Automated sanity checks

Every configuration must write one row per probe to:

```text
outputs/batch3/global_282_clustering_comparison_20260731/sanity_checks.csv
```

| Probe | Paper ID(s) | Known property | Expected diagnostic behavior |
|---|---|---|---|
| P1 same source | `cf2ee8097a06`, `f941f0424e1a` | Chapters 4 and 6 of the same Baldwin & Clark book | Very small original-space distance; normally same cluster |
| P2 keyword homonym | `1d033f71eb02` | Tissue-interfaced bioelectronics; “design rules” may be material-science usage | Density method should leave it weak/noise or isolate it; forced inclusion signals greediness |
| P3 cross-language | `5c096c480387` | Slovenian full text, English abstract; publication year 2008 | Compare `R_kw` and `R_neutral` position and audit actual embedded text |
| P4 derivative | `2a22a314dcfb` | Dissertation-summary report | Observation item; no hard pass/fail |

These are diagnostic probes, not labels for training or post-hoc tuning.

### Required columns

```text
representation, space, umap_components, umap_neighbors, seed,
method, configuration, probe, paper_id, partner_paper_id,
raw_cosine_distance, cluster, partner_cluster, is_noise,
strongest_membership_cluster, strongest_membership_strength,
expected_behavior, observed_behavior, pass_warning
```

P3 must also export the exact neutral and keyword-conditioned text lengths,
languages if detected, selected passage IDs, and whether non-English full-text
content entered `R_kw`. This answers an input question; it must not be inferred
from the plot.

P1 is also a duplicate-dependence warning. Passing it does not prove a method
is good, and the pair must not be allowed to create a misleading stability
advantage.

## 10. Required output structure

```text
outputs/batch3/global_282_clustering_comparison_20260731/
├── representation_manifest.csv
├── configuration_metrics.csv
├── all_cluster_assignments.csv
├── eligible_configurations.csv
├── pairwise_ari_intersection.csv
├── keyword_ari_by_representation.csv
├── dbscan_k_distance_curves.csv
├── dbscan_eps_sensitivity.csv
├── hdbscan_parameter_sensitivity.csv
├── hdbscan_membership_vectors.csv
├── bootstrap_stability.csv
├── umap_seed_stability.csv
├── sanity_checks.csv
├── cluster_profiles.csv
├── human_review_shortlist.csv
├── umap/
│   ├── *.npy
│   └── SHA256SUMS.txt
└── run_metadata.json
```

`run_metadata.json` must include:

- exact CLI command and git commit;
- Python and package versions;
- paper and embedding hashes;
- UMAP matrix hashes;
- all seeds and parameter grids;
- distance metrics for every space;
- noise policy;
- bootstrap procedure;
- eligibility rules frozen before result inspection.

## 11. Blinded human comparison

Numeric validation screens configurations; it does not establish usefulness for
the systematic literature review.

For each shortlisted configuration, export:

- a blinded configuration code;
- cluster size and noise count;
- five representative papers nearest the original-space centroid;
- five random papers;
- two boundary papers;
- one likely intruder from the nearest competing cluster;
- top class-based TF-IDF terms;
- original keyword distribution;
- a separate private answer key.

Two reviewers independently rate:

| Criterion | Question |
|---|---|
| Membership coherence | Do the papers belong together? |
| Distinctiveness | Is the cluster meaningfully different from others? |
| Granularity | Is it neither too broad nor too narrow? |
| Labelability | Can it receive a concise evidence-based description? |
| Review usefulness | Does it support reading and synthesis? |

Reviewers also inspect:

- representative noise papers and their soft membership;
- clusters that should merge or split;
- P1–P4 sanity probes;
- whether a solution mainly reconstructs keyword groups;
- preferred configuration without seeing the algorithm name.

Algorithm identities are revealed only after ratings are frozen.

### Prior research

- Automated topic-model metrics can disagree with human interpretability:
  [Chang et al., *Reading Tea Leaves*
  (2009)](https://papers.neurips.cc/paper/2009/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html).
- The Vizuara workflow also calls for representative-document sampling after
  UMAP-HDBSCAN; this is a useful practice but not a substitute for blinded
  review:
  [Vizuara practical pipeline](https://vizuara.substack.com/p/from-text-to-insights-hands-on-text).

## 12. Topic descriptions remain a separate experiment

Do not allow either class-based TF-IDF or an LLM to change cluster membership
during method selection.

Apply both description methods to identical shortlisted assignments and compare:

- faithfulness to source papers;
- specificity;
- distinctiveness;
- coverage of cluster membership;
- usefulness for the review outline;
- unsupported claims.

This evaluates topic interpretation, not clustering.

BERTopic provides a direct precedent for separating document embeddings,
clustering, and class-based TF-IDF representation:
[Grootendorst (2022)](https://arxiv.org/abs/2203.05794).

## 13. Meeting-ready deliverables

- [ ] One slide: why Vizuara parameters cannot be copied from 44,949 to 282.
- [ ] One slide: revised matrix — two representations × raw/shared UMAP × four methods.
- [ ] One slide: incumbent Spectral versus three challengers.
- [ ] One slide: HDBSCAN `min_cluster_size` sensitivity showing cluster count,
  noise percentage, and cluster sizes; show both `eom` and `leaf`.
- [ ] One table: stability, `z_rho`, original-space silhouette, sizes, coverage,
  and sanity checks.
- [ ] One slide: keyword ARI under `R_kw` versus `R_neutral`.
- [ ] One slide: P1–P4 sanity-check results.
- [ ] One slide: examples of informative noise with soft membership strength.
- [ ] One slide: blinded human-review result.
- [ ] One recommendation, one backup method, and explicit downstream migration
  cost if Spectral is replaced.
- [ ] GitHub link, exact commands, hashes, environment, seeds, and output paths.

### Suggested meeting wording

```text
The previous high-noise HDBSCAN result may have been a parameter-scale artifact,
not a property of HDBSCAN. We therefore rescaled min_cluster_size for 282 papers,
tested eom and leaf extraction, and retained noise as soft-membership evidence.

We also added the incumbent Spectral method and a keyword-neutral BGE-M3
representation. This separates real emergent structure from keyword-conditioned
representation leakage.

All algorithms shared the same saved UMAP matrix within each condition. We
evaluated labels in the original 1024D cosine space and selected configurations
primarily by 80%-subsample stability, then blinded human usefulness—not by
silhouette alone.
```

## 14. Recommended execution order

### Minimum meeting-safe deliverable

1. [ ] Add Spectral to the comparison matrix.
2. [ ] Freeze shared 5D UMAP matrices for `n_neighbors ∈ {5,15,30}`.
3. [ ] Correct HDBSCAN ranges and run `eom` plus `leaf`.
4. [ ] Apply the inherited `z_rho > 2` and bootstrap ARI `≥ 0.60` rules.
5. [ ] Run the P3 cross-language input audit and all four sanity probes.
6. [ ] Generate at least one `R_neutral` title+abstract embedding matrix.

### Full controlled comparison

1. [ ] Audit the final 282-paper corpus and related-document cases.
2. [ ] Freeze `R_kw` and `R_neutral` with SHA-256.
3. [ ] Generate each UMAP condition once and share it across methods.
4. [ ] Run the exploratory grid with 50 80%-subsample repeats.
5. [ ] Freeze a human-review shortlist without a single combined score.
6. [ ] Complete two-reviewer blinded assessment.
7. [ ] Run 100-repeat confirmation on the frozen shortlist.
8. [ ] Build the comparison explorer and meeting slides.

## 15. Definition of done

- [ ] The final 282-paper corpus—not the old 284-paper corpus—is used.
- [ ] Spectral, K-Means, DBSCAN, and HDBSCAN are all compared.
- [ ] Both `R_kw` and `R_neutral` are tested.
- [ ] Raw and UMAP-space results use L2-normalized BGE-M3 inputs.
- [ ] Each controlled UMAP matrix is fit once, saved, hashed, and shared.
- [ ] UMAP is refit once per bootstrap cell for end-to-end stability.
- [ ] Metrics are computed in the original normalized 1024D cosine space.
- [ ] Bootstrap ARI `≥ 0.60` and `z_rho > 2` are the primary numeric checks.
- [ ] Silhouette is descriptive, not the winner-selection score.
- [ ] Noise remains `-1`, with coverage and soft membership reported.
- [ ] Keyword ARI is interpreted as a leakage/discovery diagnostic.
- [ ] P1–P4 sanity checks are exported.
- [ ] Related-document sensitivity is reported.
- [ ] At least two humans complete a blinded comparison.
- [ ] The frozen Batch 3 branch is not overwritten without a decision record.
- [ ] Exact commands, hashes, versions, inputs, outputs, and prior research are
  recorded.
