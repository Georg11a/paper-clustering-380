# Global Clustering Comparison — Executable Plan

Date: 2026-07-29

Corpus: 282 retained papers
Meeting request: run all papers together, add UMAP before clustering, compare
K-Means/DBSCAN/HDBSCAN, follow the UMAP + HDBSCAN pipeline shared by Zhicheng,
and compare traditional and LLM-based topic outputs.

## 1. Recommended change in one sentence

Do **not** replace the current keyword-conditioned pipeline yet. Add a separate,
reproducible global comparison experiment using the same frozen 282-paper
BGE-M3 embeddings:

```text
Frozen 282 papers + frozen 1024D BGE-M3 embeddings
                    |
          +---------+---------+
          |                   |
      Raw 1024D          UMAP 5D / 10D
          |                   |
    +-----+-----+       +-----+-----+
 K-Means DBSCAN HDBSCAN K-Means DBSCAN HDBSCAN
          |
 numeric comparison -> blinded human review -> final method selection
          |
 same shortlisted memberships -> c-TF-IDF vs LLM topic descriptions
```

This design separates two questions:

1. Does UMAP improve clusterability?
2. Given the same representation space, which clustering family is most useful?

It also prevents topic-label quality from being confused with paper-assignment
quality.

## 2. Current repository state

- [x] The active cleaned corpus contains 282 papers.
- [x] Frozen BGE-M3 embeddings already exist.
- [x] An earlier global comparison exists for 284 papers.
- [x] Raw DBSCAN and UMAP-HDBSCAN are already implemented in
  `scripts/run_batch3_alternative_algorithms.py`.
- [x] K-Means and raw HDBSCAN are already implemented in
  `scripts/run_batch3_clustering.py`.
- [ ] The global comparison must be rerun on the final 282-paper corpus.
- [ ] One runner must evaluate the complete raw/UMAP × algorithm matrix with
  identical metrics and resampling.
- [ ] UMAP-K-Means and UMAP-DBSCAN are not currently included in the same
  controlled comparison.

The old 284-paper results are useful as an audit trail but should not be
presented as the final meeting result.

## 3. Freeze the experimental inputs

Use these files and do not regenerate embeddings during algorithm comparison:

```text
Input:
outputs/batch2/global_282_bge_m3_contextual_final2/global_contextual_input_282.csv

Embeddings:
outputs/batch2/global_282_bge_m3_contextual_final2/embeddings_bge_m3_global_282_k12.npy

Embedding audit:
outputs/batch2/global_282_bge_m3_contextual_final2/refresh_metadata.json
```

### Execute

```bash
cd "/Users/baiyixin/Documents/Survey - design knowledge/paper-clustering-380"

.venv/bin/python - <<'PY'
import numpy as np
import pandas as pd

papers = pd.read_csv(
    "outputs/batch2/global_282_bge_m3_contextual_final2/"
    "global_contextual_input_282.csv"
)
vectors = np.load(
    "outputs/batch2/global_282_bge_m3_contextual_final2/"
    "embeddings_bge_m3_global_282_k12.npy"
)

assert len(papers) == 282
assert papers["paper_id"].nunique() == 282
assert vectors.shape == (282, 1024)
print("Frozen input validated:", len(papers), vectors.shape)
PY
```

### Completion criteria

- [ ] Exactly 282 rows and 282 unique `paper_id` values.
- [ ] Embedding shape is exactly `282 × 1024`.
- [ ] Input and embedding SHA-256 values are copied from
  `refresh_metadata.json` into the new run metadata.
- [ ] No paper is added, removed, translated, or deduplicated during this run.

### Prior research

- The comparison must hold representation constant because clustering outcomes
  depend strongly on the complete representation–reduction–algorithm
  configuration: [Eklund, Forsman, and Drewes, *An Empirical Configuration
  Study of a Common Document Clustering Pipeline*
  (2023)](https://nejlt.ep.liu.se/article/view/4396).
- Contextual-embedding clustering can be a direct and competitive route to
  topic discovery: [Zhang et al., *Is Neural Topic Modelling Better than
  Clustering?* (2022)](https://arxiv.org/abs/2204.09874).

## 4. Implement one controlled global comparison runner

Create:

```text
scripts/run_global_282_clustering_comparison.py
```

Prefer extracting and reusing metric helpers from the two existing Batch 3
scripts instead of copying their logic.

### Required CLI

```text
--input
--embeddings
--out
--seed
--umap-components
--umap-neighbors
--k-values
--bootstrap-repeats
--bootstrap-fraction
--max-noise-fraction
--min-eligible-cluster-size
```

### Required configurations

Run every algorithm in raw space and UMAP space:

| Space | K-Means | DBSCAN | HDBSCAN |
|---|---|---|---|
| Raw normalized 1024D | required baseline | required | required |
| UMAP 5D | required | required | required |
| UMAP 10D | required | required | required |

Do not use the 2D visualization coordinates for clustering. Generate separate
5D/10D UMAP representations for clustering and a separate 2D representation
only for visualization.

### Initial parameter grid

Use a bounded grid rather than one hand-picked configuration:

```yaml
umap:
  n_components: [5, 10]
  n_neighbors: [5, 10, 15, 30]
  min_dist: [0.0]
  metric: [cosine]
  seeds: [11, 23, 37, 53, 71]

kmeans:
  k: [2, 3, 5, 6, 8, 10, 12]
  n_init: [50]

dbscan:
  min_samples: [3, 5, 8, 10]
  eps: derive from each space's cosine/Euclidean k-distance distribution

hdbscan:
  min_cluster_size: [5, 8, 10, 12, 15, 20]
  min_samples: [1, 3, 5, 8, 10]
  cluster_selection_method: [eom]
```

`k=2` and `k=3` are diagnostic macro-structure baselines. Do not select them
only because they are numerically stable if they are too coarse for literature
synthesis.

For DBSCAN, export the sorted k-distance values used to choose `eps`; do not
silently auto-select one value.

### Required output files

```text
outputs/batch3/global_282_clustering_comparison_20260729/
├── configuration_metrics.csv
├── all_cluster_assignments.csv
├── eligible_configurations.csv
├── pairwise_ari_matrix.csv
├── dbscan_k_distance_curves.csv
├── umap_seed_stability.csv
├── cluster_profiles.csv
├── human_review_shortlist.csv
└── run_metadata.json
```

### Implementation checks

- [ ] All configurations consume the identical frozen embedding matrix.
- [ ] All stochastic configurations record the random seed.
- [ ] UMAP is refit inside each bootstrap subsample; otherwise stability is
  overstated.
- [ ] Noise remains label `-1`; it is never silently assigned to the nearest
  cluster for numeric evaluation.
- [ ] Each assignment row retains `paper_id`, title, original keyword, method,
  configuration, cluster, and `is_noise`.
- [ ] The full command and library versions are saved in `run_metadata.json`.

### Prior research

- Zhicheng's shared practical pipeline reduces high-dimensional text
  embeddings with UMAP and then applies HDBSCAN; it also explicitly recommends
  comparing HDBSCAN with K-Means and DBSCAN:
  [Vizuara, *From Text to Insights: Hands-on Text Clustering and Topic
  Modeling — Part 1*](https://vizuara.substack.com/p/from-text-to-insights-hands-on-text).
- UMAP's original formulation supports nonlinear dimensionality reduction and
  dimensions beyond visualization:
  [McInnes, Healy, and Melville, *UMAP*](https://arxiv.org/abs/1802.03426).
- DBSCAN introduces density-connected clusters and explicit noise without a
  predefined number of clusters:
  [Ester et al., *A Density-Based Algorithm for Discovering Clusters in Large
  Spatial Databases with Noise*
  (1996)](https://file.biolab.si/papers/1996-DBSCAN-KDD.pdf).
- HDBSCAN generalizes density-based clustering across a hierarchy of density
  levels:
  [Campello, Moulavi, and Sander, *Density-Based Clustering Based on
  Hierarchical Density Estimates*
  (2013)](https://doi.org/10.1007/978-3-642-37456-2_14).
- A direct empirical document-clustering comparison of BERT/Doc2Vec,
  PCA/UMAP, and K-Means/HDBSCAN supports evaluating the complete pipeline
  rather than declaring one universally best algorithm:
  [Eklund, Forsman, and Drewes
  (2023)](https://nejlt.ep.liu.se/article/view/4396).

## 5. Run the global 282-paper experiment

After implementing the runner:

```bash
.venv/bin/python -m py_compile \
  scripts/run_global_282_clustering_comparison.py

.venv/bin/python scripts/run_global_282_clustering_comparison.py \
  --input \
  outputs/batch2/global_282_bge_m3_contextual_final2/global_contextual_input_282.csv \
  --embeddings \
  outputs/batch2/global_282_bge_m3_contextual_final2/embeddings_bge_m3_global_282_k12.npy \
  --out \
  outputs/batch3/global_282_clustering_comparison_20260729 \
  --seed 20260729 \
  --umap-components 5,10 \
  --umap-neighbors 5,10,15,30 \
  --k-values 2,3,5,6,8,10,12 \
  --bootstrap-repeats 50 \
  --bootstrap-fraction 0.8 \
  --max-noise-fraction 0.35 \
  --min-eligible-cluster-size 5
```

Run the full 100-repeat confirmation only after the shortlist is frozen:

```bash
.venv/bin/python scripts/run_global_282_clustering_comparison.py \
  --input \
  outputs/batch2/global_282_bge_m3_contextual_final2/global_contextual_input_282.csv \
  --embeddings \
  outputs/batch2/global_282_bge_m3_contextual_final2/embeddings_bge_m3_global_282_k12.npy \
  --out \
  outputs/batch3/global_282_clustering_confirmation_20260729 \
  --seed 20260729 \
  --bootstrap-repeats 100 \
  --bootstrap-fraction 0.8 \
  --confirm-shortlist \
  outputs/batch3/global_282_clustering_comparison_20260729/human_review_shortlist.csv
```

### Completion criteria

- [ ] Every method has at least one raw-space result.
- [ ] Every method has results for both UMAP 5D and UMAP 10D.
- [ ] The run completes without regenerating BGE-M3 embeddings.
- [ ] Re-running with the same seed reproduces the exported assignments.
- [ ] A different UMAP seed does not radically change a shortlisted solution.

### Prior research

- The shared Vizuara article uses 10-dimensional UMAP with `min_dist=0`,
  cosine distance, followed by Euclidean HDBSCAN in reduced space. Its values
  are a starting point, not evidence that one setting is optimal for 282
  papers: [Vizuara practical
  pipeline](https://vizuara.substack.com/p/from-text-to-insights-hands-on-text).
- Empirical work comparing UMAP as clustering preprocessing reports that the
  effect depends on the downstream algorithm and dataset:
  [Allaoui et al., *Considerably Improving Clustering Algorithms Using UMAP
  Dimensionality Reduction Technique*
  (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7340901/).
- Text-embedding experiments find meaningful trade-offs: density-based methods
  can outperform K-Means while labeling more documents as outliers:
  [Thakur et al., *Influence of Various Text Embeddings on Clustering
  Performance in NLP* (2023)](https://arxiv.org/abs/2305.03144).

## 6. Compare configurations with the same evaluation protocol

Compute metrics in the **original normalized 1024D embedding space**, even for
UMAP-based assignments. Reduced-space metrics may reward distortions created
by UMAP.

### Required numeric metrics

| Metric | Purpose |
|---|---|
| Number of clusters | Detect overly coarse/fine solutions |
| Cluster sizes | Detect singleton/tiny/dominant clusters |
| Noise fraction | Treat unassigned papers as information and cost |
| Coverage (`1 - noise`) | Show how much of the corpus is usable downstream |
| Cosine silhouette in original space | Measure separation without scoring in the transformed space |
| Bootstrap pairwise ARI | Measure stability under 80% subsampling |
| Seed-to-seed ARI | Measure UMAP/K-Means stochastic sensitivity |
| NMI with original keyword | Diagnostic only; not a target to maximize |
| Representative-paper titles | Support human interpretation |

### Eligibility gate

A configuration enters human review only when it:

- produces at least 2 non-noise clusters;
- has smallest non-noise cluster of at least 5 papers;
- has no more than 35% noise;
- has valid original-space silhouette and bootstrap stability estimates; and
- does not place more than 70% of all papers into one cluster.

These are project decision rules, not universal thresholds. Freeze them in
`run_metadata.json` before inspecting the new 282-paper results.

Do not select the winner by a single combined score. Shortlist up to two
configurations per family and preserve at least:

1. one complete-assignment K-Means baseline;
2. one raw-space density result if eligible;
3. one UMAP-HDBSCAN result;
4. one UMAP-DBSCAN result if eligible.

### Prior research

- Clustering stability under subsampling provides evidence about whether
  structure persists after perturbing the dataset:
  [Ben-Hur, Elisseeff, and Guyon, *A Stability Based Method for Discovering
  Structure in Clustered Data*
  (2002)](https://psb.stanford.edu/psb-online/proceedings/psb02/benhur.pdf).
- HDBSCAN noise should be retained as an outlier result rather than treated
  automatically as algorithm failure:
  [Campello et al., *Hierarchical Density Estimates for Data Clustering,
  Visualization, and Outlier Detection*
  (2015)](https://doi.org/10.1145/2733381).
- BERTopic generalizability research documents the practical risk that
  HDBSCAN can exclude a large share of documents as outliers, motivating
  explicit coverage reporting and a K-Means comparison:
  [de Groot, Aliannejadi, and Haas
  (2022)](https://arxiv.org/abs/2212.08459).

## 7. Conduct blinded human comparison

Numeric metrics screen configurations; they do not establish that clusters are
useful for the systematic literature review.

### Build the review packet

For each shortlisted configuration, export:

- a blinded configuration code;
- cluster size and noise count;
- five representative papers nearest the original-space cluster centroid;
- five random papers;
- two boundary papers;
- one likely intruder from the nearest competing cluster;
- top c-TF-IDF terms;
- original keyword distribution;
- a separate private answer key.

Two reviewers independently rate each cluster from 1–5:

| Criterion | Question |
|---|---|
| Membership coherence | Do the papers belong together? |
| Distinctiveness | Is this cluster meaningfully different from others? |
| Granularity | Is it neither too broad nor too narrow? |
| Labelability | Can reviewers assign a concise, evidence-based topic? |
| Review usefulness | Would this cluster support reading and synthesis? |

Reviewers also mark:

- papers that do not belong;
- clusters that should merge;
- clusters that should split;
- whether each noise paper is meaningfully peripheral;
- preferred configuration without seeing the algorithm name.

### Completion criteria

- [ ] At least two reviewers complete the same blinded packet.
- [ ] Agreement and disagreements are recorded.
- [ ] Algorithm identities are revealed only after ratings are frozen.
- [ ] The chosen solution passes both numeric eligibility and human review.
- [ ] A decision record explains why rejected alternatives were not selected.

### Prior research

- Topic-model likelihood and other automated measures do not guarantee human
  interpretability; direct human tasks reveal qualities missed by model-only
  metrics: [Chang et al., *Reading Tea Leaves: How Humans Interpret Topic
  Models* (2009)](https://papers.neurips.cc/paper/2009/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html).
- The Vizuara pipeline also validates clusters through representative-document
  sampling and interactive inspection after UMAP-HDBSCAN:
  [Vizuara practical
  pipeline](https://vizuara.substack.com/p/from-text-to-insights-hands-on-text).
- Stability is useful for identifying reproducible structure, but it should be
  paired with substantive review:
  [Ben-Hur et al.
  (2002)](https://psb.stanford.edu/psb-online/proceedings/psb02/benhur.pdf).

## 8. Compare statistical and LLM topic descriptions separately

Do not let the LLM change cluster membership during this comparison. Apply both
topic-description methods to the same shortlisted/final assignments:

### Method A — statistical topic representation

- c-TF-IDF key phrases;
- representative titles/abstract passages;
- extractive source-evidence sentences;
- no generated claims unsupported by a paper passage.

### Method B — LLM topic representation

- same papers and same evidence packet as Method A;
- descriptive topic phrase;
- short prose summary;
- sentence-level citations back to source papers;
- two-human review with approve/revise/reject.

### Side-by-side evaluation

Reviewers compare:

- faithfulness to source papers;
- specificity;
- distinctiveness from other topics;
- coverage of cluster membership;
- usefulness for the literature-review outline;
- unsupported or invented claims.

This comparison answers whether LLMs improve **topic interpretation**, not
whether they produced a better clustering assignment.

### Prior research

- BERTopic explicitly separates document embeddings, clustering, and
  class-based TF-IDF topic representation:
  [Grootendorst, *BERTopic: Neural Topic Modeling with a Class-Based TF-IDF
  Procedure* (2022)](https://arxiv.org/abs/2203.05794).
- Direct clustering of contextual embeddings with an appropriate term
  selection method can produce coherent and diverse topics, making it a
  meaningful statistical baseline:
  [Zhang et al.
  (2022)](https://arxiv.org/abs/2204.09874).
- Human interpretability must be evaluated directly:
  [Chang et al.
  (2009)](https://papers.neurips.cc/paper/2009/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html).

## 9. Update the explorer for comparison

Add a temporary comparison view; do not replace the current frozen
keyword-conditioned explorer.

Required controls:

- configuration selector;
- algorithm/space label shown only after blinded review is complete;
- one-cluster-at-a-time highlighting;
- explicit noise toggle;
- cluster sizes and coverage;
- representative/boundary paper markers;
- side-by-side statistical and LLM topic descriptions;
- download links for assignments and metrics.

Use distinct colors for all visible clusters and a neutral gray for noise.
Never reuse the same color for two clusters in the same view.

### Prior research

- The shared Vizuara article combines manual document inspection with
  interactive UMAP visualization:
  [Vizuara practical
  pipeline](https://vizuara.substack.com/p/from-text-to-insights-hands-on-text).
- UMAP was designed as a general-purpose nonlinear reduction technique and is
  widely used for visualization, but the 2D plot should not be interpreted as
  a literal measurement space:
  [McInnes, Healy, and Melville
  (2018)](https://arxiv.org/abs/1802.03426).

## 10. Meeting-ready deliverables

Prepare exactly these items:

- [ ] One slide: experimental design (`raw/UMAP × three algorithms`).
- [ ] One table: cluster count, sizes, noise, coverage, silhouette, bootstrap
  ARI, seed ARI.
- [ ] One slide: best K-Means, DBSCAN, and HDBSCAN visualizations using a
  shared legend policy.
- [ ] One slide: examples of informative HDBSCAN/DBSCAN noise papers.
- [ ] One slide: blinded human-review result.
- [ ] One slide: c-TF-IDF vs LLM topic-description comparison.
- [ ] One recommendation with one backup method.
- [ ] GitHub link, exact run command, environment, seeds, and output paths.

### Suggested conclusion structure

```text
We held the 282-paper corpus and BGE-M3 representation constant.
We compared K-Means, DBSCAN, and HDBSCAN in raw and UMAP-reduced spaces.
We evaluated coverage, original-space separation, resampling stability,
and blinded human usefulness.
We selected [configuration] because [...], while retaining [configuration]
as a sensitivity analysis.
Topic descriptions were compared separately using identical memberships.
```

### Prior research

- No clustering family is universally best; empirical selection must reflect
  the corpus and research goal:
  [Murugesan, Cho, and Tortora, *Benchmarking Clustering Algorithms*
  (2021)](https://doi.org/10.1007/978-3-030-60104-1_20).
- The complete text-clustering pipeline should be reported, including
  vectorization, dimensionality reduction, clustering, and evaluation:
  [Eklund, Forsman, and Drewes
  (2023)](https://nejlt.ep.liu.se/article/view/4396).

## 11. Recommended execution order

### Day 1 — controlled numeric comparison

- [ ] Validate the final 282-paper input and embeddings.
- [ ] Implement the unified comparison runner.
- [ ] Run a smoke test with five bootstrap repeats.
- [ ] Run the full exploratory grid with 50 repeats.
- [ ] Freeze the human-review shortlist.

### Day 2 — human and topic review

- [ ] Generate blinded review packets.
- [ ] Complete two-reviewer assessment.
- [ ] Reveal method identities and record the decision.
- [ ] Compare c-TF-IDF and LLM topic descriptions on identical memberships.

### Day 3 — reporting

- [ ] Run 100-repeat confirmation on shortlisted configurations.
- [ ] Build the temporary comparison explorer.
- [ ] Prepare the meeting table and slides.
- [ ] Push code, metadata, compact CSV outputs, and documentation to GitHub.

## 12. Definition of done

This task is complete only when:

- [ ] The final 282-paper corpus—not the old 284-paper corpus—has been used.
- [ ] K-Means, DBSCAN, and HDBSCAN have all been run on all papers combined.
- [ ] Each family has a raw-space baseline and a UMAP-space result.
- [ ] Noise is reported and inspected, not treated automatically as failure.
- [ ] Numeric evaluation is reproducible and performed in original space.
- [ ] At least two humans conduct a blinded comparison.
- [ ] Statistical and LLM topic descriptions are compared on identical
  memberships.
- [ ] A written method-selection decision and backup configuration exist.
- [ ] Exact commands, seeds, inputs, outputs, and prior research are recorded.
