# All-283 keyword-conditioned clustering

Date: 2026-07-28

Topic modeling has **not** started. This run first assigns every retained paper
within its predefined keyword group.

Representation:

- BGE-M3 contextual embedding;
- Title + Abstract + up to 12 PDF passages;
- the K12 passages for each paper were selected using that paper's predefined
  keyword;
- clustering was performed within keyword groups, never across all 283 papers.

Rules:

- every keyword group with enough papers to form two groups is tested rather
  than being excluded by the Design Theory-specific absolute cutoff;
- the minimum acceptable cluster size is group-relative:
  `max(3, ceil(0.15 * group_n))`;
- eligible groups estimate k computationally from the K-means inertia elbow;
- the estimated k is then passed to Spectral clustering on a cosine 10-NN
  graph;
- cohesion, cluster size, and 80%-subsample stability are validation
  diagnostics;
- Design Theory reuses its previously frozen 58-paper assignment.

Current status:

| Keyword | n | k | Sizes | Status |
|---|---:|---:|---:|---|
| Design knowledge | 68 | 3 | 23/35/10 | Numeric stability review required |
| Design theory | 58 | 3 | 17/23/18 | Frozen |
| Design Patterns | 51 | 2 | 35/16 | Numerically passed; human granularity veto pending |
| Design methods | 27 | 3 | 14/9/4 | Numeric review required |
| Design Guidelines | 22 | 4 | 3/4/6/9 | Numeric review required |
| Design principles | 15 | 2 | 10/5 | Stability review required |
| Design rationale | 14 | 2 | 12/2 | Size and stability review required |
| Design Rules | 11 | 2 | 4/7 | Stability review required |
| Design Heuristics | 10 | 2 | 7/3 | Numerically passed; human granularity veto pending |
| Design frameworks | 6 | 2 | 2/4 | Size and stability review required |
| Design Procedures | 1 | 1 | 1 | Cannot be subdivided |

`all_283_keyword_conditioned_assignments.csv` contains one provisional
assignment for every retained paper. It must not yet be passed to topic
modeling as a fully frozen file: all groups marked for numeric or human review
still require a recorded validation decision.

Review materials:

- `keyword_cluster_profiles_for_review.csv`: all 26 candidate cluster profiles,
  including representative titles and diagnostic terms;
- `human_review_shortlist.csv`: 95 representative or boundary papers from
  groups that still need a decision.

The diagnostic TF-IDF terms in the profile file are not final topics. Final
cluster interpretation starts only after membership is frozen.
