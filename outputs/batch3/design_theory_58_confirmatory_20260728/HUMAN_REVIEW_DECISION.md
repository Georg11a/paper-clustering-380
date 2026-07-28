# Human-review decision

Date: 2026-07-28

## Corpus correction

`c1db52f69aae`, *Editorial: A Critical Look at Theories in Design Science
Research*, was excluded by corpus-scope decision because editorials are outside
the retained survey/book-chapter scope. Its PDF is preserved for audit. The
global analysis-ready corpus is now 283 publications, and the Design Theory
group contains 58.

The 2002 DRS and 2003 Design Studies versions of *Theory construction in
design research* remain separate publication records because their contents
are not identical:

- `c2c57ac5453e` — 2002 DRS conference version;
- `f56243f52335` — 2003 Design Studies journal version,
  DOI `10.1016/S0142-694X(03)00039-5`.

## Numerical confirmation after exclusion

Only two configurations passed all frozen numeric criteria:

| Configuration | Sizes | Pairwise bootstrap ARI | \(z_\rho\) |
|---|---:|---:|---:|
| Spectral knn15, k=2 | 41/17 | 0.611 | 3.861 |
| Spectral knn10, k=3 | 17/23/18 | 0.608 | 9.231 |

Spectral knn15, k=3 fell below the stability threshold with bootstrap ARI
0.579 and was rejected.

The retained-paper assignments for the eligible k=2 and k=3 configurations
were unchanged by removing the editorial (ARI 1.0 against their corresponding
59-paper results after restricting to the 58 retained papers).

## Intruder review

One internal reviewer completed the blinded title-based paper-intrusion task.
Two responses naming multiple choices were treated as unanswered.

Original scores:

- Spectral knn15, k=2: 7/10 = 70%;
- Spectral knn10, k=3: 9/15 = 60% overall, 9/14 = 64.3% among
  single-choice answers;
- Spectral knn15, k=3: 5/15 = 33.3% overall.

After excluding all questions that displayed the now-excluded editorial:

- Spectral knn15, k=2: 4/7 = 57.1%;
- Spectral knn10, k=3: 7/11 = 63.6%;
- Spectral knn15, k=3: 3/11 = 27.3%.

The review is an internal pipeline-freeze check, not a multi-reviewer
publication-grade validation study.

## Final decision

Spectral knn15, k=2 was vetoed as too coarse for organizing the survey.
Spectral knn10, k=3 was selected because it:

- passed every frozen numeric criterion;
- produced three balanced clusters of 17, 23, and 18 papers;
- provided the required finer survey granularity;
- outperformed the other k=3 candidate in the blinded intrusion task; and
- retained exactly the same memberships on the 58 common papers after the
  editorial exclusion.

The final configuration is therefore:

> BGE-M3 contextual embeddings → cosine 10-nearest-neighbor graph →
> Spectral clustering at k=3.
