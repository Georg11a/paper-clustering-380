# Cleaned 282-paper keyword-conditioned clustering

Date: 2026-07-28

This run supersedes the earlier 283-paper candidate run.

## Corpus change

- Retained `242644458c6e`, the 2001 AI & Society journal article
  `A Pattern Approach to Interaction Design`.
- Excluded `b8507dbe97ca`, the earlier DIS 2000 conference version.
- Preserved the excluded PDF and recorded the publication-family decision in
  `data/publication_version_reviews.csv`.
- Repaired confirmed encoding corruption in titles, abstracts, and one
  subdocument before refreshing affected BGE-M3 vectors.

## Method

- clustering scope: within predefined keyword groups;
- representation: BGE-M3, L2 normalized, using title, abstract, and up to 12
  keyword-conditioned PDF passages;
- k estimate: K-means inertia elbow;
- assignment: Spectral clustering on a cosine 10-nearest-neighbor graph;
- Design Theory: previously frozen 58-paper assignment reused;
- all other groups: candidate assignments pending human confirmation.

## Current groups

| Keyword | n | k | Sizes | Status |
|---|---:|---:|---:|---|
| Design knowledge | 68 | 3 | 23/37/8 | Numeric exception requires review |
| Design theory | 58 | 3 | 17/23/18 | Frozen |
| Design Patterns | 50 | 3 | 14/15/21 | Numeric exception requires review |
| Design methods | 27 | 3 | 17/4/6 | Numeric exception requires review |
| Design Guidelines | 22 | 4 | 3/4/6/9 | Numeric exception requires review |
| Design principles | 15 | 2 | 10/5 | Numeric exception requires review |
| Design rationale | 14 | 2 | 12/2 | Numeric exception requires review |
| Design Rules | 11 | 2 | 2/9 | Numeric exception requires review |
| Design Heuristics | 10 | 2 | 7/3 | Numeric pass; human veto pending |
| Design frameworks | 6 | 2 | 2/4 | Numeric exception requires review |
| Design Procedures | 1 | 1 | 1 | No subclustering |

## Important interpretation

The computational assignment is complete. On 2026-07-28, all 282-paper
assignments were frozen for downstream analysis so that topic-interpretation
methods can be compared against identical memberships. This is an analysis
freeze, not a claim that every cluster has already been human-validated.

Design Theory retains its previously frozen 58-paper assignment. The other
keyword groups require confirmation by at least two reviewers before they are
reported as human-validated results. See
`docs/human_cluster_confirmation_guide.md`.

The cleanup materially changed several candidate solutions. In particular,
Design Patterns changed from k=2 to k=3. This is why the old 283-paper review
cannot simply be copied onto the cleaned run.
