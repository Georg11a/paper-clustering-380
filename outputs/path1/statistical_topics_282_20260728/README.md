# Path 1: provisional statistical cluster interpretation

This directory contains an adapted class-based TF-IDF interpretation of the
cleaned 282-paper candidate assignment.

Path 1 does not form clusters and does not change paper membership. It treats
the papers assigned to each cluster as a class document and extracts terms
that distinguish that cluster from sibling clusters in the same keyword
group.

## Outputs

- `cluster_statistical_interpretations.csv`: one row per cluster, with
  statistical descriptors, terms, representative papers, and representative
  passage IDs;
- `cluster_term_weights.csv`: long-form ranked c-TF-IDF terms and weights;
- `run_metadata.json`: method and status metadata.

## Display cleanup

- `design science research`, `design rationale`, and `user interface` are
  displayed in full;
- only HCI and AI remain as abbreviations;
- years, citation fragments, generic procedural words, ambiguous `DR`, and
  detected unexplained acronyms are removed;
- keyword anchor terms are removed from the descriptor so that the output
  emphasizes within-keyword differences.

## Status

Only the three Design Theory interpretations are based on frozen membership.
The other 24 interpretations are provisional and are intended to support
granularity and coherence review. They must be rerun after human confirmation
before being reported as final topic results.
