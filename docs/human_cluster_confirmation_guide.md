# Human confirmation of keyword-conditioned clusters

## What is being confirmed

The computational pipeline has assigned all 282 retained publications within
their predefined keyword groups. Human review does not create topics and does
not compare embedding models again. It checks whether the proposed groupings
are substantively defensible for the survey.

Design Theory has already been frozen. The remaining candidate clusters are
reviewed in two passes.

## Files

1. `human_review_clusters.csv` contains one row for each cluster that needs a
   decision.
2. `human_review_shortlist.csv` contains representative and boundary papers
   from those clusters.
3. `keyword_cluster_profiles_for_review.csv` provides diagnostic terms and
   representative titles. These terms are not final topic labels.

An LLM pre-screen may help identify suspicious cases, but it must not be
recorded as the independent human judgment.

## Pass 1: cluster-level review

Read the representative titles for one cluster and fill:

- `reviewer_cluster_coherent_yes_no`
  - `yes`: the papers share a recognizable substantive focus;
  - `no`: the cluster mixes unrelated themes.
- `reviewer_granularity_too_coarse_ok_too_fine`
  - `too_coarse`: two or more clearly distinct themes were merged;
  - `ok`: the cluster is useful at the survey's intended level;
  - `too_fine`: the split creates groups that are too small or conceptually
    indistinguishable.
- `reviewer_accept_current_membership_yes_no`
  - `yes`: no systematic membership problem is apparent;
  - `no`: several papers appear misplaced.
- `reviewer_accept_numeric_exception_yes_no`
  - complete this only when `assignment_status` says the numeric rule failed;
  - `yes` requires a short substantive rationale in `reviewer_notes`;
  - `no` means the cluster configuration cannot be frozen as-is.

Do not judge a cluster from c-TF-IDF terms alone. Use the representative titles
and open papers when a title is ambiguous.

## Pass 2: representative and boundary papers

In `human_review_shortlist.csv`, review:

- representative papers, which indicate the central meaning of a cluster;
- boundary papers, which have the smallest margin over another cluster.

Fill:

- `reviewer_accept_membership_yes_no`;
- `reviewer_suggested_cluster_id` when the answer is `no`;
- the cluster coherence and granularity fields when the paper exposes a
  cluster-level problem;
- `reviewer_notes` with a short reason.

The reviewer does not need to read all 282 papers. Read the full abstract or
PDF only when the title and extracted context are insufficient.

## Freeze rule

A candidate configuration can be frozen only when:

1. its clusters are judged coherent;
2. granularity is judged `ok`;
3. current membership is accepted, or every rejected boundary paper has a
   recorded resolution;
4. any numeric-rule exception is explicitly accepted with a rationale.

If a small keyword group produces clusters that are too fine, the defensible
decision is `k=1` (report the keyword group without subclustering), not a
forced two-cluster result.

## After review

Save the completed CSVs without changing paper IDs, cluster IDs, numeric
diagnostics, or assignment-status fields. The reviewed decisions can then be
used to freeze the accepted assignments and rerun Path 1 statistical
interpretation as a final rather than provisional output.
