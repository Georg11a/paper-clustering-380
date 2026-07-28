# Topic interpretation plan

The final stage begins only after the 282-paper keyword-conditioned assignment is
frozen. It explains fixed clusters; it does not discover or reassign clusters.
The current Path 1 run is explicitly provisional for every keyword group
except the already frozen Design Theory pilot.

## Statistical interpretation

For every fixed cluster:

1. concatenate the cluster's title, abstract, and keyword-conditioned
   subdocuments into one class document;
2. normalize whitespace and punctuation while preserving meaningful
   multi-word expressions;
3. use unigram, bigram, and trigram counts;
4. compute class-based TF-IDF across clusters within the same keyword group;
5. report the highest-weight contrastive terms and phrases;
6. identify representative papers and representative passages using the fixed
   BGE-M3 cluster centroid;
7. create a short evidence-linked statistical cluster descriptor.

The main output is one row per fixed cluster containing:

- keyword group and cluster ID;
- paper count;
- top c-TF-IDF unigrams, bigrams, trigrams, and weights;
- representative paper IDs and titles;
- representative passage IDs;
- overlap and distinctiveness diagnostics;
- a statistical descriptor assembled from the terms and evidence.

This is an adaptation of BERTopic's c-TF-IDF representation component. It is
not a complete BERTopic run because embedding and clustering have already been
fixed as BGE-M3 plus Spectral clustering.

### Term-display normalization

Raw statistical tokens are never presented directly as final topics.

- `DSR` is displayed as `design science research`.
- `UI` is displayed as `user interface`.
- `DR` is expanded to `design rationale` only inside that keyword group;
  otherwise ambiguous bare `DR` is removed.
- `HCI` and `AI` may remain as abbreviations.
- citation fragments, years, and generic procedural terms such as `et al`,
  `use`, `used`, and `project` are removed.
- meaningful multiword expressions such as `design science research`,
  `design rationale`, and `design knowledge` are preserved before the generic
  word `design` is filtered.

The first run on a newly regenerated assignment may be marked provisional.
Provisional c-TF-IDF output is suitable for inspecting granularity and lexical
quality, but it must not be reported as a final topic result until the
underlying cluster membership is frozen.

## LLM interpretation

The Path 2 comparison receives the **same fixed cluster memberships** and the
same evidence budget. An LLM reads representative and boundary evidence,
proposes a cluster label and summary, cites supporting paper/passage IDs, and
states uncertainty or internal variation.

The statistical and LLM interpretations are then compared blindly on:

- granularity;
- coherence;
- coverage;
- distinctiveness;
- faithfulness to source evidence;
- usefulness for the survey.

## Methodological references

- Grootendorst (2022) introduces BERTopic and its class-based TF-IDF
  representation.
- Chagnon et al. (2024), *Benchmarking topic models on scientific articles
  using BERTeley*, supports scientific-article preprocessing, model comparison,
  and multi-metric evaluation.
- Ajinaja et al. (published online 2025; version of record 2026) compares LDA,
  LDA2Vec, Top2Vec, and BERTopic across short, long, domain-specific, and
  multilingual corpora and combines quantitative with human-centered
  evaluation.
- Nikbakht and Zojaji (2026), *Fuzzy BERTopic*, is relevant to the limitation
  that hard assignments do not represent every secondary topic of a document.
  It does not justify changing the primary hard assignment required by the
  downstream citation graph.
