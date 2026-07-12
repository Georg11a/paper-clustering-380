# Cluster Labeling Prompt Template

Use this prompt after each per-keyword clustering run to turn representative papers and keyphrases into human-readable labels.

```text
You are labeling a cluster from a design-knowledge literature-review pipeline.

Keyword group: {keyword}
Cluster id: {cluster_id}
Top c-TF-IDF phrases: {theme_terms}
Representative papers:
{representative_papers}

Task:
Create a concise label and one-sentence summary for this cluster. The label should describe the substantive theme, not simply repeat the keyword group. Ground the summary in the representative paper titles and theme terms.

Return JSON:
{
  "label": "short cluster label",
  "summary": "one sentence",
  "evidence_terms": ["term 1", "term 2", "term 3"]
}
```
