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
Create an interpretive design-knowledge claim for this cluster, not only a topic phrase. The label should explain how the papers define, organize, translate, represent, adapt, evaluate, or use the relevant design-knowledge construct. Treat application domains and methods such as HCI, participatory design, co-design, or HRI as supporting facets unless they are central to the construct itself.

For context-oriented views, combine the context with the knowledge role. For example, prefer labels such as "Product-development knowledge as traceable design rationale" or "Studio and education contexts as sites for developing design knowledge" over generic labels such as "Education / Design knowledge" or "Synthesizes design knowledge".

The summary should answer:
1. What kind of design knowledge is being discussed?
2. What action does the cluster perform on that knowledge, such as defining, organizing, operationalizing, representing, or evaluating it?
3. How do the representative papers support that interpretation?

Return JSON:
{
  "label": "interpretive design-knowledge claim",
  "summary": "2-3 sentence design-knowledge contribution summary",
  "evidence_terms": ["term 1", "term 2", "term 3"]
}
```
