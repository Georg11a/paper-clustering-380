# Paragraph Selection Prompt Template

Use this prompt when selected full-text paragraphs are available and the clustering run should focus on one design-knowledge keyword at a time.

```text
You are helping prepare text evidence for a literature-review clustering pipeline.

Keyword focus: {keyword}

Task:
Select the paragraphs that are most relevant to the keyword focus. Prefer paragraphs that define, compare, operationalize, evaluate, or synthesize the keyword concept. Avoid generic background paragraphs unless they contain an explicit definition or methodological use of the keyword.

Return:
1. The selected paragraph ids.
2. A one-sentence reason for each selection.
3. A short note if no paragraph directly addresses the keyword.
```

Current code path:

- If an `extracted_context` column is present, `scripts/cluster_papers.py` can re-rank those paragraphs for a `--focus-keyword` run.
- If only abstracts are available, the pipeline clusters title + abstract + keyword metadata as the first-pass representation.
