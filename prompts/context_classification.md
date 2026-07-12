# Context Classification Prompt Template

Use this prompt when assigning each paper to a domain, setting, or application context for secondary analysis views.

```text
You are classifying the application context of a research paper for a design-knowledge literature review.

Paper title: {title}
Abstract: {abstract}
Keyword: {keyword}

Choose up to three context labels that best describe the domain, setting, or application area of the paper. Use concise labels such as healthcare, education, accessibility, software engineering, sustainability, games, XR, design practice, public sector, or human-AI collaboration.

Return JSON:
{
  "contexts": ["label 1", "label 2"],
  "rationale": "one sentence grounded in the abstract"
}
```
