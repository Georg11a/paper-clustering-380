# K-Means Silhouette Selection

- Requested k range: 2-8
- Dynamic k range after paper-count cap: 2-8
- Selected k: 7
- Selected average silhouette score: 0.1778
- Distance metric for silhouette: cosine
- Selection note: Selected k is the highest-silhouette candidate; when scores are nearly tied, the smaller k is used for readability.

Silhouette analysis evaluates whether each paper is closer to papers in its own cluster than to papers in neighboring clusters. It is a diagnostic signal for k selection, not a replacement for human interpretation.