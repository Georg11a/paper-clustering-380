# K-Means Silhouette Selection

- Requested k range: 2-6
- Dynamic k range after paper-count cap: 2-6
- Selected k: 3
- Selected average silhouette score: 0.0383
- Distance metric for silhouette: cosine
- Selection note: Selected k is the smallest stable candidate after silhouette, size, and balance checks.

Silhouette analysis evaluates whether each paper is closer to papers in its own cluster than to papers in neighboring clusters. It is a diagnostic signal for k selection, not a replacement for human interpretation.