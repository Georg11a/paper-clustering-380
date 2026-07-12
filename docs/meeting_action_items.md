# Meeting Action Items Reflected In This Repo

This repo addresses the clustering-related action items from the team meeting.

## Implemented Direction

- Use the final advancing-paper CSV as the main input.
- Cluster papers per keyword instead of clustering all papers together.
- Skip or flag keyword groups that have too few papers for meaningful clustering.
- Treat k-means and HDBSCAN as the main methods.
- Keep DBSCAN only as an optional baseline.
- Preserve prompts and run instructions so the process is reviewable.

## Interface For Carter And Radwan

The most useful downstream file is:

```text
outputs/clustering_results.csv
```

Expected join keys:

- `paper_id`
- `doi`
- `url`
- `focus_keyword`
- `cluster_method`
- `cluster`

This lets citation graph code join paper metadata with keyword-local cluster assignments.
