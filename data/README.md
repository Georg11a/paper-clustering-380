# Data

The main input for the current pipeline is `final_advancing_list.csv`, the advancing-paper list produced by the screening pipeline.

Required columns:

- `paper_id`
- `title`
- `abstract`
- `keyword`

Recommended metadata columns:

- `authors`
- `doi`
- `year`
- `venue`
- `url`
- `fulltext_flag`
- `primary_reason`
- `val_reason`

The source file is maintained in the UVA VIDAR Lab design-knowledge-review repository:

<https://github.com/UVA-VIDAR-Lab/design-knowledge-review/blob/main/screener/pipeline_outputs/final_advancing_list.csv>

Download the raw CSV and save it here as:

```text
data/final_advancing_list.csv
```

The pipeline treats this CSV as the current final working set, regardless of whether meeting notes describe it as approximately 380, 386, or 400 papers.
