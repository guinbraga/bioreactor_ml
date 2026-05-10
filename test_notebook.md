

```python
scalers = {
    "CLRTransformer": CLRTransformer()
}

scalers.values()
```

```python
import pandas as pd
asv_df = pd.read_csv("../04_bioreactor_ml_project/data/map_complete_absolute_n_hits_table.csv", index_col=0).T
metadata_df = pd.read_csv("../04_bioreactor_ml_project/data/metadata_enrichment.csv",
                          index_col="Cluster sample ID")["EXPERIMENT"]
merged = pd.merge(asv_df, metadata_df, left_index=True, right_index=True)
type(merged["EXPERIMENT"])
```

```python
len(merged)
```

