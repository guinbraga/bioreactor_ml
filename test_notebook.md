

```python
scalers = {
    "CLRTransformer": CLRTransformer()
}

scalers.values()
```

```python
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)
from src.data_manager import DataManager
dm = DataManager(
    genomic_file_path="../04_bioreactor_ml_project/data/map_complete_absolute_n_hits_table.csv",
    metadata_file_index="Cluster sample ID",
    metadata_file_path="../04_bioreactor_ml_project/data/metadata_enrichment_clean.csv",
    target_column="EXPERIMENT"
)
dm.merge_datasets()

from src.experiment_evaluator import ExperimentEvaluator
from sklearn.model_selection import LeaveOneOut
from optuna import logging
logging.set_verbosity(logging.WARNING)
X, y = dm.get_X_y()
exp_evaluator = ExperimentEvaluator("RandomForest", LeaveOneOut(), "../04_bioreactor_ml_project/results/")
exp_evaluator.evaluate(X, y)
```

```python
X.iloc[1].name
```

```python
rows_list = []
for i in range(5):
    rows_list.append({'A': i, 'B': i * 2})

# Final concatenation
df = pd.concat([df, pd.DataFrame(rows_list)], ignore_index=True)
df
```


```python
a = (1, 2)
b, c = a
c
```

