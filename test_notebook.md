

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

from src.evaluator import ExperimentEvaluator
from sklearn.model_selection import LeaveOneOut
X, y = dm.get_X_y()
exp_evaluator = ExperimentEvaluator("RandomForest", LeaveOneOut())
exp_evaluator.evaluate(X, y)
```

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
split = loo.split(X, y)

for i, (train_index, test_index) in enumerate(split):
    print(X.iloc[train_index])
```


```python
a = (1, 2)
b, c = a
c
```

