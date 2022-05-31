The models in this file were acquired as follows:

**XGBClassifier**
1. Create conda environment with desired xgboost version.
2. Load model from `Gremlin/test/data/data_api/ar_data/ar_model_xgbclassifier.bin`.
Use the `bin` format as xgboost guarantees backwards-compatibility.
```
import pickle
model = xgboost.XGBClassifier()
model.load_model('ar_model_xgbclassifier.bin')
model.save_model('xgbclassifier_<major_version>_<minor_version>.json')
```

**XGBRegressor**
1. Create conda environment with desired xgboost version
2. Use script `ForestRelaxation/scripts/xgbregressor_train.py`. Modify
the save file name as `xgbregressor_<major_version>_<minor_version>. 
