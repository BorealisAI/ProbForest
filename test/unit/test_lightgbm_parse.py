# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
import torch
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification

from prob_forest.activations import binary_logits
from prob_forest.ensemble import SampledPathEnsemble
from prob_forest.lightgbm_parse import (
    DummyVarDict,
    parse_lightgbm,
    parse_lightgbm_tree_with_dummy_variables,
)

DatasetType = Tuple[pd.DataFrame, np.ndarray]


@pytest.fixture(scope="session")
def lgbm_train_dataset() -> DatasetType:
    n_features = 20
    n_samples = 300
    n_informative = 10
    n_classes = 2
    random_state = 0

    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        random_state=random_state,
    )

    x = pd.DataFrame(x)
    # replace one feature with cat feature
    cat_values = [str(x) for x in range(10)]
    cat_features = [random.choice(cat_values) for x in y]
    for i in range(0, n_features):
        x[i] = cat_features
        x[i] = x[i].astype("category")
    return x, y


@pytest.fixture(scope="session")
def lgbm_classifier_with_cat(
    lgbm_train_dataset: DatasetType,
) -> LGBMClassifier:
    x, y = lgbm_train_dataset
    clf = LGBMClassifier()
    clf.fit(x, y)
    return clf


class DummyVarTransformer:
    def __init__(
        self,
        categorical_feats: List[int],
        columns: pd.RangeIndex,
        dummy_var_dict: DummyVarDict,
    ):
        self.categorical_feats = categorical_feats
        self.columns = columns

        self.new_features = sorted(dummy_var_dict)
        self.split_feature = [
            dummy_var_dict[k]["split_feature"] for k in self.new_features
        ]
        self.threshold = [
            set(dummy_var_dict[k]["threshold"].split("||"))
            for k in self.new_features
        ]

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x_out = x.copy()
        for i, ind in enumerate(self.new_features):
            f_ind, t = self.split_feature[i], self.threshold[i]
            f = self.columns[f_ind]
            t_str = "||".join(sorted(t))
            dummy_feat_name = "dummy_feat-{}-{}-{}".format(i, f, t_str)
            x_out[dummy_feat_name] = (~x_out[f].isin(t)).astype(int)

        x_out[self.categorical_feats] = 0.0

        return x_out


def test_model_categorical_splits_with_dummy_variables(
    lgbm_classifier_with_cat: LGBMClassifier,
) -> None:
    (n_feature,) = lgbm_classifier_with_cat.feature_importances_.shape

    dummy_variables_num = n_feature + 1
    running_dict_with_dummy_features: DummyVarDict = {}
    booster = lgbm_classifier_with_cat.booster_
    tree_info = booster.dump_model()["tree_info"]

    for tree in tree_info:
        (
            _param,
            dummy_variables_num,
            running_dict_with_dummy_features,
        ) = parse_lightgbm_tree_with_dummy_variables(
            tree,
            dummy_variables_num=dummy_variables_num,
            running_dict_with_dummy_features=running_dict_with_dummy_features,
            verbose=True,
        )

        assert "feature" in _param
        assert "threshold" in _param
        assert "children_left" in _param
        assert "children_right" in _param


def test_parse_lightgbm(
    lgbm_classifier_with_cat: LGBMClassifier, lgbm_train_dataset: DatasetType,
) -> None:
    (
        n_feature,
        max_depth,
        param,
        running_dict_with_dummy_features,
        dummy_variables_num,
    ) = parse_lightgbm(lgbm_classifier_with_cat, use_padding=True,)

    model = SampledPathEnsemble(
        n_feature, max_depth, activation=binary_logits, source="lightgbm"
    )
    model.set_param(param)

    x, _ = lgbm_train_dataset
    var_transformer = DummyVarTransformer(
        x.columns.values.tolist(), x.columns, running_dict_with_dummy_features
    )

    new_df = var_transformer.transform(x)
    new_x = torch.FloatTensor(new_df.values)  # type: ignore
    hard_tree_pred = lgbm_classifier_with_cat.predict_proba(x)
    soft_tree_pred = model.predict(new_x)
    assert np.allclose(hard_tree_pred, soft_tree_pred.numpy())
