# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from pathlib import Path

import xgboost

from prob_forest.parse import parse_xgboost_sklearn


class TestXGBoostSkLearnParserForXGBClassifier:
    def test_parser_can_parse_xgbclassifier_1_4(self, data_path: Path) -> None:
        xgbclassifier = xgboost.XGBClassifier()
        xgbclassifier.load_model(data_path / "xgbclassifier_1_4.json")

        (n_features, max_depth, params) = parse_xgboost_sklearn(xgbclassifier)

        assert n_features == 30
        assert max_depth == 2
        assert params is not None

    def test_parser_can_parse_xgbclassifier_1_3(self, data_path: Path) -> None:
        xgbclassifier = xgboost.XGBClassifier()
        xgbclassifier.load_model(data_path / "xgbclassifier_1_3.json")

        (n_features, max_depth, params) = parse_xgboost_sklearn(xgbclassifier)

        assert n_features == 30
        assert max_depth == 2
        assert params is not None

    def test_parser_can_parse_xgbregressor_1_4(self, data_path: Path) -> None:
        xgbregressor = xgboost.XGBRegressor()
        xgbregressor.load_model(data_path / "xgbregressor_1_4.json")

        (n_features, max_depth, params) = parse_xgboost_sklearn(xgbregressor)

        assert n_features == 13
        assert max_depth == 6
        assert params is not None

    def test_parser_can_parse_xgbregressor_1_3(self, data_path: Path) -> None:
        xgbregressor = xgboost.XGBRegressor()
        xgbregressor.load_model(data_path / "xgbregressor_1_3.json")

        (n_features, max_depth, params) = parse_xgboost_sklearn(xgbregressor)

        assert n_features == 13
        assert max_depth == 6
        assert params is not None
