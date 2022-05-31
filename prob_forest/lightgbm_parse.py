# Copyright (c) 2021-present, Royal Bank of Canada.
# Copyright (c) 2018-present, Scott Lundberg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on SHAP
# from https://github.com/slundberg/shap/blob/master/shap/explainers/_tree.py
# by Scott Lundberg
#################################################################################### 

import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from lightgbm import LGBMClassifier
from lightgbm.basic import Booster
from sklearn.tree._tree import TREE_UNDEFINED
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


DummyVarDict = Dict[int, Dict[str, str]]
Params = Dict[str, Union[np.ndarray, list]]
ParsedLightGBM = Tuple[int, int, Params, DummyVarDict, int]


def parse_lightgbm_tree_with_dummy_variables(
    tree: Dict[str, Any],
    dummy_variables_num: int,
    running_dict_with_dummy_features: DummyVarDict,
    scaling: float = 1.0,
    verbose: bool = False,
) -> Tuple[Params, int, DummyVarDict]:
    assert type(tree) == dict
    assert "tree_structure" in tree

    start = tree["tree_structure"]
    num_parents = tree["num_leaves"] - 1
    children_left = np.empty((2 * num_parents + 1), dtype=np.int32)
    children_right = np.empty((2 * num_parents + 1), dtype=np.int32)
    children_default = np.empty((2 * num_parents + 1), dtype=np.int32)
    features = np.empty((2 * num_parents + 1), dtype=np.int32)
    thresholds = np.empty((2 * num_parents + 1), dtype=np.float64)
    values = [-2] * (2 * num_parents + 1)
    node_sample_weight = np.empty((2 * num_parents + 1), dtype=np.float64)
    visited: List[str] = []
    queue = [start]
    while queue:
        vertex = queue.pop(0)
        if "split_index" in vertex.keys():
            if vertex["split_index"] not in visited:
                if "split_index" in vertex["left_child"].keys():
                    children_left[vertex["split_index"]] = vertex[
                        "left_child"
                    ]["split_index"]
                else:
                    children_left[vertex["split_index"]] = (
                        vertex["left_child"]["leaf_index"] + num_parents
                    )
                if "split_index" in vertex["right_child"].keys():
                    children_right[vertex["split_index"]] = vertex[
                        "right_child"
                    ]["split_index"]
                else:
                    children_right[vertex["split_index"]] = (
                        vertex["right_child"]["leaf_index"] + num_parents
                    )
                if vertex["default_left"]:
                    children_default[vertex["split_index"]] = children_left[
                        vertex["split_index"]
                    ]
                else:
                    children_default[vertex["split_index"]] = children_right[
                        vertex["split_index"]
                    ]

                if isinstance(
                    vertex["threshold"], str
                ):  # and '||' in vertex['threshold']:
                    if verbose:
                        logger.info(
                            f"found a categorical feature "
                            f"{vertex['threshold']}, "
                            f"replace it with the dummy "
                            f"variable with index {dummy_variables_num}"
                        )

                    features[vertex["split_index"]] = dummy_variables_num
                    thresholds[vertex["split_index"]] = 0.5
                    # check that we don't override dummy_features
                    if dummy_variables_num in running_dict_with_dummy_features:
                        raise ValueError(
                            f"trying to override "
                            f"the {dummy_variables_num} dummy variables"
                        )
                    # update running dict with new dummy feature
                    running_dict_with_dummy_features[dummy_variables_num] = {
                        "split_feature": vertex["split_feature"],
                        "threshold": vertex["threshold"],
                    }
                    # update number for dummy_variables
                    dummy_variables_num += 1
                else:
                    # normal parsing
                    features[vertex["split_index"]] = vertex["split_feature"]
                    thresholds[vertex["split_index"]] = vertex["threshold"]

                values[vertex["split_index"]] = [vertex["internal_value"]]
                node_sample_weight[vertex["split_index"]] = vertex[
                    "internal_count"
                ]
                visited.append(vertex["split_index"])
                queue.append(vertex["left_child"])
                queue.append(vertex["right_child"])
        else:
            children_left[vertex["leaf_index"] + num_parents] = -1
            children_right[vertex["leaf_index"] + num_parents] = -1
            children_default[vertex["leaf_index"] + num_parents] = -1
            features[vertex["leaf_index"] + num_parents] = -1
            children_left[vertex["leaf_index"] + num_parents] = -1
            children_right[vertex["leaf_index"] + num_parents] = -1
            children_default[vertex["leaf_index"] + num_parents] = -1
            features[vertex["leaf_index"] + num_parents] = -1
            thresholds[vertex["leaf_index"] + num_parents] = -1
            values[vertex["leaf_index"] + num_parents] = [vertex["leaf_value"]]
            node_sample_weight[vertex["leaf_index"] + num_parents] = vertex[
                "leaf_count"
            ]
    values = np.asarray(values)  # type: ignore
    values = np.multiply(values, scaling)  # type: ignore

    param: Params = {}
    param["feature"] = features
    param["threshold"] = thresholds
    param["children_left"] = children_left
    param["children_right"] = children_right
    param["value"] = values  # type: ignore

    return param, dummy_variables_num, running_dict_with_dummy_features


def _depth_of_free(tree_structure: Dict[str, Any]) -> int:
    if "left_child" in tree_structure or "right_child" in tree_structure:
        l_depth = _depth_of_free(tree_structure["left_child"])
        r_depth = _depth_of_free(tree_structure["right_child"])
        if l_depth > r_depth:
            return l_depth + 1
        else:
            return r_depth + 1
    else:
        return 0


def parse_lightgbm(
    lgbm_classifier: LGBMClassifier, use_padding: bool = True
) -> ParsedLightGBM:
    booster = lgbm_classifier.booster_

    return parse_lightgbm_booster(booster)


def parse_lightgbm_booster(
    booster: Booster, use_padding: bool = True
) -> ParsedLightGBM:

    n_feature = booster.num_feature()
    # dummy variables feature id starts from n_feature + 1
    # for example: we have 40 features in our model,
    # new dummy features will be from 41

    dummy_variables_num = n_feature
    running_dict_with_dummy_features: DummyVarDict = {}
    tree_info = booster.dump_model()["tree_info"]

    param: Params = {}
    param["feature"] = []
    param["threshold"] = []
    param["children_left"] = []
    param["children_right"] = []
    param["value"] = []

    depths = []

    for tree in tree_info:
        (
            _param,
            dummy_variables_num,
            running_dict_with_dummy_features,
        ) = parse_lightgbm_tree_with_dummy_variables(
            tree,
            dummy_variables_num=dummy_variables_num,
            running_dict_with_dummy_features=running_dict_with_dummy_features,
        )

        for key, val in _param.items():
            if key != "value":
                param[key].append(torch.Tensor(val[:, None]))  # type: ignore
            else:
                param[key].append(torch.Tensor(val))  # type: ignore

        # TODO optimize this later
        _depth = _depth_of_free(tree["tree_structure"])
        depths.append(_depth)

    max_depth = max(depths)

    if use_padding:
        for key, val in param.items():
            if key != "value":
                padded = pad_sequence(val, padding_value=TREE_UNDEFINED)  # type: ignore # noqa: E501
                padded = padded.permute(1, 0, 2)
                param[key] = padded.squeeze(2)  # type: ignore
            else:
                padded = pad_sequence(val, padding_value=TREE_UNDEFINED)  # type: ignore # noqa: E501
                param[key] = padded.permute(1, 0, 2)  # type: ignore
    else:
        for key, val in param.items():
            if key != "value":
                param[key] = [vi for vi in val]  # type: ignore

    return (
        dummy_variables_num,
        max_depth,
        param,
        running_dict_with_dummy_features,
        dummy_variables_num,
    )


def non_categorical_parse_lightgbm(
    lgbm_classifier: LGBMClassifier, use_padding: bool = True
) -> Tuple[int, int, Params]:
    booster = lgbm_classifier.booster_
    n_total_feature, max_depth, parsed_params, _, _ = parse_lightgbm_booster(
        booster, use_padding=True
    )

    return n_total_feature, max_depth, parsed_params
