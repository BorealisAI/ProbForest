# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import json
import logging
from typing import Dict, Tuple

import lightgbm as lgb
import torch
from sklearn.base import is_classifier
from sklearn.ensemble._forest import BaseForest
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
from torch.nn.utils.rnn import pad_sequence
from xgboost.sklearn import XGBModel

from .activations import (
    get_lightgbm_activation,
    get_sklearn_activation,
    get_xgboost_activation,
)
from .ensemble import (
    ExhaustiveEnsemble,
    ListEnsemble,
    SampledPathEnsemble,
    TreeEnsemble,
)
from .lightgbm_parse import non_categorical_parse_lightgbm as parse_lightgbm
from .utils import reindex

_RELAXATIONS = {
    None: TreeEnsemble,
    "sampling": SampledPathEnsemble,
    "exhaustive": ExhaustiveEnsemble,
}

logger = logging.getLogger(__name__)


def parse_nested_dict(tree_dict):  # type: ignore
    nodeid, leafid = [], []

    param = {}
    param["feature"] = []
    param["threshold"] = []
    param["children_left"] = []
    param["children_right"] = []
    param["value"] = []

    depths = []

    def _dfs(tree, depth, is_left):  # type: ignore
        if "leaf" in tree:
            nodeid.append(tree["nodeid"])
            leafid.append(tree["nodeid"])
            param["value"].append(tree["leaf"])

            # Is this needed? Just set the default value
            # Better yet, set it to -2
            param["children_left"].append(TREE_LEAF)
            param["children_right"].append(TREE_LEAF)
            param["feature"].append(TREE_LEAF)
            param["threshold"].append(TREE_LEAF)

            depths.append(depth)
        else:
            _nodeid, children = tree["nodeid"], tree["children"]
            feature, threshold = tree["split"], tree["split_condition"]

            if feature[0] == "f":
                feature = int(feature[1:])
            else:
                feature = int(feature)

            node0, node1 = children[0]["nodeid"], children[1]["nodeid"]

            if (node0 == tree["yes"]) and (node1 == tree["no"]):
                left_subtree = children[0]
                right_subtree = children[1]
            elif (node1 == tree["yes"]) and (node0 == tree["no"]):
                left_subtree = children[1]
                right_subtree = children[0]
            else:
                raise ValueError("nodeids do not match!")

            nodeid.append(_nodeid)
            param["children_left"].append(left_subtree["nodeid"])
            param["children_right"].append(right_subtree["nodeid"])

            param["feature"].append(feature)
            param["threshold"].append(threshold)

            _dfs(left_subtree, depth=depth + 1, is_left=1)
            _dfs(right_subtree, depth=depth + 1, is_left=0)

    _dfs(tree_dict, 0, is_left=0)

    # The above is a depth-first traversal of the tree
    # However, XGBOOST uses breadth-first ordering of its leaves
    # We can use nodeid to reindex the params
    for k, v in param.items():
        if k == "value":
            param[k] = reindex(leafid, v)[:, None]
        else:
            param[k] = reindex(nodeid, v)[:, None]

    nodeid = sorted(nodeid)

    return param, max(depths), nodeid, leafid


def parse_xgboost_sklearn(
    forest: XGBModel, use_padding: bool = True
) -> Tuple[int, int, Dict]:
    """
    Parser for models using the SkLearn API of the XGBoost library.

    Args:
        forest:
            Instance of xgboost.sklearn.XGBModel to parse.
        use_padding:
            Pads returned parsed structured with TREE_UNDEFINED.
            Defaults to True.

    Returns:
        Tuple representing number of features in model, maximum depth of
        the tree and a dictionary of parsed parameters from model.
    """
    logger.info(f"Parsing model {forest} using the SkLearn API.")

    n_features = forest.n_features_in_
    logger.debug(f"Found {n_features} features in model {forest}")

    booster = forest.get_booster()

    dump = booster.get_dump(dump_format="json")
    tree_list = [json.loads(tree) for tree in dump]

    param = {}  # type: ignore
    param["feature"] = []
    param["threshold"] = []
    param["children_left"] = []
    param["children_right"] = []
    param["value"] = []

    depths = []
    node_count = []

    for t in tree_list:
        _param, _depth, nodeid, leafid = parse_nested_dict(t)
        depths.append(_depth)
        node_count.append(len(nodeid))

        for key, val in _param.items():
            param[key].append(val)

    max_depth = max(depths)

    if use_padding:
        for key, val in param.items():
            padded = pad_sequence(val, padding_value=TREE_UNDEFINED)
            padded = padded.permute(1, 0, 2)
            if key != "value":
                param[key] = padded.squeeze(2)
            else:
                param[key] = padded
    else:
        for key, val in param.items():
            if key != "value":
                param[key] = [vi.squeeze(-1) for vi in val]  # type: ignore

    return n_features, max_depth, param


def parse_sklearn(forest, use_padding=True):  # type: ignore
    n_feature = forest.n_features_

    param = {}
    param["feature"] = []
    param["threshold"] = []
    param["children_left"] = []
    param["children_right"] = []
    param["value"] = []

    depths = []
    node_count = []

    for clf in forest.estimators_:
        depths.append(clf.tree_.max_depth)
        for key in param:
            tensor = torch.Tensor(clf.tree_.__getattribute__(key))
            if key != "value":
                param[key].append(tensor[:, None])
            else:
                if is_classifier(clf):
                    val = tensor.squeeze(1)
                    val = val / val.sum(-1)[:, None] / forest.n_estimators
                else:
                    val = tensor.squeeze(2) / forest.n_estimators

                param["value"].append(val)

        node_count.append(len(param[key][-1]))

    max_depth = max(depths)

    if use_padding:
        for key, val in param.items():
            padded = pad_sequence(val, padding_value=TREE_UNDEFINED)
            padded = padded.permute(1, 0, 2)
            if key != "value":
                param[key] = padded.squeeze(2)
            else:
                param[key] = padded
    else:
        for key, val in param.items():
            if key != "value":
                param[key] = [vi.squeeze(-1) for vi in val]

    return n_feature, max_depth, param


def parse_model(  # type: ignore
    forest, relaxation="sampling", use_padding=True, **kwargs
):
    """
    Given an ensemble of trees trained with an external library, construct
    a TreeEnsemble with the specified relaxation

    Parameters
    ----------
    forest :
        Ensemble of trees.  Only sklearn and xgboost models are currently
        supported

    relaxation : str
        Type of relaxation to use.
        Must be one of None, 'sampling', or 'exhaustive'

    use_padding : bool
        If true, pad the parameters of each tree to a constant size,
        and concatenate them into a single model.

        Otherewise, represent the forest as a list of single trees.

    Returns
    -------
    model :
        Pytorch tree ensemble of type specified by relaxation

    Raises
    ------
    ValueErrror
        If forest or relaxation are of an unsupported type.
    """
    # optional: add 'activation' as a keyword
    try:
        model_cls = _RELAXATIONS[relaxation]
    except KeyError:
        raise ValueError(
            "{} is a presently unsupported relaxation strategy".format(
                relaxation
            )
        )

    # A note on activations:
    # It is assumed that the activations are summed
    if isinstance(forest, BaseForest):
        parse_fn = parse_sklearn
        source = "sklearn"
        activation_fn = get_sklearn_activation(forest)
    elif isinstance(forest, lgb.LGBMModel):
        parse_fn = parse_lightgbm
        source = "lightgbm"
        activation_fn = get_lightgbm_activation(forest)
    elif isinstance(forest, XGBModel):
        parse_fn = parse_xgboost_sklearn
        source = "xgboost"
        activation_fn = get_xgboost_activation(forest)
    else:
        raise ValueError(
            "{} models are not presently supported".format(type(forest))
        )

    n_feature, max_depth, param = parse_fn(forest, use_padding=use_padding)
    # use_padding is the default behaviour
    # the ListEnsemble is used for testing purposes
    # but in future one might also use it for mememory considerations?
    if use_padding:
        model = model_cls(
            n_feature,
            max_depth,
            activation=activation_fn,
            source=source,
            **kwargs,
        )
        model.set_param(param)
    else:
        model_list = []

        for i in range(len(param["feature"])):
            tree_param = {k: v[i] for k, v in param.items()}
            model_i = model_cls(
                n_feature,
                max_depth,
                activation=activation_fn,
                source=source,
                **kwargs,
            )
            model_i.set_param(tree_param)
            model_list.append(model_i)

        model = ListEnsemble(model_list, activation_fn)

    return model
