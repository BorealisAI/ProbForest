# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
from sklearn.base import is_classifier
from torch.nn import functional as F


def binary_logits(x):  # type: ignore
    classoneprob = torch.sigmoid(x.sum(-1).sum(-1))
    return torch.stack((1.0 - classoneprob, classoneprob)).T


def multi_softmax(x, n_class):  # type: ignore
    x = x.squeeze(-1)
    n_sample, _ = x.shape
    output = x.reshape(n_sample, -1, n_class).sum(1)
    return F.softmax(output, -1)


def get_sklearn_activation(forest):  # type: ignore
    if is_classifier(forest):
        return lambda x: x.sum(1)
    else:
        return lambda x: x.sum(-1).sum(-1)


def get_xgboost_activation(forest):  # type: ignore
    if is_classifier(forest):
        n_class = forest.n_classes_
        if n_class == 2:
            return binary_logits

        elif n_class > 2:
            return lambda x: multi_softmax(x, n_class=n_class)

    else:
        if forest.base_score is not None:
            global_bias = forest.base_score
        else:
            global_bias = 0.5
        return lambda x: x.sum(-1).sum(-1) + global_bias


def get_lightgbm_activation(forest):  # type: ignore
    if is_classifier(forest):
        n_class = forest.n_classes_
        if n_class == 2:
            return binary_logits

        elif n_class > 2:
            return lambda x: multi_softmax(x, n_class=n_class)

    else:
        return lambda x: x.sum(-1).sum(-1)
