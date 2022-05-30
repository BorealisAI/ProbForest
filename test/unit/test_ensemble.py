# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import itertools

import numpy as np
import pytest
import torch
from lightgbm import LGBMClassifier, LGBMModel, LGBMRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from xgboost.sklearn import XGBModel


def true_grad_log_sigmoid(x):  # type: ignore
    return 1 / (1 + torch.exp(x))


# TODO:
# Add test for the xgboost parsing

_MODEL_TYPES = ["rf", "xgb", "lgb"]
if torch.cuda.is_available():
    _DEVICES = ["cpu", "cuda"]
else:
    _DEVICES = ["cpu"]
_TASK = ["binary", "multi", "regression"]
_COMBINATIONS = list(itertools.product(_MODEL_TYPES, _DEVICES, _TASK))


@pytest.fixture(params=_COMBINATIONS)
def forest_and_dataset(request):  # type: ignore
    model_type, device, task = request.param

    if model_type == "lgb":
        n_samples = 50
    else:
        n_samples = 30

    if task == "binary":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=8,
            n_classes=2,
            random_state=0,
        )
        yt = torch.LongTensor(y)
    elif task == "multi":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=8,
            n_classes=4,
            random_state=0,
        )
        yt = torch.LongTensor(y)
    elif task == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            n_features=10,
            n_informative=8,
            noise=10,
            random_state=0,
        )
        yt = torch.FloatTensor(y)

    Xt = torch.FloatTensor(X)

    if task in {"binary", "multi"}:
        if model_type == "rf":
            clf = RandomForestClassifier(
                max_depth=3, random_state=0, n_estimators=7
            )
        elif model_type == "xgb":
            clf = XGBClassifier(
                max_depth=3,
                random_state=0,
                min_samples_split=2,
                n_estimators=7,
            )
        elif model_type == "lgb":
            clf = LGBMClassifier(max_depth=3, n_estimators=7, random_state=0)

        pred_fn = clf.predict_proba

    elif task == "regression":
        if model_type == "rf":
            clf = RandomForestRegressor(
                max_depth=3, random_state=0, n_estimators=7
            )
        elif model_type == "xgb":
            clf = XGBRegressor(
                max_depth=3,
                random_state=0,
                min_samples_split=2,
                n_estimators=7,
            )
        elif model_type == "lgb":
            clf = LGBMRegressor(max_depth=3, n_estimators=7, random_state=0)
        pred_fn = clf.predict

    clf.fit(X, y)

    Xt = Xt.to(device)
    yt = yt.to(device)

    return clf, Xt, yt, device, pred_fn, task


def test_ensemble_ops(forest_and_dataset):  # type: ignore
    clf, Xt, yt, device, pred_fn, task = forest_and_dataset

    if isinstance(clf, LGBMModel):
        # lightgbm does not have an apply method
        # so we cannot test it
        return 0

    Xt = Xt[:1]

    from prob_forest.parse import parse_model

    torch_ensemble = parse_model(clf, relaxation=None).to(device)
    torch_list_ensemble = parse_model(
        clf, relaxation=None, use_padding=False
    ).to(device)

    # Testing ensemble apply

    assert np.array_equal(
        clf.apply(Xt.cpu().numpy()), torch_ensemble.apply(Xt).cpu().numpy()
    )
    assert np.array_equal(
        clf.apply(Xt.cpu().numpy()),
        torch_list_ensemble.apply(Xt).cpu().numpy(),
    )

    lst_nodes = []
    lst_vis, lst_ft, lst_st, lst_lr = [], [], [], []
    for i, t in enumerate(torch_list_ensemble.model_list):
        _node = t.apply(Xt)
        _vis, ft, st, lr = t._backtrace(_node)
        lst_nodes.append(_node.squeeze(1))
        lst_vis.append(_vis.squeeze(1))
        lst_ft.append(ft.squeeze(1))
        lst_st.append(st.squeeze(1))
        lst_lr.append(lr.squeeze(1))

    lst_nodes = torch.stack(lst_nodes).permute(1, 0).cpu().numpy()
    lst_vis = torch.stack(lst_vis).permute(1, 0, 2).cpu().numpy()
    lst_ft = torch.stack(lst_ft).permute(1, 0, 2).cpu().numpy()
    lst_st = torch.stack(lst_st).permute(1, 0, 2).cpu().numpy()
    lst_lr = torch.stack(lst_lr).permute(1, 0, 2).cpu().numpy()

    nodes = torch_ensemble.apply(Xt)
    visited, feat, split, node_sign = torch_ensemble._backtrace(nodes)

    visited = visited.cpu().numpy()
    feat = feat.cpu().numpy()
    split = split.cpu().numpy()
    node_sign = node_sign.cpu().numpy()

    # Testing ensemble backtrace...

    assert np.array_equal(lst_nodes, visited[:, :, 0])
    assert np.array_equal(lst_vis, visited)
    assert np.array_equal(lst_ft, feat)
    assert np.array_equal(lst_st, split)
    assert np.array_equal(lst_lr, node_sign)

    # Testing predict
    clf_pred = pred_fn(Xt.cpu().numpy())
    torch_pred = torch_ensemble.predict(Xt).cpu().numpy()  # [:,1]

    # Because of numerical imprecision in parsing xgboost from
    # a json file, we cannot enforce exact equality
    assert np.allclose(clf_pred, torch_pred, atol=1e-5)


def test_log_sigmoid_grad(forest_and_dataset):  # type: ignore
    _, Xt, _, _, _, _ = forest_and_dataset

    from torch.autograd import Variable

    Xs = Variable(Xt[:1], requires_grad=True)

    log_probs = torch.log(torch.sigmoid(Xs))
    sumlogprobs = log_probs.sum(-1)

    (grad_on_xvals,) = torch.autograd.grad(sumlogprobs, Xs)

    true_grad = true_grad_log_sigmoid(Xs)

    assert np.allclose(
        grad_on_xvals.cpu().numpy(), true_grad.data.cpu().numpy()
    )


def test_ensemble_grad_across_depth(forest_and_dataset):  # type: ignore
    clf, Xt, yt, device, pred_fn, task = forest_and_dataset

    from prob_forest.parse import parse_model

    approximate_torch_ensemble = parse_model(clf, relaxation="sampling").to(
        device
    )

    torch.manual_seed(0)

    T = 0.2
    from torch.autograd import Variable

    Xs = Variable(Xt[:11], requires_grad=True)

    _, _, log_probs, trace = approximate_torch_ensemble.sample(
        Xs, T, return_trace=True
    )

    x_grads = []

    with torch.no_grad():
        Zp = Xs.data.new_full((Xs.shape[0],), 0).view((-1, 1))
        Xp = torch.cat((Xs, Zp), 1)

    x_grads = []

    for d in range(len(log_probs)):
        x_grad = torch.zeros_like(Xp)

        node_sign = trace["node_sign"][d]
        feat = trace["feat"][d]

        xvals = torch.gather(Xp, 1, feat)
        split = trace["split"][d]

        deltas = (xvals - split) / T

        grads = node_sign * true_grad_log_sigmoid(node_sign * deltas) / T
        x_grad.scatter_add_(dim=1, index=feat, src=grads)

        sumlogprobs = log_probs[d].sum()
        (grad_on_prob,) = torch.autograd.grad(
            sumlogprobs, Xs, retain_graph=True
        )

        a, b = x_grad[:, :-1].data.cpu().numpy(), grad_on_prob.cpu().numpy()
        assert np.allclose(a, b, atol=1e-5)

        x_grads.append(x_grad[:, :-1])

    sumlogprobs = sum(log_probs).sum()
    x_grads = sum(x_grads)
    (grad_on_prob,) = torch.autograd.grad(sumlogprobs, Xs, retain_graph=True)

    a, b = x_grads.data.cpu().numpy(), grad_on_prob.cpu().numpy()
    assert np.allclose(a, b, atol=1e-5)


def test_ensemble_grad_across_trees(forest_and_dataset):  # type: ignore
    clf, Xt, yt, device, pred_fn, task = forest_and_dataset

    from prob_forest.parse import parse_model

    approximate_torch_ensemble = parse_model(clf, relaxation="sampling").to(
        device
    )

    torch.manual_seed(0)

    T = 0.2
    from torch.autograd import Variable

    Xs = Variable(Xt[:11], requires_grad=True)

    _, _, log_probs, trace = approximate_torch_ensemble.sample(
        Xs, T, return_trace=True
    )

    x_grads = []
    with torch.no_grad():
        Zp = Xs.data.new_full((Xs.shape[0],), 0).view((-1, 1))
        Xp = torch.cat((Xs, Zp), 1)

    x_grads = []

    lp = torch.stack(log_probs).sum(0)

    lr = torch.stack(trace["node_sign"]).permute(1, 2, 0)
    st = torch.stack(trace["split"]).permute(1, 2, 0)
    ft = torch.stack(trace["feat"]).permute(1, 2, 0)

    for t in range(clf.n_estimators):
        x_grad = torch.zeros_like(Xp)
        node_sign = lr[:, t, :]
        feat = ft[:, t, :]

        xvals = torch.gather(Xp, 1, feat)
        split = st[:, t, :]

        deltas = (xvals - split) / T

        grads = node_sign * true_grad_log_sigmoid(node_sign * deltas) / T
        x_grad.scatter_add_(dim=1, index=feat, src=grads)

        arr = lp.data.new_full((lp.shape), 0)
        arr[:, t] = 1
        (grad_on_prob,) = torch.autograd.grad(
            lp, Xs, grad_outputs=arr, retain_graph=True
        )

        a, b = x_grad[:, :-1].data.cpu().numpy(), grad_on_prob.cpu().numpy()
        assert np.allclose(a, b, atol=1e-5)

        x_grads.append(x_grad[:, :-1])


def test_predict(forest_and_dataset):  # type: ignore
    clf, Xt, yt, device, pred_fn, task = forest_and_dataset

    from prob_forest.parse import parse_model

    approximate_torch_ensemble = parse_model(clf, relaxation=None).to(device)

    clf_pred = pred_fn(Xt.cpu().numpy())
    torch_pred = approximate_torch_ensemble.predict(Xt).cpu().numpy()  # [:,1]

    assert np.allclose(clf_pred, torch_pred, atol=1e-5)


def test_sampling_forward(forest_and_dataset):  # type: ignore
    """
    Check that in the low temperature limit the predictions align

    This needn't be too strict...it is statistical after all
    """
    clf, Xt, yt, device, pred_fn, task = forest_and_dataset
    temperature = 0.0001

    from prob_forest.parse import parse_model

    approximate_torch_ensemble = parse_model(
        clf, relaxation="sampling", temperature=temperature
    ).to(device)

    clf_pred = pred_fn(Xt.cpu().numpy())
    torch_pred = approximate_torch_ensemble(Xt).cpu().numpy()  # [:,1]

    assert np.allclose(clf_pred, torch_pred, atol=1e-5)


def test_exhaustive_forward(forest_and_dataset):  # type: ignore
    """
    Check that in the low temperature limit the predictions align

    This needn't be too strict.
    """
    clf, Xt, yt, device, pred_fn, task = forest_and_dataset
    temperature = 0.0001

    from prob_forest.parse import parse_model

    approximate_torch_ensemble = parse_model(
        clf, relaxation="exhaustive", temperature=temperature
    ).to(device)

    clf_pred = pred_fn(Xt.cpu().numpy())
    torch_pred = approximate_torch_ensemble(Xt).cpu().numpy()  # [:,1]

    assert np.allclose(clf_pred, torch_pred, atol=1e-5)


def test_decision_path(forest_and_dataset):  # type: ignore
    clf, Xt, yt, device, pred_fn, task = forest_and_dataset

    from prob_forest.parse import parse_model

    if isinstance(clf, XGBModel) or isinstance(clf, LGBMModel):
        # XGBOoost does not expose the internals of its decision path
        # so we cannot test it
        return 0

    exhaustive_ensemble = parse_model(
        clf, relaxation="exhaustive", use_padding=True
    ).to(device)

    Xs = Xt[1:2]

    dec_path = (
        exhaustive_ensemble.decision_path(Xs)
        .to_dense()
        .cpu()
        .numpy()
        .squeeze()
    )
    dec_path[dec_path > 1] = 1

    sklearn_dec_path, _ = clf.decision_path(Xs.cpu().numpy())
    sklearn_dec_path = sklearn_dec_path.toarray().squeeze()

    assert np.array_equal(sklearn_dec_path, dec_path)
