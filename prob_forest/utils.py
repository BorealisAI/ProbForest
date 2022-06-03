# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
from sklearn.tree._tree import TREE_UNDEFINED


def check_tensor(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    y = torch.Tensor(x)
    if y.ndim == 1:
        return y.view(1, -1).to(dtype)
    else:
        return y.to(dtype)


def reindex(index, val, fill_value=TREE_UNDEFINED):  # type: ignore
    arr = fill_value * torch.ones((max(index) + 1,))
    arr[index] = torch.FloatTensor(val)
    return arr
