# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
from torch.nn.utils.rnn import pad_sequence

from .utils import check_tensor

_BUFFER_TYPES = {
    "feature": torch.int64,
    "children_left": torch.int64,
    "children_right": torch.int64,
    "threshold": torch.float32,
    "value": torch.float32,
}


class TreeEnsemble(nn.Module):
    """
    Pytorch implementation of a decision tree ensemble.
    Allows GPU-compatible vectorized traversal.

    Parameters
    ----------
    n_features : int
        Dimension of the inputs

    depth : int
        (Maximum) depth of the trees

    activation: callable
        Function called on the leaf values to obtain output of the forest

    source : str
        Library used to train the ensemble

    Attributes
    ----------
    feature : torch.LongTensor
        Featids of the features used in the decision functions at the
        nodes.  Shape [n_tree, n_node]

    threshold : torch.FloatTensor
        Thresholds used in the decision functions at the nodes.
        Shape [n_tree, n_node]

    children_left : torch.LongTensor
        Nodeids of the left children of the decision nodes
        Shape [n_tree, n_node]

    children_right : torch.LongTensor
        Nodeids of the right children of the decision nodes
        Shape [n_tree, n_node]

    value : torch.LongTensor
        Values of each node.  The output of a tree is the value of the
        leaf node reached by traversal.  Shape [n_tree, n_node, n_output]

    parent : torch.LongTensor
        Nodeid for the parent of the node.  Parent of root is root.
        Shape [n_tree, n_node]

    node_sign : torch.LongTensor
        Indicator whetehr a node is the left-child (-1) or right child (+1)
        of its parent.  Root has node_sign 0.  Shape [n_tree, n_node]


    """

    def __init__(self, n_feature, depth, activation, source, **kwargs):  # type: ignore # noqa: E501
        """constructor"""

        super(TreeEnsemble, self).__init__()

        self.register_buffer("feature", None)
        self.register_buffer("children_left", None)
        self.register_buffer("children_right", None)

        self.register_buffer("threshold", None)
        self.register_buffer("value", None)

        self.depth = depth
        self.activation = activation

        self.source = source

        self.n_feature = n_feature

    def is_fit(self) -> None:
        for buffer_name in _BUFFER_TYPES:
            buffer = self.__getattr__(buffer_name)
            if buffer is None:
                raise ValueError("Must run set_param before using model")

    def set_param(self, param_dict):  # type: ignore
        """
        Set the parameters of the TreeEnsemble
        """

        for k, v in param_dict.items():
            self._buffers[k] = check_tensor(v, _BUFFER_TYPES[k])

        self.n_output = self.value.shape[-1]
        self._setup_backtrace()

    def _setup_backtrace(self) -> None:
        """
        Construct buffers necessary for running backtrace
        """
        self.n_tree, self.n_node = self.feature.shape  # type: ignore

        self.register_buffer(
            "parent",
            self.children_left.new_full((self.n_tree, self.n_node), 0),  # type: ignore # noqa: E501
        )
        self.register_buffer(
            "node_sign",
            self.children_left.new_full((self.n_tree, self.n_node), 0),  # type: ignore # noqa: E501
        )

        cl, cr = self.children_left.t(), self.children_right.t()  # type: ignore # noqa: E501
        for i, (l, r) in enumerate(zip(cl, cr)):
            self.parent[l > 0, l[l > 0]] = i  # type: ignore
            self.node_sign[l > 0, l[l > 0]] = -1  # type: ignore

            self.parent[r > 0, r[r > 0]] = i  # type: ignore
            self.node_sign[r > 0, r[r > 0]] = 1  # type: ignore

    def apply(self, x):  # type: ignore
        """
        Vectorized traversal of the tree

        Parameters
        ----------
        x : torch.FloatTensor
            Input data.  Shape [n_batch, n_feature]

        Returns:
        --------
        node : torch.LongTensor
            Nodeids indicating the leaves reached during traversal
            Shape [n_batch, n_tree]
        """
        n_batch, _ = x.shape
        xp = F.pad(x, (0, 1), mode="constant", value=0)

        node = self.children_left.new_full((n_batch, self.n_tree), 0)
        leaf_mask = node > -1
        for _ in range(self.depth):
            feat = torch.gather(self.feature, 1, node.t()).t()
            feat[feat < 0] = self.n_feature
            xval = torch.gather(xp, 1, feat)

            split = torch.gather(self.threshold, 1, node.t()).t()
            lchild = torch.gather(self.children_left, 1, node.t()).t()
            rchild = torch.gather(self.children_right, 1, node.t()).t()

            check = xval <= split
            new_node = torch.where(check, lchild, rchild)

            leaf_mask = new_node > TREE_LEAF
            node = torch.where(leaf_mask, new_node, node)

            if not leaf_mask.any():
                break

        return node

    def _gather_value(self, node: torch.LongTensor) -> torch.Tensor:
        """
        In allowing the leaves to have multi-dimensional outputs,
        the gather operation becomes a bit more nuanced.  We contain
        that logic inside this function.
        """
        node = node[:, :, None].expand(-1, -1, self.n_output)  # type: ignore
        val = torch.gather(self.value, 1, node.permute(1, 0, 2))  # type: ignore # noqa: E501
        val = val.permute(1, 0, 2)

        return val

    def predict(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute the output of the forest, given inputs

        Parameters:
        -----------
        x : torch.FloatTensor
            Input data.  Shape [n_batch, n_feature]

        Returns:
        --------
        out : torch.FloatTensor
            Output of the forest.  Shape [n_batch, n_output]
        """
        node = self.apply(x)
        val = self._gather_value(node)
        out = self.activation(val)

        return out

    def _backtrace(self, node):  # type: ignore
        """
        Vectorized traversal through the decision tree.  In contrast to
        predict, backtrace starts its traversal at a specified node, and moves
        upward until it reaches the root.

        The backtrace can be used for efficient construction of the full
        decision path.

        Parameters
        ----------
        node : torch.LongTensor
            Tensor containing nodeids to initiate the reverse traversal.  Has
            shape [n_batch, n_tree]

        Returns
        -------
        visited : torch.LongTensor
            Nodeids encountered during the reverse traversal (in order of
            decreasing depth).  Shape [n_batch, n_tree, depth + 1]

        feat : torch.LongTensor
            Featids of the features used in the decision functions at the
            nodes.  Shape [n_batch, n_tree, depth + 1]

        split : torch.FloatTensor
            Thresholds used in the decision functions at the nodes.
            Shape [n_batch, n_tree, depth + 1]

        node_sign : torch.LongTensor
            Tensor indicating whether the node is the left-child (-1) or right
            child (+1) of its parent.  A value of 0 is used to indicate root.
            Shape [n_batch, n_tree, depth + 1]
        """
        n_batch, _ = node.shape
        root_mask = node > 0
        mask_val = torch.zeros_like(node).float()

        def _lookup(source, inds):  # type: ignore
            _data = torch.gather(source.float(), 1, inds.t()).t()
            return torch.where(root_mask, _data, mask_val)

        visited = node.new_full(
            (n_batch, self.n_tree, self.depth + 1), TREE_UNDEFINED
        )
        visited[:, :, 0] = node

        feat = self.threshold.new_full(
            (n_batch, self.n_tree, self.depth + 1), TREE_UNDEFINED
        )

        split = self.threshold.new_full(
            (n_batch, self.n_tree, self.depth + 1), TREE_UNDEFINED
        )

        node_sign = self.threshold.new_full(
            (n_batch, self.n_tree, self.depth + 1), 0
        )
        node_sign[:, :, 0] = _lookup(self.node_sign, node)

        # Obtain the parent nodes
        pnode = torch.gather(self.parent, 1, node.t()).t()
        pnode[pnode < 0] = 0
        for i in range(self.depth):
            visited[:, :, i + 1] = torch.where(
                root_mask, pnode, mask_val.long()
            )

            feat[:, :, i] = _lookup(self.feature, pnode)
            split[:, :, i] = _lookup(self.threshold, pnode)
            node_sign[:, :, i + 1] = _lookup(self.node_sign, pnode)

            pnode = torch.gather(self.parent, 1, pnode.t()).t()
            root_mask = pnode > -1

            # Certain versions of torch.gather do not accept negative indices
            # It doesn't matter though, because of the masked assignment
            pnode[pnode < 0] = 0

            if not (root_mask).any():
                break

        feat, node_sign = feat.long(), node_sign.long()

        return visited, feat, split, node_sign


class SamplePredictAutograd(torch.autograd.Function):
    """
    Smooth relaxation of decision forests that use sampling methods
    to compute leaf outputs.

    The forward method simply returns the value of the sampled leaf
    The backward pass uses REINFORCE to compute the expected gradient
    """

    @staticmethod
    def forward(ctx, input, val, log_prob):  # type: ignore
        """
        Val will be returned, the rest will be stored for the
        backward pass
        """
        if input.requires_grad:
            context_manager = torch.enable_grad
            val0 = 0.0
        else:
            context_manager = torch.no_grad
            val0 = 0.0

        val0 = torch.tensor(val0)

        with context_manager():
            log_prob = torch.stack(log_prob)
            sumlogprob = log_prob.sum(0)

            weighted_sumlogprob = (val - val0) * sumlogprob[:, :, None]

        ctx.save_for_backward(input, weighted_sumlogprob)
        return val

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        input, wslp = ctx.saved_tensors
        (grad_on_input,) = torch.autograd.grad(
            wslp, input, grad_outputs=grad_output
        )

        return grad_on_input, None, None


sample_predict = SamplePredictAutograd.apply


class SampledPathEnsemble(TreeEnsemble):
    """
    Smooth relaxation of a Decision Forest.
    Leaves are stochastically sampled according to a sigmoidal
    splitting condition

    See TreeEnsemble for a more complete description of Parameters and
    Attributes

    Parameters
    ----------
    scale: torch.FloatTensor
        Scale used to normalize split differences.  (Defaults to tensor of 1s)
        Shape [n_feature]

    temperature: float
        Default hyperparameter controlling the smoothness of the relaxation

    Attributes
    ----------
    scale: torch.FloatTensor
        Similar to scale, but padded to allow gather ops
        Shape [n_feature + 1]

    """

    def __init__(  # type: ignore
        self,
        n_feature,
        depth,
        activation,
        scale=None,
        temperature=0.1,
        **kwargs
    ):
        super().__init__(n_feature, depth, activation, **kwargs)

        if scale is None:
            self.register_buffer(
                "scale", torch.ones(self.n_feature + 1).float()
            )
        else:
            self.register_buffer(
                "scale", F.pad(scale, (0, 1), mode="constant", value=1)
            )

        self.buff = 1e-5
        self.temperature = temperature

    def sample(self, x, T=None, return_trace: bool = False):  # type: ignore
        """
        This is a stochastic version of 'apply'

        In apply, one goes left if xval <= split

        In the stochastic version, the value of (xval - split) is used
        to compute a probability of going left, and we sample from this
        distribution.

        Parameters
        ----------
        X : torch.FloatTensor
            Input data.  Shape [n_batch, n_feature]

        T : float
            Temperature value (overrides default temperature)

        return_trace : bool
            Return the full forward trace, in addition to the output

        Returns
        -------
        val : torch.FloatTensor
            The values of the leaves reached.
            Shape [n_batch, n_tree, n_output]

        node : torch LongTensor
            The nodeids of the leaves reached.
            Shape [n_batch, n_tree]

        logprob list[torch.FloatTensor]
            List of length depth containing the probabilities to visit the
            left/right children. sum(logprob) gives the probability of the
            total path.  Elements have shape [n_batch, n_tree]

        trace : dict[List[torch.Tensor]]
            Dictionary containing the full forward trace (optionally returned).
            The trace for each key is a list of tensors, one for each attribute
            of the nodes encountered along the traversal.  All tensors have
            shape [n_batch, n_tree]
            Keys [feat, split, node_sign]

        """
        if T is None:
            T = self.temperature
        n_batch, _ = x.shape
        # We pad with 1's to avoid confusing the computation graph
        # with unecessary gather operations
        xp = F.pad(x, (0, 1), mode="constant", value=0)

        node = self.children_left.new_full((n_batch, self.n_tree), 0)  # type: ignore # noqa: E501
        log_prob = []

        mask_val = torch.ones_like(node)

        if return_trace:
            trace = {}  # type: ignore
            trace["feat"] = []
            trace["split"] = []
            trace["node_sign"] = []

        leaf_mask = node > -1

        for d in range(self.depth):
            feat = torch.gather(self.feature, 1, node.t()).t()  # type: ignore

            # If we have reached a leaf node, feat < 0, which gather does not
            # support.  Instead, we gather from the last index, which we've
            # padded with 0's to avoid confusing the computation graph
            feat[feat < 0] = self.n_feature
            xval = torch.gather(xp, 1, feat)

            # is there some way to do this without copying?
            split = torch.gather(self.threshold, 1, node.t()).t()  # type: ignore # noqa: E501

            lchild = torch.gather(self.children_left, 1, node.t()).t()  # type: ignore # noqa: E501
            rchild = torch.gather(self.children_right, 1, node.t()).t()  # type: ignore # noqa: E501

            diffs = (xval - split) / self.scale[feat] / T  # type: ignore
            prob_right = torch.sigmoid(diffs)
            prob_left = torch.sigmoid(-diffs)

            with torch.no_grad():
                check = prob_right <= torch.rand_like(prob_right)

            new_node = torch.where(check, lchild, rchild)
            new_probs = torch.where(check, prob_left, prob_right)
            # Could also use this, but torch.where seems more elegant:
            # new_probs = \
            # check.float() * prob_left + (1 - check.float()) * prob_right

            leaf_mask = new_node > TREE_LEAF
            node = torch.where(leaf_mask, new_node, node)
            probs = torch.where(leaf_mask, new_probs, mask_val.float())

            log_prob.append(torch.log(probs))

            if return_trace:
                trace["feat"].append(feat)
                trace["split"].append(split)

                node_sign = torch.where(check, -mask_val, mask_val)
                node_sign = torch.where(leaf_mask, node_sign, 0 * mask_val)

                trace["node_sign"].append(node_sign)

            if not leaf_mask.any():
                break

        val = self._gather_value(node)  # type: ignore

        if return_trace:
            return val, node, log_prob, trace

        else:
            return val, node, log_prob

    def forward(self, x, T=None):  # type: ignore
        """
        Stochastically sample a path through the tree based on x.
        Then pass collected values through the activation function
        to determine the output of the ensemble.

        Parameters
        ----------
        x : torch.FloatTensor
            Input data.  Shape [n_batch, n_feature]

        T : float
            Temperature value (overrides default temperature)

        Returns
        -------
        out : torch.FloatTensor
            Output of the ensemble.  Shape depends on activation.
        """

        if T is None:
            T = self.temperature

        val, _, log_prob = self.sample(x, T=T)
        out = self.activation(sample_predict(x, val, log_prob))

        return out


class ListEnsemble(nn.Module):
    def __init__(self, model_list, activation):  # type: ignore
        super(ListEnsemble, self).__init__()
        self.model_list = nn.ModuleList(model_list)
        self.activation = activation

    def apply(self, X):  # type: ignore
        output = torch.stack([m.apply(X) for m in self.model_list])
        # output_shape [n_tree, n_batch, 1]
        output = output.permute(1, 0, 2).squeeze(2)
        return output

    def forward(self, X):  # type: ignore
        output = torch.stack([m(X) for m in self.model_list])
        # output_shape [n_tree, n_batch, 1]
        output = output.permute(1, 0, 2).squeeze(2)
        return self.activation(output)


class ExhaustiveEnsemble(TreeEnsemble):
    """
    Smooth relaxation of a Decision Forest.
    A probability is computed to reach each leaf of every tree in the forest,
    and this is used to compute a weighted average of the leaf outputs.

    The probability to reach a leaf is given by the product of a sigmoidal
    splitting probability at each node encountered on the path to reach the
    leaf.

    See TreeEnsemble for a more complete description of Parameters and
    Attributes

    Parameters
    ----------
    scale: torch.FloatTensor
        Scale used to normalize split differences.  (Defaults to tensor of 1s)
        Shape [n_feature]

    temperature: float
        Default hyperparameter controlling the smoothness of the relaxation

    Attributes
    ----------
    scale: torch.FloatTensor
        Similar to scale, but padded to allow gather ops
        Shape [n_feature + 1]

    """

    def __init__(  # type: ignore
        self,
        n_feature,
        depth,
        activation,
        temperature=0.1,
        scale=None,
        **kwargs
    ):
        super().__init__(n_feature, depth, activation, **kwargs)

        if scale is None:
            self.register_buffer(
                "scale", torch.ones(self.n_feature + 1).float()
            )
        else:
            self.register_buffer(
                "scale", F.pad(scale, (0, 1), mode="constant", value=1)
            )

        self.buff = 1e-5
        self.temperature = temperature

    def set_param(self, param_dict):  # type: ignore
        super().set_param(param_dict)
        self._setup_exhaustive()

    def _setup_exhaustive(self) -> None:
        """
        For the ExhaustiveEnsemble, there are a few preprocessing steps that
        must be performed upon the buffers and parameters of the TreeEnsemble.
        This is done to support our approach to efficiently performing an
        exhaustive traversal of the ensemble, which proceeds as follow:

        (1) Flatten all the parameters in the ensemble (splits along features)
            * Concatenate the non-leaf nodes together into a single long vector
            * This vector has one element for every non-leaf node in the
            forest, which is ascribed a nodeid
            * In the ensemble, there are n_total_leaf leaves, and n_total_node
            nodes

        (2) For each leaf in the ensemble, compute the path taken by that leaf
            * This amounts to two pieces of information:
                (1) leaf_dec_path: the nodes encountered by a given leaf
                    * [n_total_leaf, max_depth] integer matrix
                    * Gives the (ordered) node ids encountered during traversal
                    * Different path lengths -> pad with root
                (2) leaf_sign: whether you went left or right at that node
                    * [n_total_leaf, max_depth] integer matrix
                    * Different path lengths -> pad with 0
                    * Values of (-1,0,1) -> (left, padding, right)

        (3) Given an input X, evaluate distance to all splits

        (4) Use the leaf_dec_path to gather these distances, and then compute
        the probability of reaching the leaf
        """
        # Get a count of how many nodes are in each tree
        # This will be used to index the nodes in one large array
        self.node_count = (self.children_left != TREE_UNDEFINED).sum(1)
        self.n_total_node = self.node_count.sum()

        # Flatten the buffers
        self._flatten_buffers()

        # Get counts of the leaves
        self.leaf_count = (self.children_left == TREE_LEAF).sum(1)
        self.max_leaf = self.leaf_count.max().item()
        self.n_total_leaf = self.leaf_count.sum()

        # Pre-compute the paths
        self._precompute_paths()

        # Construct the reverse index
        self._construct_reverse_index()

    def _flatten_buffers(self) -> None:
        # Instead of this flattening operation
        # we could just call torch.cat instead of torch.stack

        real_nodes = self.children_left != TREE_UNDEFINED
        _flatten = lambda x: x.reshape(-1)[real_nodes.reshape(-1)]  # noqa:E731
        for name in ["feature", "threshold"]:
            buff = self._buffers[name]
            self.register_buffer("flat_{}".format(name), _flatten(buff))

        flat_value = self.value.reshape(-1, self.n_output)[  # type: ignore
            real_nodes.reshape(-1)
        ]

        self.register_buffer(
            "leaf_mask", _flatten(self.children_left == TREE_LEAF)
        )
        self.register_buffer("flat_value", flat_value[self.leaf_mask])  # type: ignore # noqa: E501

        self.flat_feature[self.leaf_mask] = self.n_feature  # type: ignore

    def _get_leaves(self, fill_value=TREE_UNDEFINED):  # type: ignore
        """
        Obtain the leaf nodes from the tree, as marked by TREE_LEAF
        in self.children_left (per the sklearn datastructure)
        """

        leaf_list = [
            torch.where(lchild == TREE_LEAF)[0][:, None]
            for lchild in self.children_left
        ]

        return pad_sequence(leaf_list, padding_value=fill_value).squeeze(2).t()

    def _construct_reverse_index(self) -> None:
        """
        To assist with lookup in the precomputed leaf path, define a reverse
        index that, given the nodeid of the leaf, will return the index of
        that leaf in the leaf collection.

        This will be used to compute the decision_path, and to aggregate
        leaf-predictions into tree-predictions
        """

        # These start indices describe the offsets after all the leaves have
        # been concatenated into the precomputed leaf path collection
        # Note: this is defined with respect to the leafid, not the nodeid
        leaf_start_index = torch.zeros(self.n_tree).long()
        leaf_start_index[1:] = torch.cumsum(self.leaf_count, 0)[:-1]

        leaf_mask = (self.children_left == TREE_LEAF).long()
        _reverse_index = torch.cumsum(leaf_mask, 1)
        _reverse_index = _reverse_index * leaf_mask - 1

        _reverse_index += leaf_start_index[:, None] * (_reverse_index != -1)
        _reverse_index[_reverse_index == -1] = self.n_total_leaf

        self.register_buffer("reverse_index", _reverse_index)

    def _precompute_paths(self) -> None:
        """
        Run _backtrace to perform a reverse traversal on each leaf in the
        forest. Namely, we recursively determine the parent of a given node,
        starting from a leaf node.

        This allows us to precompute the path to each leaf.
        """
        # Get all leaves from the forest
        # We assume there will never be a leaf with id 0, as that would
        # correspond to a tree that was only a root node
        # If that is encountered, simply change the fill_value
        # We pad the leaves with 0 (root), because the parent of root is root
        leaves = self._get_leaves(fill_value=0)
        fake_leaves = leaves == 0

        # Use _backtrace to pre-compute the path
        # Take the transpose of leaves so that we can pass it through
        # _backtrace
        leaf_dec_path, _, _, leaf_sign = self._backtrace(leaves.T)

        # Now undo the transpose operation
        # (leave the last dimension alone, it contains the path taken)
        leaf_dec_path = leaf_dec_path.permute(1, 0, 2)

        # At present, the nodeids referenced in leaf_dec_path are the node-ids
        # for the tree, rather than the node-ids which appear in the flattened
        # parameter collection
        # We can fix this by adding an offset to those 'local' node-ids

        # Compute the offsets for each leaf
        node_start_index = self.children_left.new_full((self.n_tree,), 0)  # type: ignore # noqa: E501
        node_start_index[1:] = torch.cumsum(self.node_count, 0)[:-1]
        node_start_index = (~fake_leaves).long() * node_start_index[:, None]

        # Perform the re-indexation, and remove padding
        # flatten for easier indexing
        leaf_dec_path = leaf_dec_path.reshape(-1, leaf_dec_path.shape[-1])
        # re-index to match up with dec_path indices
        leaf_dec_path += node_start_index.reshape(-1)[:, None]
        # remove padding
        leaf_dec_path = leaf_dec_path[~fake_leaves.reshape(-1)]

        # Similar steps for leaf_sign
        leaf_sign = leaf_sign.permute(1, 0, 2)
        leaf_sign = leaf_sign.reshape(-1, leaf_sign.shape[-1])
        leaf_sign = leaf_sign[~fake_leaves.reshape(-1)]

        # One additional step for leaf_sign:
        # At present, leaf_sign is written in the reverse traversal order
        # for the decision path, we want this traversal in forward order
        # this can easily be accomodated by shifting the elements of leaf_sign
        # up by 1 along the path dimension
        leaf_sign = torch.roll(leaf_sign, 1, 1)

        self.register_buffer("leaf_dec_path", leaf_dec_path)
        self.register_buffer("leaf_sign", leaf_sign)

    def _leaf_idx_to_path(self, leaf_idx):  # type: ignore
        """"""
        n_batch, _ = leaf_idx.shape
        dec_path = self.leaf_dec_path[leaf_idx.reshape(-1)].reshape(
            n_batch, -1
        )

        xi = torch.stack(
            dec_path.shape[1] * [torch.arange(n_batch)]
        ).T.flatten()
        xi = xi.to(leaf_idx.device)
        yi = dec_path.flatten()

        inds = torch.stack([xi, yi])
        vals = inds.new_full((inds.shape[1],), 1)

        return torch.sparse.FloatTensor(
            inds, vals, torch.Size([n_batch, self.n_total_node])
        )

    def decision_path(self, X):  # type: ignore
        leaf_node = self.apply(X)
        leaf_idx = torch.gather(self.reverse_index, dim=1, index=leaf_node.T).T
        return self._leaf_idx_to_path(leaf_idx)

    def path_prob(self, x, T=None, buff=1e-5):  # type: ignore
        """
        Compute the probability of reaching a given leaf in the ensemble.
        This amounts to evaluating the probability of the path taken to reach
        that leaf.

        The leaf probabilities are concatenated together to form a single
        tensor. For information about ordering, please review _setup_exhaustive

        Parameters
        ----------
        x : torch.FloatTensor
            Input data.  Shape [n_batch, n_feature]

        T : float
            Temperature value (overrides default temperature)

        Returns
        -------
        path_prob : torch.FloatTensor
            Probability of reaching each leaf.
            Shape [n_batch, n_total_leaf]
        """
        if T is None:
            T = self.temperature
        n_batch, _ = x.shape

        # We pad with 1's to avoid confusing the computation graph
        # with unecessary gather operations
        # Xp = pad_with_zeros(X)
        xp = F.pad(x, (0, 1), mode="constant", value=0)

        # If we have reached a leaf node, feat < 0, which gather does not
        # support.  Instead, we gather from the last index, which we've padded
        # with 0's to avoid confusing the computation graph
        # feat[feat < 0] = self.n_feature
        xval = torch.index_select(xp, dim=1, index=self.flat_feature)

        # xval contains a great number of zeros, indicating a lot of wasted
        # space.  However, if we use a different indexing strategy, we cannot
        # use the _backtrace trick (well to be fair we could, we would just
        # need a 1D reindexing tensor)
        split_scale = self.scale[self.flat_feature]
        diff = (xval - self.flat_threshold) / split_scale

        # Given the batch computed diffs, align them with the precomputed
        # leaves
        diff = torch.index_select(
            diff, dim=1, index=self.leaf_dec_path.reshape(-1)
        ).reshape(n_batch, self.n_total_leaf, -1)
        diff *= self.leaf_sign[None, :, :]

        split_log_prob = (
            F.logsigmoid(diff / T) * (self.leaf_sign != 0)[None, :, :]
        )
        path_prob = torch.exp(split_log_prob.sum(-1))

        return path_prob

    def _collect_leaves(self, leaf_output):  # type: ignore
        """
        Aggregate the outputs of the leaves to obtain the outputs
        of the trees

        Parameters
        ----------
        leaf_output : torch.FloatTensor
            Probability of reaching a leaf. Shape [n_batch, n_total_leaf]

        Returns
        -------
        tree_output : torch.FloatTensor
            The output of each tree. Shape [n_batch, n_tree, n_output]
        """
        padded_leaf_output = F.pad(
            leaf_output, (0, 0, 0, 1), mode="constant", value=0
        )

        tree_output = padded_leaf_output[:, self.reverse_index.flatten(), :]
        tree_output = tree_output.reshape(
            -1, self.n_tree, self.n_node, self.n_output
        )

        return tree_output.sum(2)

    def forward(self, x, T=None):  # type: ignore
        """
        Compute the probability of reaching a given leaf given the input x.
        Use this to compute the expected value of the ouptut.

        Parameters
        ----------
        x : torch.FloatTensor
            Input data.  Shape [n_batch, n_feature]

        T : float
            Temperature value (overrides default temperature)

        Returns
        -------
        out : torch.FloatTensor
            Output of the ensemble.  Shape depends on activation.
        """
        if T is None:
            T = self.temperature

        path_prob = self.path_prob(x, T=T)
        tree_prob = self._collect_leaves(
            path_prob[:, :, None] * self.flat_value[None, :, :]
        )
        return self.activation(tree_prob)
