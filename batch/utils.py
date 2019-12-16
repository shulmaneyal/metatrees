import numpy as np
import torch

EPS = 1e-8


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def to_one_hot(values, n_values=2):
    return np.eye(n_values)[values]


def get_inner_nodes(tree):
    if tree.is_leaf():
        return []
    return [tree] + get_inner_nodes(tree.left) + get_inner_nodes(tree.right)


def max_depth(tree):
    if tree.is_leaf():
        return 0
    return 1 + max(max_depth(tree.right), max_depth(tree.left))


def to_x_y(data, group_parameter, target):
    return data.drop([group_parameter, target], axis=1), data[target]


def dist_to_sparse(w):
    shape = w.size()
    _, ind = w.abs().max(dim=-1)
    w_hard = torch.zeros_like(w).view(-1, shape[-1])
    w_hard.scatter_(1, ind.view(-1, 1), 1)
    return w_hard


def weights_to_sparse(w):
    shape = w.size()
    _, ind = w.abs().max(dim=-1)
    w_hard = torch.zeros_like(w).view(-1, shape[-1])
    w_hard.scatter_(1, ind.view(-1, 1), 1)
    w_hard = w_hard.view(*shape)
    return w_hard * w


def weighted_mean(x, w):
    return (w * x).sum(1) / (w.sum(1) + EPS)


def hard_softmax(logits, temperature=1):
    y = torch.softmax(logits / temperature, dim=-1)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard


DIST_HARD = hard_softmax
DIST_SOFT = lambda x: torch.softmax(x, dim=-1)
