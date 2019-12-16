from batch.utils import get_inner_nodes


def _sparse_norm(v):
    interactions = v.unsqueeze(-1).matmul(v.unsqueeze(1)).abs().sum(1).sum(1)
    diag = (v ** 2).sum(1)
    return (interactions - diag).mean()


def sparse_norm(model, x, y):
    tree = model.get_tree(x, y)
    nodes = get_inner_nodes(tree)
    if len(nodes) == 0:
        return 0
    norms = [_sparse_norm(node.rweights) for node in nodes]
    norm = sum(norms) / len(norms)
    return norm
