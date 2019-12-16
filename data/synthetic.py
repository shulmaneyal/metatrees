import numpy as np
import torch


def get_set(num_samples, x_dim, priors, threshold=0):
    x = np.random.randn(num_samples, x_dim)
    indices = np.random.choice(range(x_dim), 3, p=priors, replace=False)
    right = np.bitwise_and(x[:, indices[0]] >= threshold, x[:, indices[1]] >= threshold)
    left = np.bitwise_and(x[:, indices[0]] < threshold, x[:, indices[2]] >= threshold)
    y = np.bitwise_or(right, left)
    y = np.eye(2)[y.astype(int)]
    return x, y


def get_synthetic(batch_size, n_train, n_test, x_dim, prior_factor=1, with_train=False, cuda=True):
    num_samples = n_train + n_test
    priors = np.array([prior_factor ** i for i in range(x_dim)])
    priors = priors / sum(priors)

    x, y = [], []
    for _ in range(batch_size):
        x_set, y_set = get_set(num_samples, x_dim, priors)
        x.append(x_set)
        y.append(y_set)
    x = np.stack(x)
    y = np.stack(y)

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    if cuda:
        x = x.cuda()
        y = y.cuda()
    if with_train:
        return x[:, :n_train], y[:, :n_train], x, y
    return x[:, :n_train], y[:, :n_train], x[:, n_train:], y[:, n_train:]


def get_synthetic_varied(batch_size, min_n_train, max_n_train, n_test, x_dim, prior_factor=1, with_train=False,
                         cuda=True):
    n_train = np.random.randint(min_n_train, max_n_train + 1)
    return get_synthetic(batch_size, n_train, n_test, x_dim, prior_factor, with_train, cuda)
