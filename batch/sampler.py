import numpy as np
import torch

from batch.utils import to_one_hot


class Sampler:
    def __init__(self, batch_size, data, prepocessor, n_test, group_parameter, target, cuda=True, with_train=False,
                 regression=False):
        self.batch_size = batch_size
        self.data = data
        self.preprocessor = prepocessor
        self.n_test = n_test
        self.group_parameter = group_parameter
        self.target = target

        self.group_counts = data[group_parameter].value_counts()
        self.min_samples = self.group_counts.min()
        self.max_samples = self.group_counts.iloc[batch_size - 1]
        self.cuda = cuda
        self.with_train = with_train
        self.regression = regression

    def _to_x_y(self, data):
        x = data.drop([self.group_parameter, self.target], axis=1)
        grouped = x.groupby(data[self.group_parameter]).apply(lambda a: a.values)
        x = np.array(grouped.tolist())
        shape = x.shape
        x = self.preprocessor.transform(x.reshape(-1, shape[-1])).reshape(shape)

        if self.regression:
            y = data[self.target].astype('float')
        else:
            y = data[self.target].astype('int')
        grouped = y.groupby(data[self.group_parameter]).apply(lambda a: a.values)
        y = np.array(grouped.tolist())
        return x, y

    def sample(self):
        n_samples = int(self.min_samples + np.random.rand() * (self.max_samples - self.min_samples))
        keys = self.group_counts[self.group_counts >= n_samples].sample(self.batch_size).index
        sampled = self.data[self.data[self.group_parameter].isin(keys)] \
            .groupby(self.group_parameter).apply(lambda x: x.sample(n_samples))
        x, y = self._to_x_y(sampled)
        if self.regression:
            y = np.expand_dims(y, -1)
        else:
            y = to_one_hot(y.astype('int'))

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        if self.cuda:
            x = x.cuda()
            y = y.cuda()

        n_test = min(self.n_test, int(n_samples / 2))
        if self.with_train:
            return x[:, :-n_test], y[:, :-n_test], x, y
        return x[:, :-n_test], y[:, :-n_test], x[:, -n_test:], y[:, -n_test:]

    def __call__(self):
        return self.sample()
