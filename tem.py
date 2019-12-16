import argparse
import logging
import torch
from torch import nn
from torch.optim import Adagrad
from torch.utils import data
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np

from batch.utils import set_seed
from run_utils import set_logging, get_data


class Dataset(data.Dataset):
    def __init__(self, users, items, x, y, cuda):
        self.users = users
        self.items = items
        self.x = x
        self.y = y
        self.cuda = cuda

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        u, i, x, y = self.users[index], self.items[index], self.x[index], self.y[index]
        if self.cuda:
            u, i, x, y = u.cuda(), i.cuda(), x.cuda(), y.cuda()
        return u, i, x, y


class TEM(nn.Module):
    def __init__(self, a, k, x_dim, n_users, n_items, gbdt, mean_pool=True, sigmoid=True):
        super(TEM, self).__init__()
        self.n_trees = gbdt.n_estimators
        self.gbdt = gbdt
        self.mean_pool = mean_pool
        self.sigmoid = sigmoid
        n_features = self.n_trees * 2 ** (gbdt.max_depth + 1)
        self.emb_users = nn.Embedding(n_users, k)
        self.emb_items = nn.Embedding(n_items, k)
        self.emb_features = nn.Embedding(n_features, k)

        self.W = nn.Linear(2 * k, a)
        self.h = nn.Linear(a, 1, False)

        self.r1 = nn.Linear(k, 1, False)
        self.r2 = nn.Linear(k, 1, False)
        self.b = nn.Linear(x_dim, 1)

    def _get_features(self, x):
        return torch.tensor(get_features(self.gbdt, x.cpu().numpy())).long().cuda()

    def forward(self, users, items, x):
        features = self._get_features(x)
        p_u = self.emb_users(users)
        q_i = self.emb_items(items)
        pq = p_u * q_i
        pq_att = pq.reshape(pq.shape[0], 1, -1).repeat(1, self.n_trees, 1)
        V = self.emb_features(features)
        att_in = torch.cat([V, pq_att], 2)
        w_tag = self.h(nn.functional.relu(self.W(att_in)))
        w = torch.softmax(w_tag, dim=1)
        if self.mean_pool:
            e = (V * w).mean(1)
        else:
            e = (V * w).max(1)[0]
        y_pred = self.b(x) + self.r1(pq) + self.r2(e)
        if not self.sigmoid:
            return y_pred
        return 1 + 4 * torch.sigmoid(y_pred)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['ml_100k', 'ml_1m', 'jester'], required=True)

    parser.add_argument('--n_trees', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--a', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)

    args = parser.parse_args()
    return args


def prepare_data(train):
    remove_x = ['user_id', 'item_id', 'rating']
    x_train = train.drop(remove_x, axis=1).values
    y_train = train.rating.values
    users_train = train.user_id.values - 1
    items_train = train.item_id.values - 1
    return items_train, users_train, x_train, y_train


def get_dataset(items, users, x, y):
    users = torch.tensor(users).long()
    items = torch.tensor(items).long()
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    training_set = Dataset(users, items, x, y, True)
    return training_set


def get_leaves(tree, x, n_nodes):
    p = tree.decision_path(x)
    s, n = p.nonzero()
    leaves = pd.Series(n).groupby(s).max().values
    leaves = np.eye(n_nodes)[leaves]
    leaves = leaves[:, leaves.any(axis=0)]
    return leaves


def get_features(gbdt, x):
    n_nodes = 2 ** (gbdt.max_depth + 1) - 1
    n_trees = len(gbdt.estimators_)
    all_leaves = [get_leaves(tree, x, n_nodes) for i, tree in enumerate(gbdt.estimators_.flatten())]
    features = np.concatenate(all_leaves, axis=1)
    _, features = features.nonzero()
    features = features.reshape(-1, n_trees)
    return features


def train_model(model, lr, epochs, train_loader, val_loader, patience):
    optimizer = Adagrad(model.parameters(), lr)
    criterion = nn.MSELoss()

    best_rmse = 100
    rounds_no_imporve = 0
    for epoch in range(epochs):
        for users, items, x, y in train_loader:
            y_pred = model(users, items, x)
            loss = criterion(y_pred.reshape(-1), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logging.info('Last train loss: {0:.3f}'.format(loss.detach().cpu().numpy().tolist()))
        with torch.no_grad():
            errors = np.array([])
            for users, items, x, y in val_loader:
                y_pred = model(users, items, x)
                group_errors = (y_pred - y).reshape(-1).cpu().numpy()
                errors = np.concatenate([errors, group_errors])
            rmse = (errors ** 2).mean() ** 0.5
            logging.info('Test RMSE: {0:.3f}'.format(rmse))
            if rmse < best_rmse:
                best_rmse = rmse
                rounds_no_imporve = 0
            else:
                rounds_no_imporve += 1
            if rounds_no_imporve >= patience:
                return model
    return model


def run(data_name, n_trees, max_depth, k, a, batch_size, lr, epochs, patience):
    set_seed(0)
    log_path = 'tem_logs'
    filename = 'data={0}_ntrees={1}_maxdepth={2}_k={3}_a={4}_lr={5}'.format(data_name, n_trees, max_depth, k, a, lr)
    set_logging(log_path, filename)

    logging.info('Loading data {0}'.format(data_name))
    train, test = get_data(data_name, False)

    items_train, users_train, x_train, y_train = prepare_data(train)
    items_test, users_test, x_test, y_test = prepare_data(test)

    logging.info('Normalizing data')
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    logging.info('Extracting features using GBDT')
    gbdt = GradientBoostingRegressor(n_estimators=n_trees, max_depth=max_depth)
    gbdt = gbdt.fit(x_train, y_train)

    x_dim = x_train.shape[1]
    n_users = int(users_train.max() + 1)
    n_items = int(items_train.max() + 1)

    train_set = get_dataset(items_train, users_train, x_train, y_train)
    test_set = get_dataset(items_test, users_test, x_test, y_test)
    train_loader = data.DataLoader(train_set, batch_size=batch_size)
    # using test as validation for now
    val_loader = data.DataLoader(test_set, batch_size=batch_size)

    model = TEM(a, k, x_dim, n_users, n_items, gbdt).cuda()
    logging.info('Training TEM')
    train_model(model, lr, epochs, train_loader, val_loader, patience)
    logging.info('Done')


if __name__ == '__main__':
    args = parse_args()
    run(args.data, args.n_trees, args.max_depth, args.k, args.a, args.batch_size, args.lr, args.epochs, args.patience)
