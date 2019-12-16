import logging
import os

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from batch.evaluation import estimate_model_rmse
from batch.norms import sparse_norm
from batch.utils import set_seed


def train_model(model, sampler, r_sparse, name, path, lr=3e-4, iterations=50000, seed=0, patience=5):
    pathname = os.path.join('models', path)
    filename = os.path.join(pathname, name)
    os.makedirs(pathname, exist_ok=True)
    set_seed(seed)

    best_rmse = np.inf
    rounds = 0

    optimizer = Adam(model.parameters(), lr)
    criterion = nn.MSELoss()
    for i in range(1, iterations + 1):
        x_train, y_train, x_test, y_test = sampler()
        y_pred = model(x_train, y_train, x_test)
        loss = criterion(y_pred.reshape(-1), y_test.reshape(-1))

        sparse = sparse_norm(model, x_train, y_train)
        loss += r_sparse * sparse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            rmse = estimate_model_rmse(model, sampler)

            if rmse < best_rmse:
                best_rmse = rmse
                rounds = 0
                torch.save(model.state_dict(), filename)
            else:
                rounds += 1

            logging.info('iteration {0}: {1}, patience rounds {2}'.format(i, rmse, rounds))

            if rounds > patience:
                break

    model.load_state_dict(torch.load(filename))
    rmse = estimate_model_rmse(model, sampler)
    logging.info('Final: {0}'.format(rmse))
    return model
