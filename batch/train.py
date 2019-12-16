import logging
import os

import torch
from torch import nn
from torch.optim import Adam

from batch.evaluation import estimate_model_accuracy
from batch.utils import set_seed, EPS


def train_model(model, sampler, norm, name, path, r_sparse=0.1, iterations=50000, seed=0, patience=10):
    filename = os.path.join(path, name)
    set_seed(seed)

    y_dim = 2
    lr = 3e-4
    best_acc = 0
    rounds = 0

    optimizer = Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    for i in range(iterations):
        x_train, y_train, x_test, y_test = sampler()
        y_pred = model(x_train, y_train, x_test)
        loss = criterion(torch.log(y_pred.reshape(-1, y_dim) + EPS), y_test.reshape(-1, y_dim).argmax(1))

        reg = norm(model, x_train, y_train)
        loss += r_sparse * reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            acc = estimate_model_accuracy(model, sampler)
            if acc > best_acc:
                best_acc = acc
                rounds = 0
                torch.save(model.state_dict(), filename)
            else:
                rounds += 1

            logging.info('iteration {0}: {1}, patience rounds {2}'.format(i, acc, rounds))

            if rounds > patience:
                break

    model.load_state_dict(torch.load(filename))
    acc = estimate_model_accuracy(model, sampler)
    logging.info('Final: {0}'.format(acc))
    return model
