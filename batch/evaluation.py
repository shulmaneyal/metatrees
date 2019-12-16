import numpy as np
import torch

from batch.utils import to_one_hot, to_x_y


def _get_acc(y_test, y_pred):
    acc = (y_test.argmax(2) == y_pred.argmax(2)).float().mean().cpu().numpy().round(3)
    return acc


def _prepare_x_y(data, group_parameter, target, preprocessor, classification=True):
    x, y = to_x_y(data, group_parameter, target)
    x = preprocessor.transform(x.astype('float'))
    if classification:
        y = to_one_hot(y.astype('int'))
    else:
        y = y.values
    x = torch.from_numpy(x).float().unsqueeze(0)
    y = torch.from_numpy(y).float().unsqueeze(0)
    return x, y


def estimate_model_accuracy(model, sampler, batches=100):
    accs = []
    for _ in range(batches):
        x_train, y_train, x_test, y_test = sampler()
        y_pred = model(x_train, y_train, x_test)
        accs.append(_get_acc(y_test, y_pred))
    acc = sum(accs) / len(accs)
    return acc


def estimate_model_rmse(model, sampler, batches=20):
    errors = []
    for _ in range(batches):
        x_train, y_train, x_test, y_test = sampler()
        y_pred = model(x_train, y_train, x_test)
        batch_errors = (y_test.reshape(-1) - y_pred.reshape(-1)).detach().cpu().numpy()
        errors = np.concatenate([errors, batch_errors])
    rmse = (errors ** 2).mean() ** 0.5
    return rmse


def get_regression_performance(train, test, model, preprocessor, group_parameter, target):
    model.cpu()
    groups = test[group_parameter].unique()
    errors = np.array([])
    for group in groups:
        group_train = train[train[group_parameter] == group]
        group_test = test[test[group_parameter] == group]
        x_train, y_train = _prepare_x_y(group_train, group_parameter, target, preprocessor, False)
        x_test, y_test = _prepare_x_y(group_test, group_parameter, target, preprocessor, False)
        y_train = y_train.reshape(1, -1, 1)
        y_test = y_test.reshape(1, -1, 1)
        y_pred = model(x_train, y_train, x_test)
        group_errors = (y_test.reshape(-1) - y_pred.reshape(-1)).detach().cpu().numpy()
        errors = np.concatenate([errors, group_errors])
    rmse = (errors ** 2).mean() ** 0.5
    mae = np.abs(errors).mean()
    return rmse, mae
