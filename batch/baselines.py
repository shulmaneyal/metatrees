import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from surprise import Reader, Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import SVD, NMF, SVDpp, SlopeOne, BaselineOnly, NormalPredictor
import re
import pandas as pd

from batch.utils import to_x_y


def RMSE(y_true, y_pred):
    return MSE(y_true, y_pred) ** 0.5


def global_average(train, test, target):
    target_average = train[target].mean()
    y_true = test[target]
    y_pred = np.ones_like(y_true) * target_average
    return RMSE(y_true, y_pred), MAE(y_true, y_pred)


def one_tree(train, test, group_parameter, target, depth):
    tree = DecisionTreeRegressor(max_depth=depth, random_state=0)
    x_train, y_train = to_x_y(train, group_parameter, target)
    x_test, y_test = to_x_y(test, group_parameter, target)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    return RMSE(y_test, y_pred), MAE(y_test, y_pred)


def tree_per_group(train, test, group_parameter, target, depth):
    groups = test[group_parameter].unique()
    average = train[target].mean()

    y_tests = np.array([])
    y_preds = np.array([])

    for group in groups:
        user_train = train[train[group_parameter] == group]
        user_test = test[test[group_parameter] == group]
        tree = DecisionTreeRegressor(max_depth=depth, random_state=0)
        x_train, y_train = to_x_y(user_train, group_parameter, target)
        x_test, y_test = to_x_y(user_test, group_parameter, target)
        if len(x_train) == 0:
            y_pred = np.ones(len(y_test)) * average

        else:
            tree.fit(x_train, y_train)
            y_pred = tree.predict(x_test)

        y_tests = np.append(y_tests, y_test)
        y_preds = np.append(y_preds, y_pred)

    return RMSE(y_tests, y_preds), MAE(y_tests, y_preds)


def get_train_test(train, test):
    all_data = pd.concat([train, test], sort=False)
    rating_scale = (all_data.rating.min(), all_data.rating.max())
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(all_data[['user_id', 'item_id', 'rating']], reader)
    return train_test_split(data, test_size=len(test), shuffle=False)


def get_name(algo):
    return re.findall("\.(\w+)\'", str(algo))[0]


def surprise_algos(train, test, svdpp=False):
    train_set, test_set = get_train_test(train, test)
    algos = [NormalPredictor, BaselineOnly, SlopeOne, NMF, SVD]
    if svdpp:
        algos.append(SVDpp)
    values = {}
    values['Method'] = []
    values['RMSE'] = []
    values['MAE'] = []
    for algo_constructor in algos:
        name = get_name(algo_constructor)
        print(name)
        try:
            algo = algo_constructor(random_state=0)
        except:
            algo = algo_constructor()
        algo.fit(train_set)
        predictions = algo.test(test_set)

        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        values['Method'].append(name)
        values['RMSE'].append(rmse)
        values['MAE'].append(mae)
    return pd.DataFrame(values).sort_values('RMSE', ascending=False).set_index('Method')
