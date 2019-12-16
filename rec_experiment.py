import argparse
import logging
import os

from sklearn.preprocessing import StandardScaler

from batch.evaluation import get_regression_performance
from batch.sampler import Sampler
from batch.train_regression import train_model
from batch.tree_model import get_regression_tree
from batch.tree_model_dyn import get_regression_tree as get_regression_tree_dyn
from batch.utils import to_x_y, set_seed, DIST_HARD, DIST_SOFT
from run_utils import set_logging, get_dist_func, get_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='model name', default='')

    parser.add_argument('--height', type=int, help='tree height')
    parser.add_argument('--r_dim', type=int, help='encoder dimensions', default=512)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--dist_func', choices=['soft', 'hard'])

    parser.add_argument('--data', choices=['ml_100k', 'ml_1m', 'jester'])

    parser.add_argument('-all_train', action='store_true')
    parser.add_argument('--r_sparse', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, help='batch_size', default=256)
    parser.add_argument('--n_test', type=int, default=5)
    parser.add_argument('--lr', type=float)

    args = parser.parse_args()
    return args


def run(name, batch_size, r_dim, height, all_train, n_test, data_name, dist_func_name, r_sparse, lr, temperature):
    problem_name = 'train({0})_test({1})'.format(all_train, n_test)
    problem_name = os.path.join(data_name, problem_name)
    model_name = '{0}height({1})_rdim({2})_temp({3})_rsparse({4})_dist({5})_batch_size({6})_lr({7})'.format(
        name, height, r_dim, temperature, r_sparse, dist_func_name, batch_size, lr)

    log_path = os.path.join('models', problem_name)
    set_logging(log_path, model_name)
    set_seed(0)
    logging.info('running {0} using {1}'.format(problem_name, model_name))

    train, test = get_data(data_name)
    group_parameter = 'user_id'
    target = 'rating'
    x_train, y_train = to_x_y(train, group_parameter, target)
    preprocessor = StandardScaler()
    preprocessor.fit(x_train.astype('float'))
    x_dim = x_train.shape[1]
    sampler = Sampler(batch_size, train, preprocessor, n_test, group_parameter, target, cuda=True, with_train=all_train,
                      regression=True)
    min_rating = y_train.min()
    max_rating = y_train.max()

    dist_func = get_dist_func(dist_func_name)
    if height == 0:
        model = get_regression_tree_dyn(x_dim, min_rating, max_rating, r_dim, 5, dist_func, temperature).cuda()
    else:
        model = get_regression_tree(x_dim, min_rating, max_rating, r_dim, height, dist_func, temperature).cuda()

    model = train_model(model, sampler, r_sparse, model_name, log_path, lr)

    logging.info('Evaluating on test')
    model.set_dist_func(DIST_SOFT)
    rmse, mae = get_regression_performance(train, test, model, preprocessor, group_parameter, target)
    logging.info('RMSE {0}, MAE {1} (soft, soft)'.format(rmse, mae))
    model.set_dist_func(DIST_HARD)
    rmse, mae = get_regression_performance(train, test, model, preprocessor, group_parameter, target)
    logging.info('RMSE {0}, MAE {1} (hard, soft)'.format(rmse, mae))
    model.make_hard_routing()
    model.set_dist_func(DIST_SOFT)
    rmse, mae = get_regression_performance(train, test, model, preprocessor, group_parameter, target)
    logging.info('RMSE {0}, MAE {1} (soft, hard)'.format(rmse, mae))
    model.set_dist_func(DIST_HARD)
    rmse, mae = get_regression_performance(train, test, model, preprocessor, group_parameter, target)
    logging.info('RMSE {0}, MAE {1} (hard, hard)'.format(rmse, mae))


if __name__ == '__main__':
    args = parse_args()
    run(args.name, args.batch_size, args.r_dim, args.height, args.all_train, args.n_test, args.data, args.dist_func,
        args.r_sparse, args.lr, args.temperature)
