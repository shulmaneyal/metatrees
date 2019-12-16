import argparse
import logging
import os

from batch.evaluation import estimate_model_accuracy
from batch.norms import sparse_norm
from batch.train import train_model
from batch.tree_model import get_classification_tree
from batch.tree_model_dyn import get_classification_tree as get_classification_tree_dyn
from batch.utils import set_seed, DIST_HARD, DIST_SOFT
from data.synthetic import get_synthetic_varied
from run_utils import set_logging, get_dist_func


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='model name', default='')

    parser.add_argument('--height', type=int, help='tree height')
    parser.add_argument('--r_dim', type=int, help='encoder dimensions', default=512)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--dist_func', choices=['soft', 'hard'])

    parser.add_argument('-all_train', action='store_true')
    parser.add_argument('--r_sparse', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, help='batch_size', default=256)

    parser.add_argument('--x_dim', type=int)
    parser.add_argument('--prior_factor', type=float)
    parser.add_argument('--n_test', type=int)
    parser.add_argument('--min_n_train', type=int)
    parser.add_argument('--max_n_train', type=int)

    args = parser.parse_args()
    return args


def run(name, height, r_dim, temperature, batch_size, all_train, r_sparse, dist_func_name, x_dim, prior_factor, n_test,
        min_n_train, max_n_train):
    problem_name = '{0}_{1}_{2}_{3}'.format(x_dim, prior_factor, min_n_train, max_n_train)
    problem_name = os.path.join('syn', problem_name)
    model_name = '{0}height({1})_rdim({2})_temp({3})_rsparse({4})_dist({5})_batch_size({6})'.format(name, height, r_dim,
                                                                                                    temperature,
                                                                                                    r_sparse,
                                                                                                    dist_func_name,
                                                                                                    batch_size)

    log_path = os.path.join('models', problem_name)
    os.makedirs(log_path, exist_ok=True)
    set_logging(log_path, model_name)
    set_seed(0)

    sampler = lambda: get_synthetic_varied(batch_size, min_n_train, max_n_train, n_test, x_dim, prior_factor, cuda=True,
                                           with_train=all_train)

    dist_func = get_dist_func(dist_func_name)
    if height == 0:
        model = get_classification_tree_dyn(x_dim, 2, r_dim, 5, dist_func, temperature).cuda()
    else:
        model = get_classification_tree(x_dim, 2, r_dim, height, dist_func, temperature).cuda()

    model = train_model(model, sampler, sparse_norm, model_name, log_path, r_sparse=r_sparse)

    logging.info('Evaluating on test')
    sampler = lambda: get_synthetic_varied(64, min_n_train, max_n_train, n_test, x_dim, prior_factor, cuda=True,
                                           with_train=False)
    model.set_dist_func(DIST_SOFT)
    accuracy = estimate_model_accuracy(model, sampler)
    logging.info('Accuracy {0} (soft, soft)'.format(accuracy))
    model.set_dist_func(DIST_HARD)
    accuracy = estimate_model_accuracy(model, sampler)
    logging.info('Accuracy {0} (hard, soft)'.format(accuracy))

    model.make_hard_routing()
    model.set_dist_func(DIST_SOFT)
    accuracy = estimate_model_accuracy(model, sampler)
    logging.info('Accuracy {0} (soft, hard)'.format(accuracy))
    model.set_dist_func(DIST_HARD)
    accuracy = estimate_model_accuracy(model, sampler)
    logging.info('Accuracy {0} (hard, hard)'.format(accuracy))


if __name__ == '__main__':
    args = parse_args()
    run(args.name, args.height, args.r_dim, args.temperature, args.batch_size, args.all_train, args.r_sparse,
        args.dist_func, args.x_dim, args.prior_factor, args.n_test, args.min_n_train, args.max_n_train)
