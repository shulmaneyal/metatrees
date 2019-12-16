import logging
import os
import sys

from batch.utils import DIST_HARD, DIST_SOFT
from data.data_getter import get_movie_lens_100k, get_movie_lens_1m, get_jester


class StreamToLogger:
    """  Fake file-like stream object that redirects writes to a logger instance. """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def set_logging(log_path, filename):
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}.log".format(log_path, filename)),
            logging.StreamHandler()
        ])
    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger('STDERR')
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl


def get_dist_func(dist_func_name):
    if dist_func_name == 'soft':
        return DIST_SOFT
    return DIST_HARD


def get_data(data_name, subset=True):
    if data_name == 'ml_100k':
        return get_movie_lens_100k(subset)
    elif data_name == 'ml_1m':
        return get_movie_lens_1m(subset)
    return get_jester(subset)
