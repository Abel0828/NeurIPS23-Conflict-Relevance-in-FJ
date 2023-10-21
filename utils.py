import argparse
import numpy as np
import random
import os
import sys
# import torch
import logging
import time
import warnings
import torch

def setUp():
    args, sys_argv = getArgs()
    logger = set_up_logger(args, sys_argv)
    setRandomSeed(args.seed)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    return args, logger


def getArgs():
    parser = argparse.ArgumentParser('Interface for conflict-aware social recommendation experiments')
    parser.add_argument('--dataset', type=str, default='twitter', help='dataset name')
    parser.add_argument('--solver', type=str, default='SCS', help='solver name for computing the optimal recommendation')
    parser.add_argument('--beta', type=float, default=10.0, help='budget of recommendation')
    parser.add_argument('--d', type=float, default=5.0, help='upper bound of weight per recommended link')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity for printing solver state')
    parser.add_argument('--n', type=int, default=9, help='negative edge ratio')
    parser.add_argument('--all_edges', type=int, default=0, help='whether to allow for optimizing the whole graph')

    parser.add_argument('--ppr_alpha', type=float, default=0.85, help='1 - telescope probability for PageRank')
    parser.add_argument('--katz_alpha', type=float, default=0.15, help='decay rate for Katz')
    parser.add_argument('--node_emb_dim', type=int, default=64, help='node2vec embeddings dim')

    parser.add_argument('--downsample', type=int, default=0, help='verbosity for printing solver state')

    # parser.add_argument('--neg_rate', type=int, default=1, help='neg:pos rate in training set')
    parser.add_argument('--gpu', type=int, default=1, help='gpu to use')
    parser.add_argument('--k', type=int, default=10, help='NDCG@k')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs for GNN training')
    parser.add_argument('--precisionk', type=int, default=10, help='precision@k')
    parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized modules')
    parser.add_argument('--data_dir', type=str, default='./data/', help='data directory')
    parser.add_argument('--log_dir', type=str, default='log', help='log root directory')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv


def setRandomSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_up_logger(args, sys_argv):
    # set up running log
    runtime_id = '{}-{}'.format(args.dataset, str(time.time()))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_dir = '{}/{}/'.format(args.log_dir, args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_path = log_dir + runtime_id + '.log'
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)
    return logger