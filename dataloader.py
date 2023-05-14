import os
import pickle
import numpy as np
import networkx as nx
import random
from copy import deepcopy
from scipy.sparse.linalg import inv
from scipy.linalg import solve
from scipy import sparse
import torch

def get_device(args):
    gpu = args.gpu
    return torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

class DataLoader:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.beta = args.beta
        self.G, self.reserved_edges, self.reserved_adj_mask = self.load_graph(n_reserved=int(self.beta)) # load graph, reserve some edges
        self.target_edges, self.target_adj_mask, self.target_edges_labels = self.getTargetEdgeList(self.reserved_edges, self.reserved_adj_mask)
        if self.args.all_edges == 1:
            self.reserved_adj_mask = self.target_adj_mask = np.ones_like(self.reserved_adj_mask)
            self.reserved_edges = self.target_edges = None
        self.A = np.asarray(nx.adjacency_matrix(self.G, dtype=np.int64).todense())
        self.L = np.diag(self.A.sum(0)) - self.A
        self.s = self.load_opinions()
        self.old_conflict = self.computeConflict(self.L, self.s)
        self.device = get_device(self.args)


        self.logger.info('[DataLoader] cleaned graph nodes {}, edges {}'.format(self.G.number_of_nodes(),
                                                                                self.G.number_of_edges()))
        # self.attributes = self.load_attributes()
        # (self.X_train, self.y_train, self.train_set), \
        # (self.X_test, self.y_test, self.test_set),\
        #     self.G = self.get_Xy(self.G_org, self.attributes, self.args.n, self.args.neg_rate)
        # self.G_adj = nx.adjacency_matrix(self.G)
        # self.y_conflict_train, self.y_conflict_test = self.get_y_conflict(self.G, self.train_set, self.test_set)
        # self.logger.info('[DataLoader] cleaned graph nodes {}, edges {}')
        # self.logger.info('[DataLoader] pos edges {}, neg edges {} conflict {:.3f}+-{:.3f}'.format(int(self.y_test.sum()),
        #                                                                                      len(self.X_test)-int(self.y_test.sum()),
        #                                                                                      self.y_conflict_test.mean()
        #                                                                                      ,self.y_conflict_test.std()))

    def getTargetEdgeList(self, reserved_edges, reserved_adj_mask):
        num_negative_edges = len(reserved_edges) * self.args.n
        negative_edges = np.random.randint(low=0, high=reserved_adj_mask.shape[0], size=(num_negative_edges, 2))
        negative_adj_mask = np.zeros_like(reserved_adj_mask)
        negative_adj_mask[negative_edges[:, 0], negative_edges[:, 1]] = 1
        negative_adj_mask += negative_adj_mask.T
        target_edges = np.concatenate([reserved_edges, negative_edges], axis=0)
        target_edges_labels = np.concatenate([np.ones(len(reserved_edges)), np.zeros(num_negative_edges)])
        self.logger.info('Constrained edge set size: {}'.format(len(target_edges)))
        return target_edges, \
               np.array(negative_adj_mask + reserved_adj_mask > 0, dtype=np.int64), \
               target_edges_labels

    def computeConflict(self, L, s):
        I = np.eye(L.shape[0])
        c = s.T @ solve(I + L, s)
        return c

    def load_opinions(self):
        s = []
        with open(self.args.data_dir + self.args.dataset + '/opinions.txt', 'r') as f:
            for line in f.readlines():
                s.append(float(line.split('\t')[-1]))
        s = np.array(s)
        if self.args.downsample > 0:
            s = s[:self.args.downsample]
        s -= s.mean()
        return s

    def load_graph(self, n_reserved):
        edges = np.loadtxt(self.args.data_dir + '{}/edges.txt'.format(self.args.dataset), dtype=np.int64)
        if not self.sanity_check(edges):
            raise Exception
        G = nx.Graph()
        G.add_edges_from(edges)
        if self.args.downsample>0:
            G = G.subgraph(np.arange(self.args.downsample))
            G = G.copy()

        edges = list(G.edges)
        print(len(edges), '++'*30)
        V = G.number_of_nodes()
        reserved_edges_i = random.sample(range(G.number_of_edges()), n_reserved)
        reserved_edges = np.array([edges[i] for i in reserved_edges_i])
        reserved_adj_mask = np.zeros((V, V))
        reserved_adj_mask[reserved_edges[:, 0], reserved_edges[:, 1]] = 1
        reserved_adj_mask += reserved_adj_mask.T
        # todo: further remove false negative edges
        G.remove_edges_from(reserved_edges)
        return G, reserved_edges, reserved_adj_mask>0

    def load_attributes(self):
        attributes = np.load(self.args.data_dir + '{}/features.npy'.format(self.args.dataset))
        return attributes

    def get_Xy(self, G, attributes, n, neg_rate):
        # To Do
        # sample train pos, sample test pos
        # remove train pos, remove test pos
        # load features
        # make y
        edges = list(G.edges)
        nodes = list(G.nodes)
        pos_edges_i = random.sample(range(G.number_of_edges()), 2*n)
        pos_edges = np.array([edges[i] for i in pos_edges_i])
        neg_edges = np.array(random.choices(nodes, k=4*n*neg_rate)).reshape(-1, 2)
        # todo: further remove false negative edges

        G2 = G.copy()
        G2.remove_edges_from(pos_edges)

        train_set = np.concatenate([pos_edges[:n], neg_edges[:n*neg_rate]], axis=0)
        test_set = np.concatenate([pos_edges[:n], neg_edges[:n*neg_rate]], axis=0)

        X_train, X_test = self.load_X(attributes, train_set, test_set, embeddings=None)
        y_train = np.concatenate([np.ones(n), np.zeros(n*neg_rate)], axis=0)
        y_test = deepcopy(y_train)

        return (X_train, y_train, train_set), (X_test, y_test, test_set), G2

    def get_y_conflict(self, G, train_set, test_set):
        L = nx.laplacian_matrix(G)
        matrix_forest = self.get_matrix_forest(L, checksum=sum([u+v for u, v in G.edges()]))  # essentially compute (I+L)^(-1) or load from cache

        y_conflict_train = np.array(
            [self.compute_conflict_reduction_link(edge, matrix_forest) for edge in train_set])
        y_conflict_test = np.array(
            [self.compute_conflict_reduction_link(edge, matrix_forest) for edge in test_set])
        return y_conflict_train, y_conflict_test

    def get_matrix_forest(self, L, checksum):
        cname = 'data/{}/matrix_forest.pkl'.format(self.args.dataset)
        if os.path.exists(cname):
            with open(cname, 'rb') as f:
                cache = pickle.load(f)
                if cache['checksum'] == checksum:
                    self.logger.info('[DataLoader] Load cached (I+L)^-1.')
                    return cache['data']

        I = sparse.eye(L.shape[0])
        matrix_forest = inv((L + I).tocsc()).toarray()
        cache = {'checksum':checksum, 'data': matrix_forest}
        with open(cname, 'wb') as f:
            pickle.dump(cache, f)
        return matrix_forest

    def load_X(self, attributes, train_set, test_set, embeddings=None):
        X_train = self._pool_edge_features(attributes[train_set])
        X_test = self._pool_edge_features(attributes[test_set])
        return X_train, X_test

    def _pool_edge_features(self, X):
        return np.concatenate([X.sum(1), X.prod(1), np.abs(X[:, 0, :] - X[:, 1, :])], axis=-1)

    def compute_conflict_reduction_link(self, edge, matrix_forest):
        m = matrix_forest
        u, v = edge
        denom = 1 + m[u, u] + m[v, v] - m[u, v] - m[v, u]
        num = np.abs(matrix_forest[u] - matrix_forest[v]).sum()
        return num / denom

    def sanity_check(self, edges):
        nodes = np.unique(edges)
        n = len(nodes)
        if nodes.min()!= 0 or nodes.max() != n-1:
            return False
        return True


