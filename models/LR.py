import os.path

import networkx as nx
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from fastnode2vec import Graph, Node2Vec
from sklearn.linear_model import LogisticRegression
from itertools import product
from recommendation import *
import multiprocessing as mp


class LR:
    name = 'Logistic Regression + node2vec'
    def __init__(self, dataloader, args, logger):
        self.A = dataloader.A
        self.G = dataloader.G
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.lr = LogisticRegression()
        self.node_embeddings = self.getNodeEmbeddings(self.G, self.args.node_emb_dim)
        self.trainRecommender(self.G, self.node_embeddings)

    def recommend(self):
        all_edges = np.array([list(e) for e in product(np.arange(self.G.number_of_nodes()), repeat=2)])
        X_test = self.getX(all_edges, self.node_embeddings)
        y_pred = self.lr.predict(X_test)
        Aplus = np.zeros_like(self.A, dtype=np.float64)
        Aplus[all_edges[:, 0], all_edges[:, 1]] = y_pred

        Aplus -= np.diag(np.diagonal(Aplus))
        Aplus *= self.dataloader.target_adj_mask
        Aplus *= (2 * self.args.beta / Aplus.sum())
        if Aplus.max() > self.args.d:
            Aplus = clip(Aplus, self.args.d)
        return Recommendation(Aplus, new_conflict=None, dataloader=self.dataloader)

    def getNodeEmbeddings(self, G, dim):
        dir = self.args.data_dir + self.args.dataset +'/'
        cache = dir + 'emb{}_{}.npy'.format(G.number_of_nodes(), dim)
        # print(cache, 'exists?' ,os.path.exists(cache))
        if os.path.exists(cache):
            self.logger.info('Found cached node embeddings: {}'.format(cache))
            return np.load(cache)

        G = Graph(G.edges, directed=False, weighted=False)
        n2v = Node2Vec(G, dim=dim, walk_length=80, context=10, p=2.0, q=0.5, workers=mp.cpu_count()//2)
        n2v.train(epochs=100)
        embeddings = np.zeros((self.G.number_of_nodes(), dim))
        for i in range(self.G.number_of_nodes()):
            if i in n2v.wv:
                embeddings[i] = n2v.wv[i]
        np.save(cache, embeddings)
        return embeddings

    def trainRecommender(self, G, node_embeddings):
        pos_X = self.getX(np.array(G.edges), node_embeddings)
        neg_X = self.getX(np.random.randint(0, len(node_embeddings), size=(G.number_of_edges(), 2)), node_embeddings)

        y = np.concatenate([np.ones(len(pos_X)), np.zeros(len(neg_X))])
        self.lr.fit(np.concatenate([pos_X, neg_X], axis=0), y)

    def getX(self, edges, node_embeddings):
        u_emb = node_embeddings[edges[:, 0]]
        v_emb = node_embeddings[edges[:, 1]]
        return u_emb * v_emb

