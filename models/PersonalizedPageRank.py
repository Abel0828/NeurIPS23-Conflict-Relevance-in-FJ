import networkx as nx
from sklearn.metrics import ndcg_score
import numpy as np
from sklearn.preprocessing import normalize
from recommendation import Recommendation, clip


class PersonalizedPageRank:
    name = 'Personalized PageRank'

    def __init__(self, dataloader, args, logger, approx_dist=5):
        self.G = dataloader.G
        self.G_adj = dataloader.A
        self.beta = args.beta
        self.alpha = args.ppr_alpha
        self.approx_dist = approx_dist
        self.dataloader = dataloader
        self.args = args
        self.logger = logger

    def recommend(self):
        # adj is a csr matrix
        adj, alpha, approx_dist = self.G_adj, self.alpha, self.approx_dist
        adj_normalized = normalize(adj, norm='l1', axis=1)
        result = alpha * adj_normalized  # notice that (1-alpha) is omitted as it doesn't affect the ordering
        for i in range(2, approx_dist + 1):
            result += (alpha ** i) * np.linalg.matrix_power(adj, i)
        Aplus = result + result.T
        Aplus -= np.diag(np.diagonal(Aplus))
        Aplus *= self.dataloader.target_adj_mask
        Aplus *= (2*self.args.beta / Aplus.sum())
        if Aplus.max() > self.args.d:
            Aplus = clip(Aplus, self.args.d)
        return Recommendation(Aplus, new_conflict=None, dataloader=self.dataloader)
