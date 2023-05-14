from sklearn.metrics import ndcg_score
import numpy as np
from recommendation import Recommendation, clip


class Katz:
    name = 'Katz'

    def __init__(self, dataloader, args, logger, approx_dist=5):
        self.G = dataloader.G
        self.G_adj = dataloader.A
        self.beta = args.beta
        self.alpha = args.katz_alpha
        self.approx_dist = approx_dist
        self.dataloader = dataloader
        self.args = args
        self.logger = logger

    def recommend(self):
        # adj is a csr matrix
        adj, alpha, approx_dist = self.G_adj, self.alpha, self.approx_dist
        result = alpha * adj
        for i in range(2, approx_dist + 1):
            result += (alpha ** i) * np.linalg.matrix_power(adj, i)
        Aplus = result
        Aplus -= np.diag(np.diagonal(Aplus))
        Aplus *= self.dataloader.target_adj_mask
        Aplus *= (2 * self.args.beta / Aplus.sum())
        if Aplus.max() > self.args.d:
            Aplus = clip(Aplus, self.args.d)

        return Recommendation(Aplus, new_conflict=None, dataloader=self.dataloader)








