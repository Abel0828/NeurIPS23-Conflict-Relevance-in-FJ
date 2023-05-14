import networkx as nx
from sklearn.metrics import ndcg_score
import numpy as np
from recommendation import clip, Recommendation


class CommonNeighbors:
    name = 'CommonNeighbors'
    def __init__(self, dataloader, args, logger):
        self.A = dataloader.A
        self.G = dataloader.G
        self.args = args
        self.logger = logger
        self.dataloader = dataloader

    def recommend(self):
        Aplus = self.A @ self.A
        Aplus -= np.diag(np.diagonal(Aplus))
        Aplus *= self.dataloader.target_adj_mask
        Aplus = Aplus*(2*self.args.beta / Aplus.sum())
        if Aplus.max() > self.args.d:
            Aplus = clip(Aplus, self.args.d)
        return Recommendation(Aplus, new_conflict=None, dataloader=self.dataloader)

