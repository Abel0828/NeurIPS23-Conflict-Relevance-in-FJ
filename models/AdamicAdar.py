import networkx as nx
from sklearn.metrics import ndcg_score
import numpy as np
from recommendation import clip, Recommendation


class AdamicAdar:
    name = 'AdamicAdar'
    def __init__(self, dataloader, args, logger, ):
        self.A = dataloader.A
        self.G = dataloader.G
        self.args = args
        self.logger = logger
        self.dataloader = dataloader

    def recommend(self):

        Aplus = np.zeros_like(self.A, dtype=np.float64)
        for u,v,p in nx.preferential_attachment(self.G):
            Aplus[u, v] = p
        Aplus += Aplus.T
        Aplus *= self.dataloader.target_adj_mask
        Aplus *= (2*self.args.beta / Aplus.sum())
        if Aplus.max() > self.args.d:
            Aplus = clip(Aplus, self.args.d)
        return Recommendation(Aplus, new_conflict=None, dataloader=self.dataloader)