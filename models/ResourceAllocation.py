import networkx as nx
import numpy as np
from recommendation import clip, Recommendation


class ResourceAllocation:
    name = 'Resource Allocation'

    def __init__(self, dataloader, args, logger):
        self.A = dataloader.A
        self.G = dataloader.G
        self.args = args
        self.logger = logger
        self.dataloader = dataloader

    def recommend(self):
        Aplus = np.zeros_like(self.A, dtype=np.float64)
        for u,v,p in nx.resource_allocation_index(self.G):
            Aplus[u, v] = p
        Aplus += Aplus.T
        Aplus *= self.dataloader.target_adj_mask
        Aplus *= (2 * self.args.beta / Aplus.sum())
        if Aplus.max() > self.args.d:
            Aplus = clip(Aplus, self.args.d)
        return Recommendation(Aplus, new_conflict=None, dataloader=self.dataloader)