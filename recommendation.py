import numpy as np


class Recommendation:
    def __init__(self, Aplus, new_conflict, dataloader, edge_logits=None):
        # notice that the inputs are the results from recommenders who don't know the ground truth edges

        self.Aplus_with_neg_edges = Aplus



        self.Aplus = Aplus * dataloader.reserved_adj_mask  # CRUCIAL STEP: these are edges that are actually valid,
        self.s = dataloader.s
        self.old_conflict = dataloader.old_conflict
        self.new_conflict = dataloader.computeConflict(np.diag(self.Aplus.sum(0)) - self.Aplus + dataloader.L, self.s)

        self.new_conflict_with_neg_edges = new_conflict
        if self.new_conflict_with_neg_edges is None:
            self.new_conflict_with_neg_edges = dataloader.computeConflict(np.diag(self.Aplus_with_neg_edges.sum(0))-
                                                           self.Aplus_with_neg_edges+dataloader.L,
                                                           self.s)

        self.diff = self.new_conflict - self.old_conflict
        self.percent = -self.diff/self.old_conflict*100
        self.ca = None

        self.edge_logits = edge_logits
        if self.edge_logits is None:
            target_edges = dataloader.target_edges
            self.edge_logits = Aplus[target_edges[:, 0], target_edges[:, 1]] # notice that this can NOT be self.Aplus
        self.precision_at_k, self.recall = self.compute_precision_recall(self.edge_logits,
                                                                         dataloader.target_edges_labels,
                                                                         k=dataloader.args.precisionk)


    def printResults(self, logger=None):
        pfunc = logger.info
        if logger is None:
            pfunc = print

        pfunc('[Recommendation] old conflict: {:.6f}, new conflict: {:.6f}, change:  {:.6f} ({:.4f}%)'.format(self.old_conflict, self.new_conflict, self.diff, self.percent))

        pfunc('[Recommendation] weight of recommended edges {:.6f}+-{:.6f} (total={:.6f}, self loop={:.6f}), max single weight={:.6f}'.format(self.Aplus.mean(),
                                                                                                    self.Aplus.std(),
                                                                                                    self.Aplus.sum()/2,
                                                                                                    np.trace(self.Aplus),
                                                                                                                                              self.Aplus.max()))
    def computeConflictAwareness(self, rec_model):
        conflict_awareness = rec_model.percent / self.percent
        return conflict_awareness

    def compute_precision_recall(self, edge_logits, edge_labels, k):

        k = (np.cumsum(edge_logits) < k).sum()
        ranked_list = sorted(list(zip(edge_logits, edge_labels)), reverse=True)
        # print(ranked_list)
        precision_at_k = sum([e[0]*e[1] for e in ranked_list[:k]]) / sum([e[0] for e in ranked_list[:k]])
        recall = sum([e[0]*e[1] for e in ranked_list]) / edge_labels.sum()
        return precision_at_k, recall



def clip(Aplus, d):
    mask = new_mask = (Aplus > d)
    s = Aplus.sum()

    while new_mask.any():
        Aplus = np.clip(Aplus, a_min=None, a_max=d)
        remain = s - Aplus.sum()
        Aplus_canchange = Aplus * (1 - mask)
        Aplus += Aplus_canchange * remain / Aplus_canchange.sum()
        new_mask = Aplus > d
        mask += new_mask
        print('redistribute recommended weight', remain)
    return Aplus



