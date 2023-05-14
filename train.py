from models.CommonNeighbors import CommonNeighbors
from models.Jaccard import Jaccard
from models.ResourceAllocation import ResourceAllocation
from models.AdamicAdar import AdamicAdar
from models.PreferentialAttachment import PreferentialAttachment
from models.LR import MLP
from models.PersonalizedPageRank import PersonalizedPageRank
from models.Katz import Katz


class Evaluator:
    def __init__(self, args, logger, dl):
        self.args = args
        self.logger = logger
        self.dataloader = dl
        self.models = self._init_models()
        self.logger.info('[Trainer] initialize models: ' + '; '.join(self.models.keys()))

    def run(self):
        self._train()
        self.logger.info('[Trainer] Start evaluation ...')
        self._eval()

    def _init_models(self):
        models = {}
        models['common neighbors'] = CommonNeighbors(self.dataloader.G)
        models['jaccard'] = Jaccard(self.dataloader.G)
        models['adamic adar'] = AdamicAdar(self.dataloader.G)
        models['mlp'] = MLP()
        models['resource allocation'] = ResourceAllocation(self.dataloader.G)
        models['preferential attachment'] = PreferentialAttachment(self.dataloader.G)
        models['personalized pagerank'] = PersonalizedPageRank(self.dataloader.G, self.dataloader.G_adj)
        models['katz'] = Katz(self.dataloader.G, self.dataloader.G_adj)
        return models

    def _train(self):

        dl = self.dataloader
        X_train, y_train, y_conflict_train, X_test, y_test, y_conflict_test = \
            dl.X_train, dl.y_train, dl.y_conflict_train, dl.X_test, dl.y_test, dl.y_conflict_test

        self.models['mlp'].fit(X_train, y_train)
        # jcd =
        # cn =
        # aa =
        # ra =
        # katz = Katz()
        # pr =

    def _eval(self):
        dl = self.dataloader
        k = self.args.k
        test_set, X_test, y_test, y_conflict_test, G =  dl.test_set, dl.X_test, dl.y_test, dl.y_conflict_test, dl.G
        y_cr = y_test * y_conflict_test

        cn_score = self.models['common neighbors'].eval(test_set, y_cr, k=k)
        jc_score = self.models['jaccard'].eval(test_set, y_cr, k=k)
        aa_score = self.models['adamic adar'].eval(test_set, y_cr, k=k)
        ra_score = self.models['resource allocation'].eval(test_set, y_cr, k=k)
        pa_score = self.models['preferential attachment'].eval(test_set, y_cr, k=k)
        ppr_score = self.models['personalized pagerank'].eval(test_set, y_cr, k=k)
        katz_score = self.models['katz'].eval(test_set, y_cr, k=k)
        mlp_score = self.models['mlp'].eval(X_test, y_cr, k=k)
        self.logger.info('Common Neighbors: {:.4f}, Jaccard: {:.4f}, Adamic Adar:{:.4f}, Resource Allocation: {:.4f}, Preferential Attachment: {:.4f}, Personalized PageRank: {:.4f}, Katz: {:.4f}, MLP: {:.4f}'.format(cn_score, jc_score, aa_score, ra_score, pa_score, ppr_score, katz_score, mlp_score))

    def _eval_model(self, model, y_test, y_conflict_test, G=None):
        model.score()