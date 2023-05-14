import os.path

from utils import *
from dataloader import *
from models.Optimal import Optimal
from models.CommonNeighbors import CommonNeighbors
from models.Jaccard import Jaccard
from models.ResourceAllocation import ResourceAllocation
from models.AdamicAdar import AdamicAdar
from models.PreferentialAttachment import PreferentialAttachment
from models.LR import LR
from models.PersonalizedPageRank import PersonalizedPageRank
from models.Katz import Katz
from models.GCN import GCN
from models.GraphTransformer import GraphTransformer
from models.RGCN import RGCN
from models.SuperGAT import SuperGAT



def saveResult(key, rec):
    fname = 'result_reddit.pkl'
    dic = {}
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            dic = pickle.load(f)
    dic[key] = rec

    with open(fname, 'wb') as f:
        pickle.dump(dic, f)


if __name__ == '__main__':
    # set up and load data
    args, logger = setUp()  # set up args, logger, random seed
    dataloader = DataLoader(args, logger)

    # compute optimal recommendation
    model_opt = Optimal(dataloader, args, mask=dataloader.reserved_adj_mask, logger=logger)
    rec_opt = model_opt.recommend()
    rec_opt.printResults(logger)

    # compute model recommendation
    model_list = [SuperGAT, GraphTransformer, GCN, RGCN,
                  CommonNeighbors, Katz, Jaccard, AdamicAdar,
                  PreferentialAttachment, ResourceAllocation,
                   PersonalizedPageRank, LR, Optimal]
    for Model in model_list:
        print('=' * 30 + Model.name + '=' * 30)
        if Model == Optimal:
            model = Model(dataloader, args, mask=dataloader.target_adj_mask, logger=logger)
        else:
            model = Model(dataloader, args, logger)
        rec = model.recommend()
        rec.printResults(logger)
        ca = rec_opt.computeConflictAwareness(rec)
        rec.ca = ca
        logger.info('[main] {} CA: {:.6f}, Precision@{}: {:.5f}, Recall: {:.5f}\n'.format(Model.name,
                                                                                         ca,
                                                                                         args.precisionk,
                                                                                         rec.precision_at_k,
                                                                                         rec.recall))
        saveResult(key=(args.dataset, model.name, int(args.beta), int(args.n), int(args.seed)), rec=rec)
    ###



