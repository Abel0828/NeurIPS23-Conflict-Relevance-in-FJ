from fastnode2vec import Graph, Node2Vec
from itertools import product
from recommendation import *
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GPSConv, GINEConv
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import os


class GraphTransformer(torch.nn.Module):
    name = 'GraphTransformer'

    def __init__(self, dataloader, args, logger):
        super(GraphTransformer, self).__init__()
        self.A = dataloader.A
        self.G = dataloader.G
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        # edge_index is the original graph
        # all_edge_index is original graph + negative edges
        # target_edge_index is reserved positive edges + negatively sampled edges
        edge_index = torch.from_numpy(np.array(self.G.edges).T)
        x = torch.eye(self.G.number_of_nodes())
        # x = torch.from_numpy(self.getNodeEmbeddings(self.G, self.args.node_emb_dim)).float()
        all_edge_index, all_y = self.get_all_edge_index_y(edge_index, self.G.number_of_nodes())
        target_edge_index = torch.from_numpy(self.dataloader.target_edges.T)
        self.data = Data(x=x, edge_index=edge_index,
                         all_edge_index=all_edge_index, all_y=all_y,
                         target_edge_index=target_edge_index)

        # define GCN structure
        channels = 32
        self.emb = nn.Linear(self.data.x.shape[-1], channels)
        self.conv1 = GPSConv(channels, conv=GINEConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.Linear(channels, channels)),train_eps=True, edge_dim=1), heads=4)
        self.conv2 = GPSConv(channels, conv=GINEConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.Linear(channels, channels)),train_eps=True, edge_dim=1), heads=4)
        self.feedforward = nn.Sequential(torch.nn.Linear(2*channels, 16), nn.ReLU(), nn.Linear(16, 2))
        self.to(dataloader.device)
        self.data.to(dataloader.device)
        self.trainRecommender()

    def get_all_edge_index_y(self, edge_index, num_nodes):
        num_edges = edge_index.shape[-1]
        neg_edge_index = torch.from_numpy(np.random.randint(0, num_nodes, size=(2, num_edges)))
        all_edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)
        all_y = torch.cat([torch.ones(num_edges), torch.zeros(num_edges)]).long()
        return all_edge_index, all_y

    def trainRecommender(self):
        # Define the loss function and optimizer
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        for epoch in tqdm(range(self.args.epochs)):
            embeddings = self.forward(self.data.x, self.data.edge_index)
            edge_logits = self.get_edge_logit(embeddings, self.data.all_edge_index)
            loss = criterion(edge_logits, self.data.all_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # with torch.no_grad():
            #     embeddings = self.forward(self.data.x, self.data.edge_index)
            #     edge_logits = self.get_edge_logit(embeddings, self.data.target_edge_index)
            #     edge_logits = np.exp(edge_logits.numpy()[:, 1])

                # edge_labels = self.dataloader.target_edges_labels
                # ranked_list = sorted(list(zip(edge_logits, edge_labels)), reverse=True)
                # k = self.args.precisionk
                # precision_at_k = sum([e[0] * e[1] for e in ranked_list[:k]]) / sum([e[0] for e in ranked_list[:k]])
                # recall = sum([e[0] * e[1] for e in ranked_list]) / edge_labels.sum()
            # print('epoch {}, loss: {:.5f}, precision@k: {:.5f}, recall:{:.5f}'.format(epoch, loss.item(),
            #                                                                           precision_at_k, recall))

    def forward(self, x, edge_index):
        edge_attr = torch.zeros((edge_index.size(1), 1), dtype=torch.float32)
        x = F.relu(self.conv1(self.emb(x), edge_index, edge_attr=edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x

    def get_edge_logit(self, x, edge_index):
        u_emb = x[edge_index[0]]
        v_emb = x[edge_index[1]]
        uv_emb = torch.cat([u_emb, v_emb], dim=-1)
        edge_logit = F.log_softmax(self.feedforward(uv_emb), dim=-1)
        return edge_logit

    def recommend(self):
        with torch.no_grad():
            embeddings = self.forward(self.data.x, self.data.edge_index)
            edge_logits = self.get_edge_logit(embeddings, self.data.target_edge_index)
            edge_logits = np.exp(edge_logits.numpy()[:, 1])

        target_edges = self.dataloader.target_edges
        Aplus = np.zeros_like(self.A, dtype=np.float64)
        Aplus[target_edges[:, 0], target_edges[:, 1]] = edge_logits
        Aplus -= np.diag(np.diagonal(Aplus))
        Aplus *= self.dataloader.target_adj_mask
        Aplus *= (2 * self.args.beta / Aplus.sum())
        if Aplus.max() > self.args.d:
            Aplus = clip(Aplus, self.args.d)
        return Recommendation(Aplus, new_conflict=None, dataloader=self.dataloader, edge_logits=edge_logits)


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

# todo:
# 1. precison and use precision to find the best training hyperparameter
# 2. GAT
# 3. SOTA link recommendation models