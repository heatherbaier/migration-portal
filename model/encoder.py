import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
import json

class Encoder(nn.Module):
    
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda

        self.weight1 = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform(self.weight1)
        
        
    def forward(self, nodes):
        
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
                
        nodes_int = torch.tensor([int(i) for i in nodes])
        
        # Sends the nodeID's and a list fo lsits containing theneighbors of those nodes to the aggregator class
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[str(node)] for node in nodes], 
                self.num_sample)
        
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = torch.index_select(torch.tensor(self.features, dtype = torch.float32), 0, nodes_int)
                
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        combined = F.relu(self.weight1.mm(combined.t()))

        return combined
