import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch_geometric.nn import GCNConv
from .diffusion import Diffusion
from .model import MLP

def mlp(sizes, activation, dropout_flag=False, dropout=0.5, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        if dropout_flag:
            layers += [nn.Linear(sizes[j], sizes[j+1]), act(), nn.Dropout(dropout)]
        else:
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class GCN(torch.nn.Module):
    def __init__(self, node_features, single_emb):      
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_features, single_emb)
        self.relu = torch.nn.ReLU()
 
 
    def forward(self, x, edge_index):
            
        x = self.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, training=self.training)
        if (len(x.size()) == 3):
            state_emb = torch.flatten(x, 1)
        else:
            state_emb = torch.flatten(x)   
 
        return state_emb
    
class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, feature_matrix, edge_index, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(feature_matrix, edge_index)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi
    
class GCNCategoricalActor(Actor):

    def __init__(self, feature_num, node_num, gcn, hidden_sizes, act_num, activation):
        super().__init__()
        self.GCN = gcn
        
        self.model = MLP(
        state_dim=feature_num*node_num,
        action_dim=act_num,
        activation=nn.ReLU
    )
        
        self.logits_net = Diffusion(
        state_dim=feature_num*node_num,
        action_dim=act_num,
        model=self.model,
        max_action=20.,   
        beta_schedule='vp',
        n_timesteps=3,
        bc_coef = 0
    )

    # logits is the log probability, log_p = ln(p)
    def _distribution(self, feature_matrix, edge_index):
        obs_emb = self.GCN(feature_matrix, edge_index)
        logits = self.logits_net(obs_emb)
        return Categorical(logits=logits)

    def get_logits(self, feature_matrix, edge_index):
        obs_emb = self.GCN(feature_matrix, edge_index)
        logits = self.logits_net(obs_emb)
        return logits

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)
    
class GCNCritic(nn.Module):

    def __init__(self, feature_num, node_num, gcn, hidden_sizes, activation):
        super().__init__()
        self.GCN = gcn
        self.v_net = mlp([feature_num*node_num] + list(hidden_sizes) + [1], activation)

    def forward(self, feature_matrix, edge_index):
        return torch.squeeze(self.v_net(self.GCN(feature_matrix, edge_index)), -1) # Critical to ensure v has right shape.
    
