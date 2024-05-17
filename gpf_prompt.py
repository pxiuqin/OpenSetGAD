import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Any

from graph_prompt import center_embedding

# 用于GPF框架用
def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)

class SimplePrompt(nn.Module):
    def __init__(self, in_channels: int):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, 302))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: Tensor):
        # print(self.global_emb)
        return x + self.global_emb


class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p
    

class CenterEmbedding(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(CenterEmbedding, self).__init__()
        self.layer = nn.Linear(2* in_channels, in_channels)
        self.gpf = GPFplusAtt(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, labels):
        c_embedding = center_embedding(x, labels)
        c_embedding_prompt = center_embedding(self.gpf.add(x), labels)

        center_outs = self.layer(torch.cat((c_embedding, c_embedding_prompt), dim=1))

        return center_outs

    
class GNN(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, pre_model, JK="last", drop_ratio=0, gnn_type="gat"):
        super(GNN, self).__init__()
        self.drop_ratio = drop_ratio
        self.JK = JK

        ###List of MLPs
        self.gnn = pre_model

    def forward(self, blocks, prompt):
        if prompt is not None:
            blocks[0].srcdata['features'] = prompt.add(blocks[0].srcdata['features'])

        self.gnn(blocks)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def   __init__(self, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin",
                 head_layer=1, final_linear = True):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.final_linear = final_linear
        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.ModuleList()
            for i in range(head_layer - 1):
                self.graph_pred_linear.append(torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim,
                                                                self.mult * (self.num_layer + 1) * self.emb_dim))
                self.graph_pred_linear.append(torch.nn.ReLU())
            self.graph_pred_linear.append(
                torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks))

        else:
            self.graph_pred_linear = torch.nn.ModuleList()
            for i in range(head_layer - 1):
                self.graph_pred_linear.append(
                    torch.nn.Linear(self.mult * self.emb_dim, self.mult * self.emb_dim))
                self.graph_pred_linear[-1].reset_parameters()
                self.graph_pred_linear.append(torch.nn.ReLU())

            self.graph_pred_linear.append(torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks))
            self.graph_pred_linear[-1].reset_parameters()

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file, map_location='cpu'))

    def forward(self, *argv):
        if len(argv) == 5:
            x, edge_index, edge_attr, batch, prompt = argv[0], argv[1], argv[2], argv[3], argv[4]
        elif len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        if len(argv) == 5:
            node_representation = self.gnn(x, edge_index, edge_attr, prompt)
        else:
            node_representation = self.gnn(x, edge_index, edge_attr)

        emb = self.pool(node_representation, batch)

        if self.final_linear:
            for i in range(len(self.graph_pred_linear)):
                emb = self.graph_pred_linear[i](emb)

        # return self.graph_pred_linear(self.pool(node_representation, batch))
        return emb


def embeddings(model, blocks):
    blocks[0].srcdata['features'] = model.add(blocks[0].srcdata['features'])
    return blocks
