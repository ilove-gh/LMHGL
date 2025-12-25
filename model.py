import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCN2Conv
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor, Size
from torch import Tensor
from typing import Union, Tuple
from VotingP import RobustGatedVotingV2

class GCNIIWithFeatureFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, alpha, theta, dropout):
        super(GCNIIWithFeatureFusion, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.alpha = alpha
        self.theta = theta

        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCN2Conv(hidden_dim, alpha=alpha, theta=theta, layer=i + 1,normalize=False))

        self.fusion_layer = nn.Linear(hidden_dim * num_layers, hidden_dim)

    def forward(self, x, edge_index,edge_weight: OptTensor = None, batch: OptTensor = None):
        x = self.lin1(x)
        x_0 = x

        layer_features = []
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, x_0, edge_index=edge_index, edge_weight=edge_weight)
            layer_features.append(x)

        # 特征融合
        x = torch.cat(layer_features, dim=-1)
        x = self.fusion_layer(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

class WeightedGINConv(MessagePassing):
    def __init__(self, nn: torch.nn.Module, eps: float = 0., train_eps: bool = True, **kwargs
    ):
        super().__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))  # 修复：使用 torch.nn.Parameter
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]], edge_index: Adj, edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        # 处理边权重
        if edge_weight is not None:
            edge_weight = edge_weight.view(-1, 1)  # 确保维度匹配 [num_edges, 1]

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j  # 已广播


class WeightedGIN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3, dropout: float = 0.5, use_batchnorm: bool = True):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            mlp = self._build_mlp(
                in_dim if i == 0 else hidden_dim,
                out_dim if is_last else hidden_dim,
                use_batchnorm and not is_last
            )
            conv = WeightedGINConv(
                mlp,
                eps=0.5,
                train_eps=True
            )
            self.convs.append(conv)

    def _build_mlp(self, in_dim: int, out_dim: int, use_batchnorm: bool) -> nn.Sequential:
        layers = []
        layers.append(nn.Linear(in_dim, out_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None, batch: OptTensor = None) -> Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # 最后一层不加 dropout
        x = self.convs[-1](x, edge_index, edge_weight)
        return x


class LMHGL(nn.Module):
    def __init__(self, input_dim, hidden_dim,out_dim, num_layers, alpha, theta, dropout, device,lamba=0.2,enta=0.7):
        super(LMHGL, self).__init__()
        self.batch_R_brain = GCNIIWithFeatureFusion(input_dim, hidden_dim, num_layers, alpha, theta, dropout)
        self.batch_L_brain = GCNIIWithFeatureFusion(input_dim, hidden_dim, num_layers, alpha, theta, dropout)
        self.batch_ucn_brain = WeightedGIN(input_dim, hidden_dim,out_dim=hidden_dim, dropout = dropout, use_batchnorm=True, num_layers=2)
        self.hidden_dim = hidden_dim
        self.classify = nn.ModuleList([
            # Left Brain-0
            nn.Sequential(nn.Linear(hidden_dim, out_dim),nn.LogSoftmax(dim=-1)),
            # Right Brain-1
            nn.Sequential(nn.Linear(hidden_dim, out_dim),nn.LogSoftmax(dim=-1)),
            # UCN Brain-2
            nn.Sequential(nn.Linear(hidden_dim, out_dim),nn.LogSoftmax(dim=-1)),
            # Grobal Brain-3
            nn.Sequential(nn.Linear(hidden_dim, out_dim),nn.LogSoftmax(dim=-1)),
        ])

        self.attention = Attention(hidden_dim)
        self.Voting = RobustGatedVotingV2(hidden_dim,lamba=lamba, enta=enta)
        self.device = device

    def forward(self, left_brain, right_brain, ucn_brain):
        t_total = time.time()
        batch_ucn_brain = self.batch_ucn_brain(x = ucn_brain.x, edge_index = ucn_brain.edge_index,edge_weight = ucn_brain.edge_weight, batch=ucn_brain.batch)
        batch_left_brain = self.batch_L_brain(x = left_brain.x, edge_index = left_brain.edge_index,edge_weight = left_brain.edge_weight, batch = left_brain.batch)
        batch_right_brain = self.batch_R_brain(x = right_brain.x, edge_index = right_brain.edge_index,edge_weight = right_brain.edge_weight,batch = right_brain.batch)

        batch_size, num_nodes, dim = self.get_batch_shape(left_brain)

        emb_left = batch_left_brain.view(batch_size, num_nodes, self.hidden_dim)
        emb_left = emb_left.mean(dim=1)

        emb_right = batch_right_brain.view(batch_size, num_nodes, self.hidden_dim)
        emb_right = emb_right.mean(dim=1)

        emb_ucn = batch_ucn_brain.view(batch_size, num_nodes*2, self.hidden_dim)
        emb_ucn = emb_ucn.mean(dim=1)

        emb_grobal = torch.stack([emb_left, emb_right, emb_ucn], dim=1)
        emb_grobal, att = self.attention(emb_grobal)
        brain_class = self.Voting(emb_left, emb_right, emb_ucn,emb_grobal)
        return brain_class


    def soft_voting(self, batch_left_brain, batch_right_brain,batch_grobal_brain):
        Left_class = self.classify[0](batch_left_brain)
        Right_class = self.classify[1](batch_right_brain)
        Grobal_class = self.classify[3](batch_grobal_brain)
        all_outputs = torch.stack([Left_class, Right_class, Grobal_class], dim=0)
        avg_prob = torch.mean(all_outputs, dim=0)
        return avg_prob

    def get_batch_shape(self, batch_graph):
        batch_size = batch_graph.num_graphs
        num_nodes = batch_graph.num_nodes/batch_size
        dim = batch_graph.x.shape[1]

        return int(batch_size), int(num_nodes), int(dim)

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        # 初始化
        for layer in self.project:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.01)  # 小尺度初始化
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, z):
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6)  # L2归一化
        w = self.project(z)
        w = w / torch.sqrt(torch.tensor(z.size(-1), dtype=torch.float32))  # 缩放
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class HeatDiffusion(nn.Module):
    def __init__(self, t: float = 5.0,top_k: int = 16, threshold: float = None):
        super(HeatDiffusion, self).__init__()
        self.t = t
        self.top_k = top_k
        self.threshold = threshold

    def process(self,adj):
        heat_matrix = torch.matrix_exp(-self.t * adj)
        if self.top_k:
            heat_matrix = self.get_top_k_matrix(heat_matrix, top_k=self.top_k)
        elif self.threshold:
            heat_matrix = self.get_clipped_matrix(heat_matrix, threshold=self.threshold)
        heat_matrix = self.to_sys_matrix(heat_matrix)
        heat_matrix = self.random_walk_normalize_with_self_loops(heat_matrix)
        return heat_matrix

    def get_clipped_matrix(self, matrix: Tensor, threshold: float = 0.01) -> Tensor:
        mask = matrix >= threshold
        thresholded_matrix = torch.where(mask, matrix, torch.zeros_like(matrix))
        return thresholded_matrix

    def get_top_k_matrix(self, matrix: Tensor, top_k: int = 128) -> Tensor:
        batch_size, num_nodes, _ = matrix.size()
        top_k_values, top_k_indices = torch.topk(matrix, k=top_k, dim=-1)
        mask = torch.zeros_like(matrix, dtype=torch.bool)
        for i in range(num_nodes):
            mask[:, i, top_k_indices[:, i]] = True
            mask[:, top_k_indices[:, i], i] = True
        top_k_matrix = torch.where(mask, matrix, torch.zeros_like(matrix))
        return top_k_matrix

    def random_walk_normalize_with_self_loops(self, A):
        batch_size, n, _ = A.shape
        I = torch.eye(n, device=A.device).unsqueeze(0)
        A_with_loops = A + I
        D = torch.sum(A_with_loops, dim=2, keepdim=True)
        D_inv = 1.0 / D
        A_normalized = D_inv * A_with_loops
        return A_normalized

    def to_sys_matrix(self, adj):
        adj = (adj.transpose(1, 2) + adj)/2
        return adj