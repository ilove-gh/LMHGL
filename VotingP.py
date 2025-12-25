import torch
import torch.nn as nn

class RobustGatedVotingV2(nn.Module):
    def __init__(self, input_dim, num_experts=3,lamba=0.2,enta=0.7):
        super().__init__()
        self.lamba=lamba
        self.enta=enta

        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, num_experts),
            nn.Tanh()
        )

        self.classify = nn.ModuleList([
            self._build_expert(input_dim, leaky_slope=0.1),
            self._build_expert(input_dim, leaky_slope=0.15),
            self._build_expert(input_dim, leaky_slope=0.2)
        ])
        self._init_weights()

    def _build_expert(self, input_dim, leaky_slope):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(input_dim, 2),
            nn.Softmax(dim=-1)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, batch_left, batch_right,batch_ucn, batch_global):
        batch_left = batch_left - self.lamba * batch_global.detach()
        batch_right = batch_right - self.lamba * batch_global.detach()
        batch_global = self.enta * batch_global

        gate_logits = self.gate(batch_global)
        gate_weights = torch.softmax(gate_logits - gate_logits.mean(dim=1, keepdim=True), dim=1)  # 添加 keepdim=True

        expert_outputs = torch.stack([
            self.classify[0](batch_left),
            self.classify[1](batch_right),
            self.classify[2](batch_global)
        ], dim=1)

        return (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)