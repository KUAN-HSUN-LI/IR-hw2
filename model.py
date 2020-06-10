import torch.nn as nn
import torch
import torch.nn.functional as F


class BPR(nn.Module):
    def __init__(self, user_size, item_size, hidden_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn([user_size, hidden_dim]).uniform_(0, 1))
        self.H = nn.Parameter(torch.randn([item_size, hidden_dim]).uniform_(0, 1))

    def forward(self, u, i, j):
        u = self.W[u, :]
        i = self.H[i, :]
        j = self.H[j, :]

        x_ui = torch.mul(u, i).sum(dim=1)
        x_uj = torch.mul(u, j).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = F.logsigmoid(x_uij).sum()
        return -log_prob
