import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class LinearCombination(nn.Module):
    beta: torch.Tensor

    def __init__(self, beta):
        super(LinearCombination, self).__init__()
        # Sampling beta uniformly from [0.45, 0.55]
        if beta == -1:
            b = 0.1 * torch.rand([1]) + torch.tensor([0.45])
        else:
            b = torch.tensor(beta)
        self.beta = Parameter(b)

    def forward(self, x1, x2):
        return self.beta * x1 + (1 - self.beta) * x2
