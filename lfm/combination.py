import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class LinearCombination(nn.Module):
    beta: torch.Tensor

    def __init__(self, model_beta):
        super(LinearCombination, self).__init__()
        if model_beta == -1:
            # Sampling beta uniformly from [0.45, 0.55] as starting point
            # self.beta = Parameter(0.1 * torch.rand([1]) + torch.tensor([0.45]))
            # 0.5 as starting point
            self.beta = Parameter(torch.tensor(0.5))
        else:
            self.beta = model_beta

    def forward(self, x1, x2):
        return self.beta * x1 + (1 - self.beta) * x2