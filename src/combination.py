import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LinearCombination(nn.Module):
    def __init__(self, model_beta):
        super(LinearCombination, self).__init__()
        self.beta = Parameter(torch.tensor(model_beta))

    def forward(self, x1, x2):
        return self.beta * x1 + (1 - self.beta) * x2
