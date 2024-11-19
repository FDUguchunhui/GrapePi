from functools import partial

import torch
import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_act


@register_act('none')
class NoneAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


