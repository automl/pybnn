import torch
import torch.nn as nn
import numpy as np


class AppendLayer(nn.Module):
    def __init__(self, noise=1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_var = nn.Parameter(torch.DoubleTensor(1, 1))

        nn.init.constant_(self.log_var, val=np.log(noise))

    def forward(self, x):
        return torch.cat((x, self.log_var * torch.ones_like(x)), dim=1)
