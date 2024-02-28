import torch
from torch import nn

class Squeeze(nn.Module):
    def __init__(self, *args):
        super(Squeeze, self).__init__()
        self.args = args
    def forward(self, x:torch.Tensor):
        return x.squeeze(*self.args)

class Unsqueeze(nn.Module):
    def __init__(self, *args):
        super(Unsqueeze, self).__init__()
        self.args = args
    def forward(self, x:torch.Tensor):
        return x.unsqueeze(*self.args)

class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args
    def forward(self, x:torch.Tensor):
        return x.permute(self.args)
