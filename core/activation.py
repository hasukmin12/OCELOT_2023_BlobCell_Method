import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

class NormReLU(Module):
    def __init__(self):
        super(NormReLU, self).__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        val_relu = F.relu(input)
        result = val_relu / torch.max(val_relu) if torch.max(val_relu) != 0 else 0
        return result