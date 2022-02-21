import torch.nn as nn
import torch

class Linear(nn.Module):
    def __init__(self, in_size, out_size, bias = True):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size, bias = bias)
        torch.nn.init.xavier_normal_(self.linear.weight)
    
    def forward(self, x):
        return self.linear(x)