import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    """核心: 1.维护一个weight参数 2.forward 过程计算矩阵乘法"""
    def __init__(self, in_features, out_features, device=None, dtype=None):
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(
            torch.empty(out_features,in_features,device=device,dtype=dtype)
        )
        
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean = 0.0, std=std, a=-3 * std, b=3 * std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.transpose(-1,-2)