import torch
import torch.nn as nn

from cs336_basics.linear import Linear

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x) 

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w1 = Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w2 = Linear(d_ff,d_model,device=device,dtype=dtype)
        self.w3 = Linear(d_model,d_ff,device=device,dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x1 = self.w1(x)
        x3 = self.w3(x)
        
        glu = silu(x1) * x3
        
        out = self.w2(glu)
        return out