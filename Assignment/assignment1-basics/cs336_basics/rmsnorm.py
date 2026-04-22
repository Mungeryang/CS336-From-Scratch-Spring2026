import torch
import torch.nn as nn
import math


"""
    Transformer Block包括两个主要的sub-layers: multi-head self-attn mechanism 和 position-wise feed-forward netword.
    
    模型在两个子层的输出端均采用了残差连接机制，随后进行归一化处理
    
    post-norm: 层归一化作用于子层输出
    
    pre-norm: 层归一化作用于子层输入(提升训练稳定性)
    
    pre-norm的核心思想在于从 输入embedding 到 最终输出过程中会形成一条未经过任务归一化处理的清晰“残差流”, 可以改善梯度流动。
    
"""

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(self.d_model,device=device,dtype=dtype)
        )
        
        #nn.init.trunc_normal_(self.weight)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        
        
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) 
        result = x / rms * self.weight
        
        return result.to(in_dtype)