import torch
import torch.nn as nn
import math

"""Impletation with the paper Attention is all you Need"""
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    
    d_K = Q.shape[-1]
    
    scores = Q @ K.transpose(-1, -2) / math.sqrt(d_K)
    
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    
    scores = torch.softmax(scores, dim = -1)
    
    attn = scores @ V
    
    return attn
    