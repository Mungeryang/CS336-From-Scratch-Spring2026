from turtle import forward
import torch
from torch._prims_common import dtype_or_default
import torch.nn as nn

from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.multihead_self_attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=True,
            device=device,
            dtype=dtype,
        )
        
        self.ln2 = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model,d_ff=d_ff,device=device,dtype=dtype)
        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x