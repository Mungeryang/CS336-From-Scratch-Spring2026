from numpy import dtype
import torch
import torch.nn as nn
import math

from cs336_basics.linear import Linear
from cs336_basics.rope import Rope
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int, # Dimensionality of the Transformer block inputs.
        num_heads: int, # Number of heads to use in multi-head self-attention.
        max_seq_len=None, 
        theta=None, 
        use_rope=False,
        device=None, 
        dtype=None,
        
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        if self.use_rope:
            self.rope = Rope(
                theta=theta,
                d_k=self.head_dim,
                max_seq_len=max_seq_len,
                device=device
            )
        else:
            self.rope = None
        
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        
        *batch_dims, seq_len, _ = x.shape
        x = x.reshape(*batch_dims, seq_len, self.num_heads, self.head_dim)
        x = x.transpose(-3, -2)
        return x
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        
        *batch_dims, _, seq_len, _ = x.shape
        x = x.transpose(-3, -2)
        x = x.reshape(*batch_dims, seq_len, self.d_model)
        
        return x
    
    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        *batch_dims, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        if self.use_rope:
            if token_positions is not None:
                base_positions = torch.arange(
                    seq_len,
                    device=x.device,
                    dtype=torch.long,
                )
                
                token_positions = base_positions.view(
                    *([1] * len(batch_dims)),
                    seq_len,
                ).expand(*batch_dims, seq_len)
            else:
                token_positions = token_positions.to(device=x.device, dtype=torch.long)
            
            rope_positions = token_positions.unsqueeze(-2)
            
            q = self.rope(q, rope_positions)
            k = self.rope(k, rope_positions)
        
        causal_mask = self._build_causal_mask(seq_len, x.device)
        
        attn_out = scaled_dot_product_attention(
            Q=q,
            K=k,
            V=v,
            mask=causal_mask,
        )
        
        attn_out = self._merge_heads(attn_out)
        out = self.output_proj(attn_out)
        
        return out
                
        
        
        
