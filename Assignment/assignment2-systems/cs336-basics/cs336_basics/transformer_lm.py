import torch
import torch.nn as nn

from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock


class Transformer_LM(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        context_length: int, 
        num_layers: int, 
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.context_length = context_length
        self.num_layers = num_layers
        
        
        # 输入 token embedding
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        
        # 多层 Transformer block
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(self.num_layers)
            ]
        )
        
        # 最终输出前的 RMSNorm
        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        
        # LM 输出头
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )
        
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.shape[-1]
        
        if seq_len > self.context_length:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds context length {self.context_length}"
            )
            
        # (batch_size, seq_len, d_model)
        hidden_states = self.token_embeddings(token_ids)
        
        base_positions = torch.arange(
            seq_len,
            device = token_ids.device,
            dtype = torch.long
        )
        token_positions = base_positions.view(
            *([1] * (token_ids.ndim - 1)),
            seq_len,
        ).expand(*token_ids.shape[:-1], seq_len)
        
        # 依次通过每一层 block
        for layer in self.layers:
            hidden_states = layer(hidden_states, token_positions=token_positions) # 易错点: 用layer去迭代而不是 num_layers
        
        
            
        # final norm
        hidden_states = self.ln_final(hidden_states)
        
        # 输出
        logits = self.lm_head(hidden_states)
        return logits
