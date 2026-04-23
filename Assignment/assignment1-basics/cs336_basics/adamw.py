from collections.abc import Callable,Iterable
from typing import Optional
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
    
    # p.data -> sita
    # weight_decay -> lanbuda
    # lr -> alpha
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                # 旧状态
                t = state["t"]
                m = state["m"]
                v = state["v"]
                
                # 更新 step 计数
                t = t + 1
                
                # 一阶矩估计
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                
                # 二阶矩估计
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                
                p.data -= lr * weight_decay * p.data
                
                p.data -= lr * m_hat / (torch.sqrt(v_hat) + eps)

                state["t"] = t
                state["m"] = m
                state["v"] = v
                
        return loss
                    