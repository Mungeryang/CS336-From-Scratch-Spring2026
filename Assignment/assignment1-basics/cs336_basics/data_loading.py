import numpy as np
import torch


def get_batch(dataset, batch_size, context_length, device):
    
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start,size = batch_size)
    
    x = np.stack([dataset[s : s + context_length] for s in starts])
    y = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts])
    
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x,y