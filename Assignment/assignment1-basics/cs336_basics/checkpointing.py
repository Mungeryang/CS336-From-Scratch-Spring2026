import torch


def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint["iteration"]

