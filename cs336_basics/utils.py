import torch 
import os
from typing import IO, Any, BinaryIO


def save_checkpoint(model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }, out)
    return out
    

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
) -> int:
    checkpoint = torch.load(src, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]



def _model_device_and_compile(model):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else :
        device = torch.device('cpu')
    model = model.to(device)

    if device.type == 'mps':
        model = torch.compile(model, backend='aot_eager')
    else :
        model = torch.compile(model)

    return model, device