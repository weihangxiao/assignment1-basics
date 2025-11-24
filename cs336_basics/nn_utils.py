import torch

def softmax(x: torch.Tensor, dim: int):
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
    

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    logits = logits - logits.max(dim=-1, keepdim=True).values
    log_probs = logits - torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))
    
    nll = -log_probs[torch.arange(targets.size(0)), targets]
    return nll.mean()