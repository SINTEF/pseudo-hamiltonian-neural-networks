
import torch


def to_tensor(x, ttype=torch.float32):
    if x is None:
        return x
    elif not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=ttype)
    else:
        return x
