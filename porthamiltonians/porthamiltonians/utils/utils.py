
import torch

__all__ = ['to_tensor']


def to_tensor(x, ttype=torch.float32):
    """
    Converts the input to a torch tensor if the input is not None.

    Parameters
    ----------
    x : listlike or None
    ttype : torch type, default torch.float32

    Returns
    -------
    torch.tensor or None
        Return converted list/array/tensor unless *x* is None,
        in which case it returns None.

    """
    if x is None:
        return x
    elif not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=ttype)
    else:
        return x
