import torch
import torch.nn.functional as F


def _pad_mask(mask: torch.Tensor) -> tuple:
    """
    Pad the mask tensor to a multiple of 8 along the last dimension.
    Args:
        mask: Input mask tensor
    Returns:
        Tuple of (padded mask, number of 8-bit groups)
    """
    # By taking -size % 8, we get 0 if exactly divisible by 8
    # and required padding otherwise (i.e. -1 % 8 = 7 pad)
    pad = -mask.size(-1) % 8
    if pad != 0:
        mask = F.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8


def _mask_bool2byte(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a boolean (uint8) mask to a byte mask.
    Args:
        mask: Boolean mask tensor (uint8)
    Returns:
        Byte mask tensor
    """
    assert mask.dtype == torch.uint8
    # assert (mask <= 1).all()  # Precondition, disabled for efficiency
    mask, d = _pad_mask(mask)
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(
        -1, dtype=torch.uint8
    )


def _mask_byte2long(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a byte mask to a long mask (for efficient storage).
    Args:
        mask: Byte mask tensor (uint8)
    Returns:
        Long mask tensor
    """
    assert mask.dtype == torch.uint8
    mask, d = _pad_mask(mask)
    # Note this corresponds to a temporary factor 8
    # memory overhead by converting to long before summing
    # Alternatively, aggregate using for loop
    return (
        mask.view(*mask.size()[:-1], d, 8).long()
        << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)
    ).sum(-1)


def mask_bool2long(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a boolean (uint8) mask to a long mask.
    Args:
        mask: Boolean mask tensor (uint8)
    Returns:
        Long mask tensor
    """
    assert mask.dtype == torch.uint8
    return _mask_byte2long(_mask_bool2byte(mask))


def _mask_long2byte(mask: torch.Tensor, n: int = None) -> torch.Tensor:
    """
    Convert a long mask back to a byte mask.
    Args:
        mask: Long mask tensor (int64)
        n: Optional number of bits
    Returns:
        Byte mask tensor
    """
    if n is None:
        n = 8 * mask.size(-1)
    return (
        (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n]
        .to(torch.uint8)
        .view(*mask.size()[:-1], -1)[..., :n]
    )


def _mask_byte2bool(mask: torch.Tensor, n: int = None) -> torch.Tensor:
    """
    Convert a byte mask back to a boolean mask.
    Args:
        mask: Byte mask tensor (uint8)
        n: Optional number of bits
    Returns:
        Boolean mask tensor
    """
    if n is None:
        n = 8 * mask.size(-1)
    return (
        mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)
    ).view(*mask.size()[:-1], -1)[..., :n] > 0


def mask_long2bool(mask: torch.Tensor, n: int = None) -> torch.Tensor:
    """
    Convert a long mask back to a boolean mask.
    Args:
        mask: Long mask tensor (int64)
        n: Optional number of bits
    Returns:
        Boolean mask tensor
    """
    assert mask.dtype == torch.int64
    return _mask_byte2bool(_mask_long2byte(mask), n=n)


def mask_long_scatter(mask: torch.Tensor, values: torch.Tensor, check_unset: bool = True) -> torch.Tensor:
    """
    Set values in a long mask at specified positions.
    Args:
        mask: Long mask tensor (int64)
        values: Indices to set
        check_unset: Whether to check that bits are not already set
    Returns:
        Updated long mask tensor
    """
    """
    Sets values in mask in dimension -1 with arbitrary batch dimensions
    If values contains -1, nothing is set
    Note: does not work for setting multiple values at once (like normal scatter)
    """
    assert mask.size()[:-1] == values.size()
    rng = torch.arange(mask.size(-1), out=mask.new())
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))
