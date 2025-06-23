import torch
from torch import nn


"""
Sinusoidal positional encoding module for transformer models in CMIP.
"""

class PostionalEncoding(nn.Module):
    """
    Compute sinusoidal positional encoding for input sequences.
    """

    def __init__(self, d_model: int, max_len: int):
        """
        Constructor for sinusoidal encoding class.
        Args:
            d_model: Dimension of model/embedding
            max_len: Maximum sequence length
        """
        super(PostionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Get positional encoding for a batch of sequences.
        Args:
            batch_size: Batch size (unused, for API compatibility)
            seq_len: Sequence length
        Returns:
            Positional encoding tensor of shape (seq_len, d_model)
        """
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = batch_size, seq_len
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
