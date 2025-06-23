import torch
from torch import Tensor, nn
import sys

sys.path.append("./")
import utils
import nets.revtorch as rv
from xformers import ops as xops
from einops import rearrange


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, hidden_states: Tensor) -> Tensor:
        """Apply RMS normalization to hidden states."""
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + self.eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass for RMSNorm."""
        return self.weight * self._norm(hidden_states.float()).type_as(hidden_states)


class MHAEncoderLayer(torch.nn.Module):
    """
    Multi-Head Attention Encoder Layer with feed-forward and normalization.
    """
    def __init__(self, embedding_dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.Wq = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multi_head_combine = torch.nn.Linear(embedding_dim, embedding_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.norm1 = torch.nn.BatchNorm1d(embedding_dim)
        self.norm2 = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        """Forward pass for the MHAEncoderLayer."""
        q = utils.make_heads(self.Wq(x), self.n_heads)
        k = utils.make_heads(self.Wk(x), self.n_heads)
        v = utils.make_heads(self.Wv(x), self.n_heads)
        x = x + self.multi_head_combine(utils.multi_head_attention(q, k, v, mask))
        x = self.norm1(x.view(-1, x.size(-1))).view(*x.size())
        x = x + self.feed_forward(x)
        x = self.norm2(x.view(-1, x.size(-1))).view(*x.size())
        return x


class MHAEncoder(torch.nn.Module):
    """
    Stacked Multi-Head Attention Encoder.
    """
    def __init__(
        self, n_layers: int, n_heads: int, embedding_dim: int, input_dim: int, add_init_projection: bool = True
    ):
        super().__init__()
        if add_init_projection or input_dim != embedding_dim:
            self.init_projection_layer = torch.nn.Linear(input_dim, embedding_dim)
        self.attn_layers = torch.nn.ModuleList(
            [
                MHAEncoderLayer(embedding_dim=embedding_dim, n_heads=n_heads)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor, mask=None) -> Tensor:
        """Forward pass for the MHAEncoder."""
    def forward(self, x, mask=None):
        if hasattr(self, "init_projection_layer"):
            x = self.init_projection_layer(x)
        for idx, layer in enumerate(self.attn_layers):
            x = layer(x, mask)
        return x


"""
RevMHAEncoder
"""


class MHABlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_node: int = None):
        super().__init__()
        self.mixing_layer_norm = nn.BatchNorm1d(hidden_size)
        self.mha = nn.MultiheadAttention(hidden_size, num_heads, bias=False)
        self.num_node = num_node
        self.n_heads = num_heads

    def forward(self, hidden_states: Tensor):

        assert hidden_states.dim() == 3
        hidden_states = self.mixing_layer_norm(hidden_states.transpose(1, 2)).transpose(
            1, 2
        )
        hidden_states_t = hidden_states.transpose(0, 1)
        mha_output = self.mha(hidden_states_t, hidden_states_t, hidden_states_t)[
            0
        ].transpose(0, 1)

        return mha_output


class MHABlock_xformer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_node: int = None):
        super().__init__()
        self.mixing_layer_norm = nn.BatchNorm1d(hidden_size)
        self.num_node = num_node
        self.n_heads = num_heads
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, hidden_states: Tensor):
        assert hidden_states.dim() == 3
        hidden_states = self.mixing_layer_norm(hidden_states.transpose(1, 2)).transpose(
            1, 2
        )

        q, k, v = rearrange(
            self.Wqkv(hidden_states),
            "b s (three h d) -> three b s h d",
            three=3,
            h=self.n_heads,
        ).unbind(dim=0)

        mha_output = xops.memory_efficient_attention(q, k, v)
        return rearrange(mha_output, "b s h d -> b s (h d)")

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous()
            .view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(
                v.size(0),
                v.size(1) if num_steps is None else num_steps,
                v.size(2),
                self.n_heads,
                -1,
            )
            .permute(
                3, 0, 1, 2, 4
            )  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )


class FFBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.feed_forward = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_layer_norm = nn.BatchNorm1d(hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states: Tensor):
        hidden_states = (
            self.output_layer_norm(hidden_states.transpose(1, 2))
            .transpose(1, 2)
            .contiguous()
        )
        intermediate_output = self.feed_forward(hidden_states)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)

        return output


class RevMHAEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        embedding_dim: int,
        input_dim: int,
        intermediate_dim: int,
        num_node: int,
        add_init_projection=True,
        use_xformer=False,
    ):
        super().__init__()
        if add_init_projection or input_dim != embedding_dim:
            self.init_projection_layer = torch.nn.Linear(input_dim, embedding_dim)
        self.num_hidden_layers = n_layers
        blocks = []
        for _ in range(n_layers):
            f_func = (
                MHABlock_xformer(embedding_dim, n_heads, num_node=num_node)
                if use_xformer
                else MHABlock(embedding_dim, n_heads, num_node=num_node)
            )
            g_func = FFBlock(embedding_dim, intermediate_dim)
            # we construct a reversible block with our F and G functions
            blocks.append(rv.ReversibleBlock(f_func, g_func, split_along_dim=-1))

        self.sequence = rv.ReversibleSequence(nn.ModuleList(blocks))

    def forward(self, x: Tensor, mask=None, agent_num=3, distance=None):
        if hasattr(self, "init_projection_layer"):
            x = self.init_projection_layer(x)
        x = torch.cat([x, x], dim=-1)
        out = self.sequence(x)
        return (
            torch.stack(out.chunk(2, dim=-1))[-1],
            torch.stack(out.chunk(2, dim=-1))[-1].mean(dim=1),
        )
