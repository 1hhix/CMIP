import torch
from torch import nn
import math
from typing import NamedTuple
from utils.beam_search import CachedLookup
from utils.functions import sample_many
from nets.positional_encoding import PostionalEncoding
from nets.constructive.Encoder import RevMHAEncoder
from xformers import ops as xops
from einops import rearrange
import torch.nn.functional as F
import numpy as np


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """

    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key],
        )


class CM(nn.Module):
    """
    Constructive Model for mTSP using attention-based encoder-decoder architecture.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        problem,
        agent_num: int = 3,
        n_encode_layers: int = 2,
        tanh_clipping: float = 10.0,
        mask_inner: bool = True,
        mask_logits: bool = True,
        n_heads: int = 8,
        checkpoint_encoder: bool = False,
        shrink_size = None,
        add_distance_matrix: bool = False,
        num_node = None,
        use_xformer: bool = True,
        use_circle: bool = False,
        use_gate: bool = False,
    ):
        """
        Initialize the Constructive Model.
        Args:
            embedding_dim: Dimension of node embeddings
            hidden_dim: Hidden dimension
            problem: Problem instance
            agent_num: Number of agents
            n_encode_layers: Number of encoder layers
            tanh_clipping: Clipping value for tanh
            mask_inner: Whether to mask inner nodes
            mask_logits: Whether to mask logits
            n_heads: Number of attention heads
            checkpoint_encoder: Use checkpointing for encoder
            shrink_size: Optional shrink size
            add_distance_matrix: Whether to add distance matrix
            num_node: Number of nodes
            use_xformer: Use xFormers for attention
            use_circle: Use circle features
            use_gate: Use gating mechanism
        """
        super(CM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.agent_num = agent_num
        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.positional_encoding = PostionalEncoding(
            d_model=embedding_dim, max_len=10000
        )

        # Problem specific context parameters (placeholder and step context dimension)
        step_context_dim = (
            2 * embedding_dim + 2
        )  # Embedding of current_agent, current node, # of left cities and # of left agents
        node_dim = 2  # x, y
        self.init_embed_depot = nn.Linear(2, embedding_dim)
        self.pos_emb_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=False)
        )
        self.alpha = nn.Parameter(torch.Tensor([1]))

        self.dis_emb = nn.Sequential(nn.Linear(3, embedding_dim, bias=False))
        self.embedder = RevMHAEncoder(
            n_layers=self.n_encode_layers,
            n_heads=n_heads,
            embedding_dim=embedding_dim,
            input_dim=embedding_dim,
            intermediate_dim=embedding_dim * 4,
            add_init_projection=False,
            num_node=num_node,
            use_xformer=use_xformer,
        )

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(
            embedding_dim, 3 * embedding_dim, bias=False
        )
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(
            step_context_dim, embedding_dim, bias=False
        )
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.add_distance_matrix = add_distance_matrix

        self.n_heads = n_heads
        self.distance = None
        self.fusionlayer = nn.Linear(embedding_dim * 2, embedding_dim, bias=False)
        self.gatelayer = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gateactivation = nn.GELU()
        self.use_circle = use_circle
        self.use_gate = use_gate

    def set_decode_type(self, decode_type: str, temp: float = None) -> None:
        """Set the decoding type and optional temperature."""
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi: bool = False):
        """
        Forward pass for the Constructive Model.
        Args:
            input: Input tensor or dict
            return_pi: Whether to return output sequences
        Returns:
            cost, log_likelihood (and pi if return_pi)
        """

        # use distance matrix to achieve local attention
        depot_locs = input[:, 0:1, :]
        depot_locs = depot_locs.repeat(1, self.agent_num + 1, 1)
        locs_add_depot = torch.cat((depot_locs, input[:, 1:, :]), 1)
        self.distance = (
            locs_add_depot.unsqueeze(1) - locs_add_depot.unsqueeze(2)
        ).norm(dim=-1)

        embeddings, _ = self.embedder(
            self._init_embed(input), agent_num=self.agent_num, distance=None
        )

        self.local_agent_num = self.num_cities // (self.agent_num)
        _log_p, pi, cost = self._inner(input, embeddings)

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        ll = self._calc_log_likelihood(_log_p, pi, None)

        if return_pi:
            return cost, ll, pi

        return cost, ll

    def precompute_fixed(self, input):
        """Precompute and cache fixed context for decoding."""
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def _calc_log_likelihood(self, _log_p, a, mask):
        """Calculate log likelihood for selected actions."""

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (
            log_p > -1000
        ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def basesin(self, x, T, fai=0):
        """Compute a sine-based cyclic positional encoding."""
        return np.sin(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)

    def basecos(self, x, T, fai=0):
        """Compute a cosine-based cyclic positional encoding."""
        return np.cos(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)

    def cyclic_position_encoding_pattern(self, n_position, emb_dim, mean_pooling=True):
        """Generate cyclic positional encoding pattern."""
        Td_set = np.linspace(
            np.power(n_position, 1 / (emb_dim // 2)),
            n_position,
            emb_dim // 2,
            dtype="int",
        )
        x = np.zeros((n_position, emb_dim))

        for i in range(emb_dim):
            Td = (
                Td_set[i // 3 * 3 + 1]
                if (i // 3 * 3 + 1) < (emb_dim // 2)
                else Td_set[-1]
            )
            fai = (
                0
                if i <= (emb_dim // 2)
                else 2 * np.pi * ((-i + (emb_dim // 2)) / (emb_dim // 2))
            )
            longer_pattern = np.arange(0, np.ceil((n_position) / Td) * Td, 0.01)
            if i % 2 == 1:
                x[:, i] = self.basecos(longer_pattern, Td, fai)[
                    np.linspace(
                        0, len(longer_pattern), n_position, dtype="int", endpoint=False
                    )
                ]
            else:
                x[:, i] = self.basesin(longer_pattern, Td, fai)[
                    np.linspace(
                        0, len(longer_pattern), n_position, dtype="int", endpoint=False
                    )
                ]

        pattern = torch.from_numpy(x).type(torch.FloatTensor)
        pattern_sum = torch.zeros_like(pattern)

        # averaging the adjacient embeddings if needed (optional, almost the same performance)
        arange = torch.arange(n_position)
        pooling = [0] if not mean_pooling else [-2, -1, 0, 1, 2]
        time = 0
        for i in pooling:
            time += 1
            index = (arange + i + n_position) % n_position
            pattern_sum += pattern.gather(0, index.view(-1, 1).expand_as(pattern))
        pattern = 1.0 / time * pattern_sum - pattern.mean(0)

        return pattern

    def _init_embed(self, input):

        if len(input.size()) == 2:
            input = input.unsqueeze(0)
        num_cities = input.size(1) - 1
        self.num_cities = num_cities

        # Embedding of depot
        depot_embedding = self.init_embed_depot(input[:, 0:1, :])

        # Make the depot embedding the same for all agents
        depot_embedding = depot_embedding.repeat(1, self.agent_num + 1, 1)
        if self.use_circle:
            _, seq_length, embedding_dim = depot_embedding.shape
            positional_embedding = self.cyclic_position_encoding_pattern(
                seq_length, embedding_dim
            ).to(depot_embedding.device)

            # Add the positional embedding to the depot embedding to give order bias to the agents
            depot_embedding = depot_embedding + positional_embedding[None, :, :]

        else:
            positional_embedding = self.positional_encoding(
                depot_embedding.size(0), depot_embedding.size(1)
            )
            positional_embedding = positional_embedding.to(depot_embedding.device)
            positional_embedding = (
                self.alpha * self.pos_emb_proj(positional_embedding) / self.agent_num
            )

            # Add the positional embedding to the depot embedding to give order bias to the agents
            depot_embedding = depot_embedding + positional_embedding[None, :, :]

        return torch.cat((depot_embedding, self.init_embed(input[:, 1:, :])), 1)

    def _inner(self, input, embeddings):

        outputs = []
        sequences = []
        selected = None

        state = self.problem.make_state(input, self.agent_num)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        batch_size = state.ids.size(0)
        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):
            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            log_p, mask = self._get_log_p(fixed, state, selected)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(
                log_p.exp()[:, 0, :], mask[:, 0, :]
            )  # Squeeze out steps dimension
            state = state.update(selected)

            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)
            i += 1
        return torch.stack(outputs, 1), torch.stack(sequences, 1), state.lengths

    def sample_many(self, input, batch_rep=1, iter_rep=1, agent_num=3, aug=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        depot_locs = input[:, 0:1, :]
        depot_locs = depot_locs.repeat(1, self.agent_num + 1, 1)
        locs_add_depot = torch.cat((depot_locs, input[:, 1:, :]), 1)
        self.distance = (
            locs_add_depot.unsqueeze(1) - locs_add_depot.unsqueeze(2)
        ).norm(dim=-1)
        self.num_cities = input.size(1) - 1
        self.local_agent_num = self.num_cities // (self.agent_num)
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(
                input[0], pi
            ),  # Don't need embeddings as input to get_costs
            (
                input,
                self.embedder(self._init_embed(input), agent_num)[0],
            ),  # Pack input with embeddings (additional input)
            batch_rep,
            iter_rep,
            aug,
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(
                1, selected.unsqueeze(-1)
            ).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = self.project_node_embeddings(
            embeddings
        ).chunk(
            3, dim=-1
        )

        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed),
            self._make_heads(glimpse_val_fixed),
            logit_key_fixed.contiguous(),
        )
        return AttentionModelFixed(
            embeddings, fixed_context, *fixed_attention_node_data
        )

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(
                log_p.size(0), 1
            )[:, None, :],
        )

    def _get_log_p(self, fixed, state, selected, normalize=True):

        query = (
            fixed.context_node_projected
            + self.project_step_context(
                self._get_parallel_step_context(fixed.node_embeddings, state)
            )
            + self.dis_emb(
                torch.cat(
                    (
                        state.lengths.gather(-1, state.count_depot),
                        state.max_distance,
                        state.remain_max_distance,
                    ),
                    -1,
                )
            )[:, None, :]
        )

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p, _ = self._one_to_many_logits(
            query, glimpse_K, glimpse_V, logit_K, mask, selected
        )

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        return torch.cat(
            (
                embeddings.gather(
                    1,
                    torch.cat((current_node, state.agent_idx), 1)[:, :, None].expand(
                        batch_size, 2, embeddings.size(-1)
                    ),
                ).view(batch_size, 1, -1),
                1.0
                - torch.ones(
                    size=state.count_depot[:, :, None].shape, device=embeddings.device
                )
                * (state.count_depot[:, :, None] + 1)
                / self.agent_num,
                state.left_city[:, :, None] / self.num_cities,
            ),
            2,
        )

    def _local_attention(
        self, batch_size, embed_dim, selected, mask, glimpse_Q, glimpse_K, glimpse_V
    ):
        distance_to_current_nodes = self.distance[
            torch.arange(self.distance.size(0)), selected, :
        ]
        distance_to_current_nodes[mask.squeeze(1)] = float("inf")
        nearest_nodes_indices = torch.topk(
            distance_to_current_nodes, self.local_agent_num, dim=1, largest=False
        )[1]

        local_out = xops.memory_efficient_attention(
            glimpse_Q,
            glimpse_K[
                torch.arange(batch_size).unsqueeze(1), nearest_nodes_indices
            ].view(batch_size, self.local_agent_num, self.n_heads, -1),
            glimpse_V[
                torch.arange(batch_size).unsqueeze(1), nearest_nodes_indices
            ].view(batch_size, self.local_agent_num, self.n_heads, -1),
        ).view(batch_size, embed_dim)
        return local_out

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, selected):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, key_size)

        mask_attn, last_dim = self.padding_mask(mask)

        mhaout = xops.memory_efficient_attention(
            glimpse_Q,
            glimpse_K,
            glimpse_V,  # batch,seq_len,embedding
            attn_bias=mask_attn[:, :, :, :last_dim],
        ).view(batch_size, embed_dim)
        if self.use_local:
            # choose the nearest n salesman according to the distance to do the local attention
            if selected != None:
                with torch.no_grad():
                    local_out = self._local_attention(
                        batch_size,
                        embed_dim,
                        selected,
                        mask,
                        glimpse_Q,
                        glimpse_K,
                        glimpse_V,
                    )
                if self.use_gate:
                    out = mhaout + local_out * self.gateactivation(
                        self.gatelayer(mhaout)
                    )
                else:
                    out = self.fusionlayer(torch.concat((local_out, mhaout), dim=1))
            else:
                out = mhaout.view(batch_size, -1, embed_dim)
        else:
            out = mhaout

        # mhaout=glimpse_Q.transp
        glimpse = self.project_out(out.view(batch_size, -1, embed_dim))

        logits = torch.bmm(glimpse, logit_K.squeeze(1).transpose(-2, -1)) / math.sqrt(
            glimpse.size(-1)
        )

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):

        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v):
        return rearrange(v, "... g (h s) -> ... g h s", h=self.n_heads)

    def padding_mask(self, mask):
        mask_expanded = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        mask_attn = torch.zeros_like(mask_expanded, dtype=torch.float32)
        mask_attn[mask_expanded > 0] = torch.finfo(torch.float32).min

        last_dim = mask_attn.shape[-1]
        padding_size = 8 - last_dim % 8

        if padding_size > 0:
            mask_attn = F.pad(mask_attn, (0, padding_size), "constant", 0)
        return mask_attn, last_dim


def print_cuda_memory_info(i):
    total_memory = torch.cuda.get_device_properties(i).total_memory
    allocated_memory = torch.cuda.memory_allocated(i)
    cached_memory = torch.cuda.memory_reserved(i)

    print(f"Device {i}:")
    print(f"  Total memory: {total_memory / (1024 ** 3):.2f} GB")
    print(f"  Allocated memory: {allocated_memory / (1024 ** 3):.4f} GB")
    print(f"  Cached memory: {cached_memory / (1024 ** 3):.4f} GB")
    print("")
