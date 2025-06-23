from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.mtsp.state_mtsp import State_MTSP
from utils.beam_search import beam_search
import tsplib95
import numpy as np


class MTSP(object):
    """Multiple Traveling Salesman Problem (mTSP) utilities and static methods."""

    NAME = "mtsp"

    @staticmethod
    def get_costs(dataset: torch.Tensor, pi: torch.Tensor):
        """
        Compute the cost of a batch of tours for the mTSP.
        Args:
            dataset: Tensor of shape (batch, n, 2) with node coordinates
            pi: Tensor of shape (batch, n) with tour indices
        Returns:
            Tuple of (costs, None)
        """
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi)
            == pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - d[:, -1]).norm(p=2, dim=1),
            None,
        )

    @staticmethod
    def make_dataset(*args, **kwargs):
        """Create an MTSPDataset instance with the given arguments."""
        return MTSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        """Create an initial State_MTSP instance with the given arguments."""
        return State_MTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(
        input: torch.Tensor,
        beam_size: int,
        expand_size=None,
        compress_mask: bool = False,
        model=None,
        max_calc_batch_size: int = 4096,
        agent_num: int = 5,
    ):
        """
        Perform beam search for the mTSP using the provided model.
        Args:
            input: Input tensor for the model
            beam_size: Beam width
            expand_size: Optional expansion size
            compress_mask: Whether to use compressed mask
            model: Model instance with propose_expansions method
            max_calc_batch_size: Max batch size for calculations
            agent_num: Number of agents
        Returns:
            Beam search result
        """
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam,
                fixed,
                expand_size,
                normalize=True,
                max_calc_batch_size=max_calc_batch_size,
            )

        state = MTSP.make_state(
            input,
            agent_num=agent_num,
            visited_dtype=torch.int64 if compress_mask else torch.uint8,
        )

        return beam_search(state, beam_size, propose_expansions)


class MTSPDataset(Dataset):
    """PyTorch Dataset for mTSP instances, supporting .tsp and pickle formats."""

    def __init__(
        self,
        filename=None,
        size: int = 50,
        num_samples: int = 1000000,
        offset: int = 0,
        distribution=None,
        opts=None,
    ):
        """
        Initialize the dataset from a file or generate random instances.
        Args:
            filename: Path to .tsp or pickle file
            size: Number of nodes per instance
            num_samples: Number of samples to generate/load
            offset: Offset for loading from file
            distribution: Not used
            opts: Options object for custom generation
        """
        super(MTSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            if os.path.splitext(filename)[1] == ".tsp":
                problem = tsplib95.load(filename)
                max_val = np.array(list(problem.node_coords.values())).max()
                self.data = [torch.FloatTensor(list(problem.node_coords.values()))]
            else:
                try:
                    with open(filename, "rb") as f:
                        data = pickle.load(f)
                        self.data = [
                            torch.FloatTensor(row)
                            for row in (data[offset : offset + num_samples])
                        ]
                except:
                    import re

                    pattern = r"\d+"
                    match_obj = re.findall(pattern, filename)
                    self.data = [
                        torch.FloatTensor(int(match_obj[0]), 2).uniform_(0, 1)
                        for i in range(num_samples)
                    ]
                    with open(f"{filename}", "wb") as f:
                        pickle.dump(self.data, f)

                # print(self.data)
        else:
            if opts.train_on_different_node:
                self.data = [
                    torch.FloatTensor(graph_size, 2).uniform_(0, 1)
                    for graph_size in range(opts.graph_min, opts.graph_max)
                    for i in range(int(num_samples / (opts.graph_max - opts.graph_min)))
                ]
            else:
                # Sample points randomly in [0, 1] square
                self.data = [
                    torch.FloatTensor(size, 2).uniform_(0, 1)
                    for i in range(num_samples)
                ]

        self.size = len(self.data)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return the sample at the given index."""
        return self.data[idx]
