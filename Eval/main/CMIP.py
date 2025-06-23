import sys
import torch
import random
import numpy as np
import argparse

# Add local module path
sys.path.append("./")

from utils.functions import parse_softmax_temperature
from utils import load_model
from improvement.policy import (
    eval_dataset,
    unified_data,
    compute_centroid,
    compute_distance_matirx,
    Insert_idle,
    Neighbour_roll,
)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(opts) -> None:
    """
    Main evaluation loop for CMIP on various parameter settings.
    Args:
        opts: Parsed command-line arguments or options object
    """
    # Set random seed
    set_seed(opts.seed)

    opts.device = torch.device(f"cuda:{opts.device_id}")
    opts.sample_size = opts.val_size
    opts.eval_batch_size = 1

    # Load models and optimal cost reference
    model_CM, _ = load_model(opts.constructive_model_path, agent_num=opts.agent_num)
    opt_cost = torch.load(opts.opt_cost_path)

    # IP strategy parameters
    epsilon = opts.epsilon
    roll_neighbour_num = opts.neighbour_num

    # Parameter combinations (graph_size, agent_num, neighbour_num, epsilon)
    param_all = (
        [(200, i, roll_neighbour_num, epsilon) for i in range(5, 25, 5)]
        + [(500, i, roll_neighbour_num, epsilon) for i in range(20, 60, 10)]
        + [(1000, i, roll_neighbour_num, epsilon) for i in range(25, 125, 25)]
        + [(2000, i, roll_neighbour_num, epsilon) for i in range(50, 250, 50)]
        + [(5000, i, roll_neighbour_num, epsilon) for i in range(200, 600, 100)]
        + [(10000, i, roll_neighbour_num, epsilon) for i in range(250, 1250, 250)]
    )

    result_buffer = {}

    # Evaluation loop
    for param in param_all:
        graph_size, agent_num, neighbour_num, epsilon = param
        print(
            f"\nEvaluating -> graph_size: {graph_size}, agent_num: {agent_num}, neighbour_num: {neighbour_num}"
        )

        # Update opts and model
        opts.graph_size = graph_size
        opts.agent_num = agent_num
        model_CM.graph_size = graph_size
        model_CM.agent_num = agent_num

        # Load dataset
        dataset_path = f"data/mtsp/mtsp{graph_size}_test_seed{opts.seed}.pkl"
        print(f"Loading dataset: {dataset_path}")
        dataset = model_CM.problem.make_dataset(
            filename=dataset_path, num_samples=opts.sample_size, offset=0
        )

        # Constructive evaluation
        cost, duration_constructive, _, tours = eval_dataset(
            model_CM, dataset, 0, opts.softmax_temperature, opts, offset=0
        )

        # Prepare for improvement
        action_max_CMIP, reward_CM, _, reward_all_CMIP, action_all_CMIP = unified_data(
            tours, cost, dataset, opts
        )
        CM_cost = -reward_all_CMIP.min(dim=-1)[0].mean()

        # Compute centroids and distances
        centroids = compute_centroid(action_max_CMIP, dataset, opts)
        distance_matrix = compute_distance_matirx(centroids, opts)

        # Handle idle agents
        check_idle = Insert_idle(reward_all_CMIP, action_all_CMIP, opts)
        do_not_optimize_batch_index, reward_all_CMIP_optimized, action_all_CM = check_idle.cross_insert(
            dataset
        )

        # Rolling IP optimization
        Rolling = Neighbour_roll(
            centroids,
            distance_matrix,
            reward_all_CMIP_optimized,
            action_all_CM,
            do_not_optimize_batch_index,
            reward_CM,
            opts,
            num_neighbour=neighbour_num,
            rolling_num=agent_num // (neighbour_num - 1) + 1,
            epsilon=epsilon,
            CM_model_path=opts.improvement_model_path,
        )

        action_all_rolling, reward_all_scroll_optimization, duration_rolling = Rolling.epsilon_greedy_ip_cross(
            dataset
        )

        reward_min_CMIP_optimized, _ = reward_all_scroll_optimization.min(dim=-1)
        CMIP_cost = -reward_min_CMIP_optimized.mean()
        total_duration = duration_rolling + np.array(duration_constructive).mean()

        # Record results
        result_buffer[param] = [
            CM_cost,
            np.array(duration_constructive).mean(),
            CMIP_cost,
            total_duration,
        ]

        # Calculate and report gaps
        ref_cost = opt_cost[(graph_size, agent_num)]
        gap_CM = 100 * (CM_cost - ref_cost) / ref_cost
        gap_CMIP = 100 * (CMIP_cost - ref_cost) / ref_cost

        print(
            f"CM gap: {gap_CM:.1f}%, CMIP gap: {gap_CMIP:.1f}%, \n"
            f"CM cost: {CM_cost:.1f}, CMIP cost: {CMIP_cost:.1f}, Time: {total_duration:.1f}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CMIP Evaluation with IP-based Refinement"
    )
    parser.add_argument("--seed", type=int, default=3333, help="Random seed")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID")
    parser.add_argument(
        "--agent_num", type=int, default=20, help="Initial number of agents"
    )
    parser.add_argument(
        "--graph_size", type=int, default=200, help="Initial graph size"
    )
    parser.add_argument(
        "--constructive_model_path",
        type=str,
        default="pretrain/CMIP_constructive/epoch-20.pt",
        help="Path to constructive model",
    )
    parser.add_argument(
        "--improvement_model_path",
        type=str,
        default="pretrain/improvement_model/epoch-99.pt",
        help="Path to improvement model",
    )
    parser.add_argument(
        "--opt_cost_path",
        type=str,
        default="pretrain/opt_cost.pt",
        help="Path to optimal cost file",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.0, help="Epsilon for rolling strategy"
    )
    parser.add_argument(
        "--neighbour_num",
        type=int,
        default=5,
        help="Number of neighbors for IP rolling",
    )
    parser.add_argument("--problem", default="mtsp", type=str, help="problem type")
    parser.add_argument(
        "--val_size",
        type=int,
        default=100,
        help="Number of instances used for reporting validation performance",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of instances used for reporting validation performance",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset where to start in dataset (default 0)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size to use during (baseline) evaluation",
    )
    parser.add_argument(
        "--decode_type",
        type=str,
        default="greedy",
        help="Decode type, greedy or sampling",
    )
    parser.add_argument(
        "--width",
        type=int,
        nargs="+",
        default=[0],
        help="Sizes of beam to use for beam search (or number of samples for sampling), "
        "0 to disable (default), -1 for infinite",
    )
    parser.add_argument(
        "--decode_strategy",
        type=str,
        default="greedy",
        help="Sampling (sample) or Greedy (greedy)",
    )
    parser.add_argument(
        "--softmax_temperature",
        type=parse_softmax_temperature,
        default=1,
        help="Softmax temperature (sampling or bs)",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--no_progress_bar", action="store_true", help="Disable progress bar"
    )
    parser.add_argument(
        "--N_aug", default=8, type=int, help="how any augmentation of instance"
    )
    parser.add_argument(
        "--max_calc_batch_size",
        default=100000,
        type=int,
        help="max batch size for calculation",
    )
    opts = parser.parse_args()

    main(opts)
