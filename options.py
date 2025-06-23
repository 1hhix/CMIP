import os
import time
import argparse
import torch

"""
Command-line options and argument parsing for CMIP training and evaluation scripts.
"""

def get_options(args=None) -> argparse.Namespace:
    """
    Parse and return command-line options for CMIP training and evaluation.
    Args:
        args: Optional list of arguments to parse (default: None, uses sys.argv)
    Returns:
        argparse.Namespace with parsed options
    """
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Min-Max mTSP with Reinforcement Learning"
    )

    # Data
    parser.add_argument(
        "--problem", default="mtsp", help="The problem to solve, default 'mtsp'"
    )
    parser.add_argument(
        "--graph_size", type=int, default=200, help="The size of the problem graph"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of instances per batch during training",
    )
    parser.add_argument(
        "--epoch_size",
        type=int,
        default=320000,
        help="Number of instances per epoch during training",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=10000,
        help="Number of instances used for reporting validation performance",
    )
    parser.add_argument(
        "--val_dataset", type=str, help="Dataset file to use for validation"
    )
    parser.add_argument(
        "--N_aug", type=int, default=8, help="The size of the problem graph"
    )

    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Dimension of input embedding"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Dimension of hidden layers in Enc/Dec",
    )
    parser.add_argument(
        "--n_encode_layers",
        type=int,
        default=3,
        help="Number of layers in the encoder/critic network",
    )
    parser.add_argument(
        "--tanh_clipping",
        type=float,
        default=10.0,
        help="Clip the parameters to within +- this value using tanh. "
        "Set to 0 to not perform any clipping.",
    )
    parser.add_argument(
        "--normalization",
        default="batch",
        help="Normalization type, 'batch' (default) or 'instance'",
    )
    parser.add_argument(
        "--agent_min", default=5, type=int, help="decide the number of agent"
    )
    parser.add_argument(
        "--agent_max", default=30, type=int, help="decide the number of robot"
    )

    # Training
    parser.add_argument(
        "--lr_model",
        type=float,
        default=1e-4,
        help="Set the learning rate for the actor network",
    )
    parser.add_argument(
        "--lr_critic",
        type=float,
        default=1e-4,
        help="Set the learning rate for the critic network",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=1.0, help="Learning rate decay per epoch"
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="Set this value to only evaluate model"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=21, help="The number of epochs to train"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed to use")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1024,
        help="Batch size to use during evaluation",
    )
    parser.add_argument(
        "--shrink_size",
        type=int,
        default=None,
        help="Shrink the batch size if at least this many instances in the batch are finished"
        " to save memory (default None means no shrinking)",
    )
    parser.add_argument(
        "--data_distribution",
        type=str,
        default=None,
        help="Data distribution to use during training, defaults and options depend on problem.",
    )

    # Misc
    parser.add_argument("--run_name", default="run", help="Name to identify the run")
    parser.add_argument(
        "--output_dir", default="outputs", help="Directory to write output models to"
    )
    parser.add_argument(
        "--epoch_start",
        type=int,
        default=0,
        help="Start at epoch # (relevant for learning rate decay)",
    )
    parser.add_argument(
        "--checkpoint_epochs",
        type=int,
        default=2,
        help="Save checkpoint every n epochs (default 1), 0 to save no checkpoints",
    )
    parser.add_argument(
        "--no_progress_bar", action="store_true", help="Disable progress bar"
    )

    parser.add_argument(
        "--task_sample", type=bool, default=True, help="Enable task sampling"
    )
    parser.add_argument(
        "--usecircle", type=bool, default=True, help="Enable the use of CPE"
    )
    parser.add_argument(
        "--p_sample_min", type=float, default=0.02, help="Minimum sampling probability"
    )
    parser.add_argument(
        "--usegate", type=bool, default=True, help="Enable the use of gating mechanism"
    )
    parser.add_argument(
        "--device_index",
        type=int,
        default=0,
        help="Index of the device to use (e.g., GPU ID)",
    )
    parser.add_argument(
        "--use_local", type=bool, default=True, help="Whether to use local attention"
    )

    parser.add_argument("--save_checkpoint", type=bool, default=True)

    parser.add_argument("--num_start", type=int, default=1)

    parser.add_argument("--train_on_different_node", type=bool, default=False)
    parser.add_argument(
        "--train_agent_num",
        type=int,
        default=5,
        help="set training agent num when train on differnt node",
    )
    parser.add_argument(
        "--graph_min",
        type=int,
        default=50,
        help="min graph size when train on differnt node",
    )
    parser.add_argument(
        "--graph_max",
        type=int,
        default=200,
        help="max graph size when train on differnt node",
    )
    parser.add_argument(
        "--softmax_T",
        type=int,
        default=5,
        help="Temperature parameter for softmax scaling",
    )

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name
    )
    assert (
        opts.epoch_size % opts.batch_size == 0
    ), "Epoch size must be integer multiple of batch size!"
    return opts
