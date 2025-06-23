import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from utils import move_to
from utils.problem_augment import augment
import random
import numpy as np


def validate(model, dataset, opts) -> float:
    """
    Validate the model on the given dataset and print average cost.
    Args:
        model: The model to validate
        dataset: The dataset to validate on
        opts: Options/configuration object
    Returns:
        Average cost as a float
    """
    # Validate
    print("Validating...")
    cost = rollout(model, dataset, opts)
    # print(cost.shape)
    avg_cost = cost.mean()
    print(
        "Validation overall avg_cost: {} +- {}".format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))
        )
    )

    return avg_cost


def rollout(model, dataset, opts) -> torch.Tensor:
    """
    Run the model in greedy evaluation mode on the dataset and return costs.
    Args:
        model: The model to evaluate
        dataset: The dataset to evaluate on
        opts: Options/configuration object
    Returns:
        Concatenated tensor of costs
    """
    # Put in greedy evaluation mode!
    model.set_decode_type("greedy")

    model.eval()

    def eval_model_bat(bat, batch_size, aug=8):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
            # print(cost.shape)
            cost, _ = cost.view(aug, -1).min(0, keepdim=True)
            cost = cost.transpose(0, 1)

        return cost.data.cpu()

    return torch.cat(
        [
            eval_model_bat(
                augment(bat, opts.N_aug),
                batch_size=opts.eval_batch_size,
                aug=opts.N_aug,
            )
            for bat in tqdm(
                DataLoader(dataset, batch_size=opts.eval_batch_size),
                disable=opts.no_progress_bar,
            )
        ],
        0,
    )


def clip_grad_norms(param_groups, max_norm: float = math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping.
    Args:
        param_groups: List of parameter groups (from optimizer)
        max_norm: Maximum allowed norm
    Returns:
        Tuple of (grad_norms, grad_norms_clipped)
    """ 
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"],
            max_norm
            if max_norm > 0
            else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2,
        )
        for group in param_groups
    ]
    grad_norms_clipped = (
        [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    )
    return grad_norms, grad_norms_clipped


def train_epoch(
    model,
    optimizer,
    lr_scheduler,
    epoch,
    problem,
    opts,
    save_checkpoint=True,
    probabilities=None,
    best_reward=None,
):

    print(
        "Start train epoch {}, lr={} for run {}".format(
            epoch, optimizer.param_groups[0]["lr"], opts.run_name
        )
    )

    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    graph_size = opts.graph_size

    if opts.train_on_different_node:
        training_dataloader = range(int(opts.epoch_size / opts.batch_size))
    else:
        training_dataset = problem.make_dataset(
            size=graph_size,
            num_samples=opts.epoch_size,
            distribution=opts.data_distribution,
            opts=opts,
        )
        training_dataloader = DataLoader(
            training_dataset, batch_size=opts.batch_size, num_workers=1, shuffle=True
        )

    # Put model in train mode!
    model.train()
    model.set_decode_type("sampling")

    # model = get_inner_model(model)
    for batch_id, batch in enumerate(
        tqdm(training_dataloader, disable=opts.no_progress_bar, desc=f"Epoch:{epoch}")
    ):
        if opts.task_sample:
            task = [i for i in range(opts.agent_min, opts.agent_max + 1)]
            agent_num = np.random.choice(task, size=1, p=probabilities)[0]
            if opts.train_on_different_node:
                agent_num = opts.train_agent_num
                graph_size = random.sample(
                    range(opts.graph_min, opts.graph_max + 1), 1
                )[0]
                batch = torch.rand(opts.batch_size, graph_size, 2)
                model.graph_size = graph_size
        else:
            agent_num = random.sample(range(opts.agent_min, opts.agent_max + 1), 1)[0]

        model.agent_num = agent_num
        model.embedder.agent_num = agent_num

        best_reward = train_batch(
            model, optimizer, batch, opts, best_reward=best_reward
        )
        if save_checkpoint:
            if batch_id > 0 and batch_id % 100 == 0:
                print("Saving model and state...")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all(),
                    },
                    os.path.join(opts.save_dir, "epoch-{}.pt".format(epoch)),
                )

    if save_checkpoint:
        if (
            opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0
        ) or epoch == opts.n_epochs - 1:
            print("Saving model and state...")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(),
                },
                os.path.join(opts.save_dir, "epoch-{}.pt".format(epoch)),
            )
    epoch_duration = time.time() - start_time

    step += 1

    print(
        "Finished epoch {}, took {} s".format(
            epoch, time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        )
    )
    lr_scheduler.step()

    return best_reward


def augment_xy_data_by_8_fold(xy_data, training=False):

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)

    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    # data_augmented.shape = [B, N, 16]
    if training:
        data_augmented = torch.cat(
            (dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=2
        )
        return data_augmented

    # data_augmented.shape = [8*B, N, 2]
    data_augmented = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    return data_augmented


def data_augment(batch):
    batch = augment_xy_data_by_8_fold(batch, training=True)
    theta = []
    for i in range(8):
        theta.append(
            torch.atan(batch[:, :, i * 2 + 1] / batch[:, :, i * 2]).unsqueeze(-1)
        )
    theta.append(batch)
    batch = torch.cat(theta, dim=2)
    return batch


def train_batch(model, optimizer, batch, opts, best_reward=None):
    x = move_to(batch, opts.device)
    x_aug = augment(x, opts.N_aug)
    all_length, log_likelihood = model(x_aug)
    cost_max = torch.max(all_length, dim=-1)[0]
    cost = cost_max.view(opts.N_aug, -1).permute(1, 0)

    log_likelihood = log_likelihood.view(opts.N_aug, -1).permute(1, 0)
    advantage = cost - cost.mean(dim=1).view(-1, 1)
    loss = ((advantage) * log_likelihood).mean()

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    best_cost_mean = cost_max.mean()
    best_reward[model.agent_num - opts.agent_min] = (
        best_cost_mean
        if best_cost_mean < best_reward[model.agent_num - opts.agent_min]
        else best_reward[model.agent_num - opts.agent_min]
    )

    return best_reward
