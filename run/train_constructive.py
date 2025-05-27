#!/usr/bin/env python

import os
import json
import pprint as pp
import torch
import sys
sys.path.append('./')
import torch.optim as optim
from options import get_options
from train import train_epoch, validate 
from nets.constructive.model import CM 
from utils import load_problem
import os
import torch.nn.functional as F
import numpy as np  
import random
def run(opts):
    # Set the random seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)   
  
    # Pretty print the run args
    pp.pprint(vars(opts))
  
    os.makedirs(opts.save_dir,exist_ok=True)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)
    # Set the device
    opts.device = torch.device(f"cuda:{opts.device_index}" if opts.use_cuda else "cpu")
    
    # Figure out what's the problem
    problem = load_problem(opts.problem)
 
  
    model = CM(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers, 
        tanh_clipping=opts.tanh_clipping, 
        shrink_size=opts.shrink_size, 
        num_node=opts.graph_size, 
        use_circle=opts.usecircle,
        use_gate=opts.usegate
    ).to(opts.device) 
    model.use_local=opts.use_local
     
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
    )
   
  
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution,opts=opts)
  
    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        best_reward=torch.ones(opts.agent_max+1-opts.agent_min)*torch.inf
        probabilities=np.array([1/(opts.agent_max+1-opts.agent_min) for _ in range((opts.agent_max+1-opts.agent_min))])
        reward_buffer={}
        if opts.task_sample:
          last_reward=best_reward.clone()
          num_task=opts.agent_max+1-opts.agent_min
          softmax_T=opts.softmax_T
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            best_reward=train_epoch(
                model,
                optimizer,
                lr_scheduler,
                epoch, 
                problem,
                opts, 
                save_checkpoint= opts.save_checkpoint,
                probabilities=probabilities,
                best_reward=best_reward, 
            )
            
            if opts.task_sample and epoch>opts.epoch_start:
                improve=last_reward-best_reward
                # normalize
                improve_normalize=(improve-improve.min())/(improve.max()-improve.min()) 
                Reg=F.softmax((1-improve_normalize)*softmax_T,dim=-1)
                probabilities=opts.p_sample_min+(1-opts.p_sample_min*num_task)*Reg
                probabilities=probabilities.numpy()

            reward_buffer[epoch]=best_reward.clone()
            last_reward=best_reward.clone()
          

    print('Saving model and state...')
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all()
        },
        os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
    )


if __name__ == "__main__":
    run(get_options())
