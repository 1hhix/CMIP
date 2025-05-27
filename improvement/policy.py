
import torch
import argparse 
from tqdm import tqdm
from utils import move_to
from torch.utils.data import DataLoader
import time
from utils.problem_augment import augment
from utils.ops import gather_by_index, get_tour_length
from utils import load_model, move_to
import heapq
from utils.functions import parse_softmax_temperature 
def eval_dataset(model, dataset, width, softmax_temp, opts, offset):
 
    device = opts.device 
    results, max_val, start_time = _eval_dataset(model, dataset, width, softmax_temp, opts, device)

    costs, tours, durations = zip(*results)  
 
    return costs, durations, max_val,tours


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device) 
    model.set_decode_type(
        "greedy",
        temp=softmax_temp) 
    
    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    if opts.N_aug > 1:
        aug = opts.N_aug
    else:
        aug = 1

    for batch in tqdm(dataloader, disable=opts.no_progress_bar,desc='CM:'):
        if opts.problem == 'mtsp':
            max_val = batch.max()
            if max_val > 1:
                batch = batch/max_val
        else:
            max_val = None

        # For TSPLIB
        if aug > 1:
            batch = augment(batch, aug)

        # distance_matrix = torch.cdist(batch, batch, p=2)
        batch = move_to(batch, device)

        start =  time.perf_counter()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy' and opts.N_aug == 8:
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                else:
                    batch_rep = width
                    iter_rep = 1
                
                sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep, agent_num=opts.agent_num, aug=aug)
    
        duration = time.perf_counter() - start
        results.append((costs, sequences,duration))
    
    return results, max_val, start

def get_split_data(action,padding=True):
  zero_indices = (action == 0).nonzero().squeeze()
  zero_indices=zero_indices.view(action.shape[0],zero_indices.shape[0]//action.shape[0],-1)
  index=torch.roll(zero_indices, -1, dims=1)-zero_indices
  split_index=torch.cat((zero_indices[:,0,:].unsqueeze(1)+1,index[:,:-1,:]),dim=1)[:,:,1]
  split_data=torch.split(action.view(-1),split_size_or_sections=split_index.view(-1).tolist(),dim=0)
  if padding:
    return torch.nn.utils.rnn.pad_sequence(split_data, batch_first=True).to(torch.int64)
  else:
    return split_data

def get_reward(action_max,locs,single_agent=False,agent_num=1,batch_operation=True):
    action_reward=action_max.clone()
    if not single_agent:
        split_data=get_split_data(action_reward)
    else:
        split_data=action_reward
        agent_num=1
    depot = locs[..., 0:1, :].repeat_interleave(agent_num, dim=0)
    # if the input 'action_max' and 'locs' are batch data
    if batch_operation:
        locs_ordered = torch.cat([depot, gather_by_index(locs.repeat_interleave(agent_num, dim=0),split_data),depot], dim=1)
    else:
        locs_ordered = torch.cat([depot, gather_by_index(locs.repeat_interleave(agent_num, dim=0),split_data,dim=0),depot], dim=0)
    
    if single_agent:
        reward_all=-get_tour_length(locs_ordered)
    else:
        reward_all=-get_tour_length(locs_ordered).view(-1,agent_num)
    return reward_all



def normalize_coord(coord:torch.Tensor) -> torch.Tensor:  
    x, y = coord[:,:, 0], coord[:,:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_scaled = (x - x_min) / (x_max - x_min) 
    y_scaled = (y - y_min) / (y_max - y_min)
    coord_scaled = torch.stack([x_scaled, y_scaled], dim=-1)
    return coord_scaled 
   

def get_options(graph_size=200,agent_num=10):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='CUDA device index')
    parser.add_argument('--problem', default="mtsp", type=str, help="problem type")
    parser.add_argument('--graph_size', default=graph_size, type=str, help="problem type")
    parser.add_argument('--val_size', type=int, default=100,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--decode_type', type=str, default='greedy',
                        help='Decode type, greedy or sampling')
    parser.add_argument('--width', type=int, nargs='+', default=[0],
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                                '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', type=str, default='greedy',
                        help='Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--agent_num', default=agent_num, type=int, help="decide the number of agent")
    parser.add_argument('--N_aug', default=8, type=int, help="how any augmentation of instance")
    parser.add_argument('--max_calc_batch_size', default=100000, type=int, help="max batch size for calculation")
    opts = parser.parse_args([])
    
    return opts


def compute_centroid(action_max_CMIP,dataset,opts):
  action_all_CMIP_no_padding=get_split_data(action_max_CMIP,padding=False)
  locs=torch.stack(dataset.data).to(opts.device)
  locs_repeat_agent_num=locs.repeat_interleave(opts.agent_num,dim=0).to(opts.device)
  centroids_single=torch.stack([torch.mean(locs[action],dim=0) for locs,action in zip(locs_repeat_agent_num,action_all_CMIP_no_padding)])
  centroids=centroids_single.view(opts.val_size,opts.agent_num,2)
  # assert (centroids>0).all()
  return centroids
def compute_distance_matirx(centroids,opts):
  # distance matrix
  distance_matrix=(centroids.unsqueeze(1)-centroids.unsqueeze(2)).norm(dim=-1)
  ## mask self
  mask_self = torch.eye(distance_matrix.size(-1), dtype=torch.bool).unsqueeze(0).to(opts.device)
  distance_matrix.masked_fill_(mask_self.expand_as(distance_matrix), torch.inf)
  return distance_matrix

def find_nearest_masked(batch_id,current_index,distance_matrix,opts):
  distance_matrix_batch_id=distance_matrix[batch_id].clone()
  distance_matrix_batch_id[:,current_index]=torch.inf
  nearest_indices=[current_index]

  for _ in range(opts.agent_num-1):
    _,next_index=distance_matrix_batch_id[current_index].min(dim=0)
    distance_matrix_batch_id[:,next_index]=torch.inf
    nearest_indices.append(next_index.item())
  return nearest_indices

def unified_data(tours,cost,dataset,opts): 
  action_max_CMIP=torch.stack(tours).view(-1,opts.agent_num+opts.graph_size)[:,1:]-opts.agent_num
  reward_CMIP=-torch.stack(cost).view(-1)
  action_max_CMIP[action_max_CMIP<0]=0

  ## split data
  split_data=get_split_data(action_max_CMIP)

  ## Get CMIP reward action pair for each salesman
  locs=torch.stack(dataset.data).to(opts.device)
  reward_all_CMIP=get_reward(action_max_CMIP,locs,agent_num=opts.agent_num)# input action locs agent num

  ## get all salesman action 
  action_all_CMIP=split_data.view(opts.val_size,opts.agent_num,-1)  # [opts.val_size,salesman_num,actions_num]
  return  action_max_CMIP,reward_CMIP,locs,reward_all_CMIP,action_all_CMIP


def data_process_lkh3_NCE(action1,action2,dataset):
    nonzero_mask1 = action1 != 0
    nonzero_mask2 = action2 != 0
    action1_nonzero = action1[nonzero_mask1]
    action2_nonzero = action2[nonzero_mask2]
    if len(action1_nonzero) < 2 or len(action2_nonzero) < 2:
        need_optimize = False
    else:
        need_optimize = True
    regenerate_action = torch.cat((action1_nonzero, action2_nonzero))

    with open('LKH-3.0.9/two_salesman.tsp', 'r') as file:
        lines = file.readlines()

    dataset_list = dataset.tolist()
    new_lines = []
    # depot
    new_lines.append(f" {1} {dataset_list[0][0]*1000} {dataset_list[0][1]*1000}\n")
    for num, index  in enumerate(regenerate_action):
        new_lines.append(f" {num+2} {dataset_list[index][0]*1000} {dataset_list[index][1]*1000}\n")
    new_lines.append('EOF')
    lines[7:] = new_lines
    lines[3] = f'DIMENSION: {len(regenerate_action)+1}\n'

    with open('LKH-3.0.9/two_salesman.tsp', 'w') as file:
        file.writelines(lines)  
    
    return len(regenerate_action), need_optimize, regenerate_action
 

def insert(queue, priority, item):
  heapq.heappush(queue, (priority, item))
def pop_min(queue):
  _, item = heapq.heappop(queue)
  return item

def split_reproject(reprojected_actions,target_length):
  zeros_indices=(reprojected_actions==0).nonzero().squeeze()+1
  index=torch.cat((zeros_indices[:1],zeros_indices[1:]-zeros_indices[:-1]),dim=0).tolist()
  split_data_reproject=torch.split(reprojected_actions,split_size_or_sections=index)
  padding_data=torch.nn.utils.rnn.pad_sequence(split_data_reproject, batch_first=True).to(torch.int64)
  return torch.nn.functional.pad(padding_data, (0,target_length-padding_data.shape[-1]), mode='constant', value=0)

class Neighbour_roll:
    def __init__(self,
                 centroids,
                 distance_matrix,
                 reward_all_CMIP_optimized,
                 action_all_CMIP,
                 do_not_optimize_batch_index,
                 reward_CMIP,
                 opts,
                 num_neighbour=3,
                 rolling_num=5,
                 start_node=None,
                 epsilon=0.1,
                 CM_model_path=None):
        self.opts = opts
        self.reward_CMIP = reward_CMIP
        self.centroids_mask = centroids.clone()
        self.reward_all_scroll_optimization = reward_all_CMIP_optimized.clone()
        self.action_all_rolling = self._expand_action_tensor(action_all_CMIP, num_neighbour)

        self.dont_need_optimize = len(do_not_optimize_batch_index)
        self.do_not_optimize_batch_index = do_not_optimize_batch_index
        self.negative_optimization = 0
        self.positive_optimization = 0

        self.num_neighbour = num_neighbour
        self.rolling_num = rolling_num
        self.distance_matrix = distance_matrix
        self.start_node = start_node
        self.epsilon = epsilon

        import numpy as np
        self.rng = np.random.default_rng(12345)

        self.small_cross_model = None
        if CM_model_path is not None:
            model, _ = load_model(CM_model_path, agent_num=num_neighbour)
            self.small_cross_model = model.to(opts.device)
            self.small_cross_model.set_decode_type("greedy", temp=opts.softmax_temperature)

    def _expand_action_tensor(self, action_tensor, num_neighbour):
        padding_repeats = num_neighbour - 2
        for _ in range(padding_repeats):
            padding = torch.zeros_like(action_tensor)
            action_tensor = torch.cat((action_tensor, padding), dim=2)
        return action_tensor.clone()

    def check_idle(self, dataset, batch_index):
        while True:
            actions = self.action_all_rolling[batch_index].clone()
            rewards = self.reward_all_scroll_optimization[batch_index].clone()

            _, idx_min = rewards.min(dim=0)
            _, idx_max = rewards.max(dim=0)

            action_min = actions[idx_min]
            action_max = actions[idx_max]

            if torch.nonzero(action_min).size(0) == 1:
                return True  # Already optimal

            if torch.nonzero(action_max).size(0) >= 1:
                return False  # Has active agent

            # Swap action
            new_action1 = torch.zeros_like(action_min)
            new_action2 = torch.zeros_like(action_min)
            new_action1[0] = action_min[0].clone()
            new_action2[:-1] = action_min[1:].clone()

            locs = dataset.data[batch_index].to(self.opts.device)
            reward1 = get_reward(new_action1.to(self.opts.device), locs, single_agent=True, batch_operation=False)
            reward2 = get_reward(new_action2.to(self.opts.device), locs, single_agent=True, batch_operation=False)

            self.action_all_rolling[batch_index][idx_max] = new_action1
            self.action_all_rolling[batch_index][idx_min] = new_action2
            self.reward_all_scroll_optimization[batch_index][idx_max] = reward1
            self.reward_all_scroll_optimization[batch_index][idx_min] = reward2

    def epsilon_greedy_ip_cross(self, dataset):
        locs = torch.stack(dataset.data).to(self.opts.device)
        start_time = time.perf_counter()

        for batch_index in tqdm(range(self.opts.val_size), desc='IP: '):
            if batch_index in self.do_not_optimize_batch_index:
                continue

            priority_id = self.opts.agent_num
            init_index = self.reward_all_scroll_optimization[batch_index].min(dim=0)[1].item() \
                if self.start_node is None else self.start_node

            nearest_indices = find_nearest_masked(batch_index, init_index, self.distance_matrix, self.opts)
            priority_queue = [(i, (
                self.action_all_rolling[batch_index][idx].clone(),
                self.reward_all_scroll_optimization[batch_index][idx].clone(),
                idx)) for i, idx in enumerate(nearest_indices)]
            heapq.heapify(priority_queue)

            for _ in range(self.rolling_num):
                rewards = self.reward_all_scroll_optimization[batch_index]
                _, idx_min = rewards.min(dim=0)
                _, idx_max = rewards.max(dim=0)

                action_min = self.action_all_rolling[batch_index][idx_min]
                action_max = self.action_all_rolling[batch_index][idx_max]

                if torch.nonzero(action_min).size(0) == 1:
                    break

                if torch.nonzero(action_max).size(0) == 0:
                    if self.check_idle(dataset, batch_index):
                        break

                    init_index = rewards.min(dim=0)[1].item() if self.start_node is None else self.start_node
                    nearest_indices = find_nearest_masked(batch_index, init_index, self.distance_matrix, self.opts)
                    priority_queue = [(i, (
                        self.action_all_rolling[batch_index][idx].clone(),
                        rewards[idx].clone(),
                        idx)) for i, idx in enumerate(nearest_indices)]
                    heapq.heapify(priority_queue)

                neighbour_data = [pop_min(priority_queue) for _ in range(self.num_neighbour)]
                neighbour_actions = torch.stack([d[0] for d in neighbour_data])
                neighbour_rewards = torch.tensor([d[1] for d in neighbour_data])
                neighbour_indices = torch.tensor([d[2] for d in neighbour_data])
                neighbour_range = torch.tensor([batch_index] * self.num_neighbour)

                action_lengths = torch.tensor([torch.nonzero(d[0]).size(0) for d in neighbour_data])
                flat_action = torch.cat([d[0][:length] for d, length in zip(neighbour_data, action_lengths)])

                assert self.small_cross_model is not None, "Small model not loaded"
                graph_size = flat_action.shape[0]
                assert graph_size > 0, "Empty graph for re-projection"

                self.small_cross_model.graph_size = graph_size
                self.small_cross_model.agent_num = self.num_neighbour

                batch_input = torch.cat((dataset[batch_index][:1], dataset[batch_index][flat_action.cpu()]), dim=0).unsqueeze(0)
                batch_input = normalize_coord(batch_input)
                batch_input = augment(batch_input, self.opts.N_aug)
                batch_input = move_to(batch_input, self.opts.device)

                sequences, costs = self.small_cross_model.sample_many(batch_input, batch_rep=1, iter_rep=1, agent_num=self.num_neighbour, aug=self.opts.N_aug)
                sequences = sequences[1:] - self.num_neighbour
                sequences[sequences < 0] = 0

                full_action = torch.cat((torch.tensor([0]).to(self.opts.device), flat_action))
                reprojected = torch.index_select(full_action, dim=0, index=sequences)
                target_length = self.action_all_rolling.shape[-1]

                reprojected_split = split_reproject(reprojected, target_length)
                reprojected_reward = get_reward(reprojected.unsqueeze(0), locs[batch_index].unsqueeze(0), agent_num=self.num_neighbour).squeeze(0)

                accept_negative = self.rng.random() < self.epsilon

                if accept_negative or reprojected_reward.min() > neighbour_rewards.min():
                    self.positive_optimization += 1
                    self.action_all_rolling[neighbour_range, neighbour_indices] = reprojected_split
                    self.reward_all_scroll_optimization[neighbour_range, neighbour_indices] = reprojected_reward

                    best_idx = reprojected_reward.argmax() if accept_negative else reprojected_reward.argmin()
                    insert(priority_queue, 0, (
                        reprojected_split[best_idx], reprojected_reward[best_idx], neighbour_indices[best_idx]))

                    for idx in range(self.num_neighbour):
                        if idx != best_idx.item():
                            insert(priority_queue, priority_id, (
                                reprojected_split[idx], reprojected_reward[idx], neighbour_indices[idx]))
                            priority_id += 1
                else:
                    self.negative_optimization += 1
                    worst_idx = neighbour_rewards.argmin()
                    insert(priority_queue, 0, (
                        neighbour_actions[worst_idx], neighbour_rewards[worst_idx], neighbour_indices[worst_idx]))

                    for idx in range(self.num_neighbour):
                        if idx != worst_idx.item():
                            insert(priority_queue, priority_id, (
                                neighbour_actions[idx], neighbour_rewards[idx], neighbour_indices[idx]))
                            priority_id += 1

                assert len(priority_queue) == self.opts.agent_num

        duration = (time.perf_counter() - start_time) / 100
        return self.action_all_rolling, self.reward_all_scroll_optimization, duration

    def print_summary(self):
        print(f'start_node: {self.start_node}')
        print(f'{self.dont_need_optimize} graphs don\'t need optimization')
        print(f'{self.negative_optimization} negative updates')
        print(f'{self.positive_optimization} positive updates')

        reward_min_optimized, _ = self.reward_all_scroll_optimization.min(dim=-1)
        print(f'CMIP Result (Optimized): {reward_min_optimized.mean()} | CMIP Result (Original): {self.reward_CMIP.mean()}')


 
class Insert_idle:
    def __init__(
        self,
        reward_all_CMIP: torch.Tensor,
        action_all_CMIP: torch.Tensor,
        opts,
        num_iter_NCE: int = None
    ):
        self.opts = opts
        self.num_iter_NCE = num_iter_NCE
        self.reward_CMIP_mean = reward_all_CMIP.min(dim=-1)[0].mean()

        self.reward_all_CMIP_optimized = reward_all_CMIP.clone()
        padding_tensor = torch.zeros_like(action_all_CMIP)
        self.action_all_CMIP = torch.cat((action_all_CMIP, padding_tensor), dim=2)

        self.dont_need_optimize = 0
        self.negative_optimization = 0
        self.positive_optimization = 0

    def cross_insert(self, dataset):
        do_not_optimize_batch_index = []

        for batch_index in tqdm(range(self.opts.val_size), desc='Checking idle agents'):
            count = 0

            while True:
                reward_batch = self.reward_all_CMIP_optimized[batch_index]
                current_reward_min, index_min = reward_batch.min(dim=0)
                current_reward_max, index_max = reward_batch.max(dim=0)

                action_min = self.action_all_CMIP[batch_index][index_min]
                action_max = self.action_all_CMIP[batch_index][index_max]

                # Skip if agent has no real path
                if torch.count_nonzero(action_min) == 1:
                    self.dont_need_optimize += 1
                    do_not_optimize_batch_index.append(batch_index)
                    break

                # Stop condition if idle check is disabled or maximum iteration reached
                if self.num_iter_NCE is None:
                    if torch.count_nonzero(action_max) >= 1:
                        break
                else:
                    if count >= self.num_iter_NCE:
                        break
                    count += 1

                # Update: eliminate idle agent by cross-inserting
                agent1_actions = torch.zeros_like(action_min)
                agent2_actions = torch.zeros_like(action_min)

                agent1_actions[0] = action_min[0].clone()
                agent2_actions[:-1] = action_min[1:].clone()

                locs = torch.stack(dataset.data).to(self.opts.device)
                loc = locs[batch_index]

                reward1 = get_reward(agent1_actions.to(self.opts.device), loc, single_agent=True, batch_operation=False)
                reward2 = get_reward(agent2_actions.to(self.opts.device), loc, single_agent=True, batch_operation=False)

                self.action_all_CMIP[batch_index][index_max] = agent1_actions
                self.action_all_CMIP[batch_index][index_min] = agent2_actions
                self.reward_all_CMIP_optimized[batch_index][index_max] = reward1
                self.reward_all_CMIP_optimized[batch_index][index_min] = reward2

        return do_not_optimize_batch_index, self.reward_all_CMIP_optimized, self.action_all_CMIP

    def print_result(self):
        print(f"\nIdle Optimization Enabled: {self.num_iter_NCE is not None}")
        print(f"Graphs Skipped (No Optimization Needed): {self.dont_need_optimize}")
        print(f"Negative Optimizations: {self.negative_optimization}")
        print(f"Positive Optimizations: {self.positive_optimization}")

        reward_min_optimized, _ = self.reward_all_CMIP_optimized.min(dim=-1)
        print(f"Optimized CMIP Reward Mean: {reward_min_optimized.mean():.4f}")
        print(f"Original CMIP Reward Mean: {self.reward_CMIP_mean:.4f}")
