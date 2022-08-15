import os
import dill
import json
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
from tqdm import tqdm

import gym
import numpy as np
import torch
from torch_geometric.data import Data

from preprocessing import segment, QuantileDiscretizer
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
import algos


# ---------------- d4rl load ---------------- #
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

def qlearning_dataset_with_timeouts(env, dataset=None, terminate_on_end=False, **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    realdone_ = []

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        realdone_bool = bool(dataset['terminals'][i])
        final_timestep = dataset['timeouts'][i]

        if i < N - 1:
            done_bool += dataset['timeouts'][i] #+1]

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue  
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        realdone_.append(realdone_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_)[:,None],
        'terminals': np.array(done_)[:,None],
        'realterminals': np.array(realdone_)[:,None],
    }

def load_environment(name):
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env


# ---------------- Dataset ---------------- #
class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, sequence_length=250, step=10, discount=0.99, max_path_length=1000, penalty=None):
        print(f'[ datasets/sequence ] Sequence length: {sequence_length} | Step: {step} | Max path length: {max_path_length}')
        self.dataset = dataset
        self.env = env = load_environment(dataset) if type(dataset) is str else dataset
        self.sequence_length = sequence_length
        self.step = step
        self.max_path_length = max_path_length
        
        print(f'[ dataset/sequence ] Loading...', end=' ', flush=True)
        dataset = qlearning_dataset_with_timeouts(env.unwrapped, terminate_on_end=True)
        print('✓')

        observations = dataset['observations']
        actions = dataset['actions']
        next_observations = dataset['next_observations']
        rewards = dataset['rewards']
        terminals = dataset['terminals']
        realterminals = dataset['realterminals']

        self.observations_raw = observations
        self.actions_raw = actions
        self.next_observations_raw = next_observations
        self.joined_raw = np.concatenate([observations, actions], axis=-1)
        self.rewards_raw = rewards
        self.terminals_raw = terminals
        
        self.s_dim = observations.shape[1]
        self.a_dim = actions.shape[1]
        
        ## terminal penalty
        if penalty is not None:
            terminal_mask = realterminals.squeeze()
            self.rewards_raw[terminal_mask] = penalty
            self.termination_penalty = penalty

        ## segment
        print(f'[ datasets/sequence ] Segmenting...', end=' ', flush=True)
        self.joined_segmented, self.termination_flags, self.path_lengths = segment(self.joined_raw, terminals, max_path_length)
        self.rewards_segmented, *_ = segment(self.rewards_raw, terminals, max_path_length)
        print('✓')

        self.discount = discount
        self.discounts = (discount ** np.arange(self.max_path_length))[:,None]

        ## [ n_paths x max_path_length x 1 ]
        self.values_segmented = np.zeros(self.rewards_segmented.shape)

        for t in range(max_path_length):
            ## [ n_paths x 1 ]
            V = (self.rewards_segmented[:,t+1:] * self.discounts[:-t-1]).sum(axis=1)
            self.values_segmented[:,t] = V

        ## add (r, V) to `joined`
        values_raw = self.values_segmented.squeeze(axis=-1).reshape(-1)
        values_mask = ~self.termination_flags.reshape(-1)
        self.values_raw = values_raw[values_mask, None]
        self.joined_raw = np.concatenate([self.joined_raw, self.rewards_raw, self.values_raw], axis=-1)
        self.joined_segmented = np.concatenate([self.joined_segmented, self.rewards_segmented, self.values_segmented], axis=-1)

        ## get valid indices
        indices = []
        for path_ind, length in enumerate(self.path_lengths):
            end = length - 1
            for i in range(end):
                indices.append((path_ind, i, i+sequence_length))

        self.indices = np.array(indices)
        self.joined_dim = self.joined_raw.shape[1]

        ## pad trajectories
        n_trajectories, _, joined_dim = self.joined_segmented.shape
        self.joined_segmented = np.concatenate([
            self.joined_segmented,
            np.zeros((n_trajectories, sequence_length-1, joined_dim)),
        ], axis=1)
        self.termination_flags = np.concatenate([
            self.termination_flags,
            np.ones((n_trajectories, sequence_length-1), dtype=np.bool_),
        ], axis=1)

    def __len__(self):
        return len(self.indices)


# ---------------- Graph Dataset ---------------- #

def preprocess_item(item, num_virtual_tokens=1):
    '''
    x : (trans_len, n_node, feature_dim)
    edge_index : (2, num_edge)
    '''
    N, F = item.x.size()[-2:]
    num_virtual_tokens = num_virtual_tokens
    edge_index = item.edge_index
    edge_attr = torch.zeros((edge_index.shape[1]), dtype=torch.long)[:, None] # (num_edge, 1)

    adj_orig = torch.zeros([N, N], dtype=torch.bool)
    adj_orig[edge_index[0, :], edge_index[1, :]] = True

    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        edge_attr + 1
    )  # (n_nodes, n_nodes, 1) if connect 1 else 0

    shortest_path_result, path = algos.floyd_warshall(
        adj_orig.numpy()
    )  # [n_nodesxn_nodes, n_nodesxn_nodes]
    max_dist = np.amax(shortest_path_result)
    
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    rel_pos = torch.from_numpy((shortest_path_result)).long() 
    attn_bias = torch.zeros(
        [N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.float
    )  # with graph token

    adj = torch.zeros(
        [N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.bool
    )
    adj[edge_index[0, :], edge_index[1, :]] = True

    for i in range(num_virtual_tokens):
        adj[N + i, :] = True
        adj[:, N + i] = True

    # for i in range(N + num_virtual_tokens):
    #     for j in range(N + num_virtual_tokens):
    #         val = True if random.random() < 0.3 else False
    #         adj[i, j] = adj[i, j] or val

    item.adj = adj # (N+1, N+1) the last node is virtual node which is connected to all nodes
    item.attn_bias = attn_bias # (N+1, N+1) zeros
    item.attn_edge_type = attn_edge_type # (N, N, 1) if connect 1 else 0
    item.rel_pos = rel_pos # (N, N) shortest distance, unreachable 100
    item.in_degree = adj_orig.long().sum(dim=1).view(-1) # (,N)
    item.out_degree = adj_orig.long().sum(dim=0).view(-1) # (,N)
    item.edge_input = torch.from_numpy(edge_input).long() # # (N, N, 100(unreachable distance), 1)

    return item


class MujocoGraphDataset(SequenceDataset):

    def __init__(self, N=100, cached_data_name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_path = self.dataset.split('-')[0]
        self.vocab_size = N # vocab size
        self.num_virtual_tokens = 1
        
        state_space_path = open(os.path.join('asserts', 'state_space.json'))
        state_space = json.load(state_space_path)
        self.state_space = state_space[self.dataset]

        xml_path = os.path.join('asserts', self.d_path + '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        self.world_body = root.find('worldbody')
        self.actuator = root.find('actuator')
        self.discretizer = QuantileDiscretizer(self.joined_raw, self.vocab_size)

        self.cached_data_name = cached_data_name
        self.graph_dataset = self._get_graph_dataset()
        self.num_node, self.node_feature_dim = self.graph_dataset.x.size()[-2:]
        self._save_dataset()

    def _get_graph_dataset(self):
        # joined_segmneted (1000, 1009, 25) 
        # 1000 episode, each epi padded transitions(seq_len), each trans dim 25
        num_epi, len_seq, trans_dim = self.joined_segmented.shape
        print('[ discretization ] All dataset discretizing...')
        joined_discrete = [self.discretizer.discretize(epi) for epi in tqdm(self.joined_segmented)]
        joined_discrete = torch.tensor(joined_discrete, device='cpu', dtype=torch.long).contiguous()
        joined_discrete[self.termination_flags] = self.vocab_size

        # ----- for states to graph ----- # 
        all_observations = joined_discrete[:, :, :self.s_dim]
        all_observations = all_observations.view(num_epi * len_seq, self.s_dim)

        # graph data structure
        node_dict = defaultdict(dict)
        i = 0
        for child in self.world_body.iter():
            if child.tag == 'body':
                name = child.attrib['name']
                node_dict[name] = {
                    'index': i, 
                    'joint': [joint.attrib['name'] for joint in child.findall('joint')], 
                    'direction': [joint.attrib['name'] for joint in child.findall('body')],
                    'node_features': []
                    }
                i += 1
        num_node = len(node_dict.keys())

        # all_observations (num_epi * len_seq, obs_dim)
        for state_name, each_axis_obs in zip(self.state_space, all_observations.transpose(1, 0)):
            for body, node_values in node_dict.items():
                if state_name in node_values['joint']:
                    node = body
            # (node_feature_dim, num_epi * len_seq)
            node_dict[node]['node_features'].append(each_axis_obs.tolist())
            
        
        max_feature_dim = max([len(v['node_features']) for v in node_dict.values()])
        node_features = np.zeros((num_epi * len_seq, num_node, max_feature_dim))

        e_start, e_end = [], []
        for body, node_values in node_dict.items():
            # edge sindex
            n_i = node_values['index']
            n_adj = len([node_values['direction']])
            if n_adj > 0:
                for adj in node_values['direction']:
                    e_start.append(n_i)
                    e_end.append(node_dict[adj]['index'])
            
            features = torch.tensor(node_values['node_features'], dtype=torch.long)
            # node feature (total_transition, num_node, node_feature_dim)
            node_features[:, n_i, :features.size(0)] = features.transpose(1, 0)

        x = torch.from_numpy(node_features).long()
        y = joined_discrete[:, :, self.s_dim:].view(num_epi * len_seq, -1)
        edge_index = torch.tensor([e_start, e_end], dtype=torch.long)
        
        x = x.view(num_epi, len_seq, num_node, max_feature_dim)
        y = y.view(num_epi, len_seq, -1)
        data_graphs = Data(x=x, edge_index=edge_index, y=y)
        data_graphs = preprocess_item(data_graphs, self.num_virtual_tokens)

        return data_graphs

    def _save_dataset(self):
        save_data = {
            "discretizer" : self.discretizer,
            "termination_penalty" : self.termination_penalty,
            "discount" : self.discount,
            "sequence_length" : self.sequence_length,
            "step" : self.step,
            "max_path_length": self.max_path_length,
            "indices" : self.indices,
            "joined_dim" : self.joined_dim,
            "s_dim" : self.s_dim,
            "a_dim" : self.a_dim,
            "vocab_size" : self.vocab_size,
            "num_node" : self.num_node, 
            "node_feature_dim" : self.node_feature_dim,
            "num_virtual_tokens" : self.num_virtual_tokens,
            "graph_dataset" : self.graph_dataset
        }

        with open('./asserts/' + self.cached_data_name, 'wb') as f:
            dill.dump(save_data, f)
        print(f'## Cached Dataset Saved /asserts/{self.cached_data_name}...')


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, cached_dataset):
        super().__init__()
        for k, v in cached_dataset.items():
            setattr(self, k, v)

    def get_graph_attr(self, device='cpu'):
        GraphAttr = namedtuple('GraphAttr', 'adj attn_bias attn_edge_type rel_pos in_degree out_degree edge_input')
        graph_attr = GraphAttr(
                            self.graph_dataset.adj.to(device),
                            self.graph_dataset.attn_bias.to(device),
                            self.graph_dataset.attn_edge_type.to(device),
                            self.graph_dataset.rel_pos.to(device),
                            self.graph_dataset.in_degree.to(device),
                            self.graph_dataset.out_degree.to(device),
                            self.graph_dataset.edge_input.to(device),
                    )
        return graph_attr

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx] # epi_num, start_timestep, end_timestep
        x = self.graph_dataset.x[path_ind, start_ind:end_ind:self.step] # 10, 7, 5
        y = self.graph_dataset.y[path_ind, start_ind:end_ind:self.step] # 10, 8

        # make target
        s, n, f = x.size()
        padded_x = torch.cat([x.view(s, n*f), torch.zeros(s, f)], dim=-1)
        target = torch.cat([padded_x, y], dim=-1)
        target = target.long().view(s, -1)
        target = target[:, 1:]
        y = y[:, :-1]

        return x, y, target