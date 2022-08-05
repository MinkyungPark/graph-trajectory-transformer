import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict

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
def convert_to_single_emb(x, feature_num, offset=100, s=0):
    feature_offset = torch.arange(s, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    '''
    x : (seq_len/step, n_node, feature_dim)
    edge_index : (2, num_edge)
    '''
    num_virtual_tokens = 1
    # edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x

    # if edge_attr is None:
    #     edge_attr = torch.zeros((edge_index.shape[1]), dtype=torch.long)
    edge_index, x = item.edge_index[0], item.x
    edge_attr = torch.zeros((edge_index.shape[1]), dtype=torch.long)

    S = x.size(0)
    N = x.size(1)
    F = x.size(2)

    # x = convert_to_single_emb(x, F)

    # node adj matrix [N, N] bool
    adj_orig = torch.zeros([N, N], dtype=torch.bool)
    adj_orig[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here

    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr, 1) + 1
    )  # [n_nodes, n_nodes, 1]

    shortest_path_result, path = algos.floyd_warshall(
        adj_orig.numpy()
    )  # [n_nodesxn_nodes, n_nodesxn_nodes]
    max_dist = np.amax(shortest_path_result)
    
    # (N, N, 510, 1)
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

    # combine
    item.x = x
    item.adj = adj.repeat(S, 1, 1)
    item.attn_bias = attn_bias.repeat(S, 1, 1)
    item.attn_edge_type = attn_edge_type.repeat(S, 1, 1, 1)
    item.rel_pos = rel_pos.repeat(S, 1, 1)
    item.in_degree = adj_orig.long().sum(dim=1).view(-1).repeat(S, 1)
    item.out_degree = adj_orig.long().sum(dim=0).view(-1).repeat(S, 1)
    item.edge_input = torch.from_numpy(edge_input).long().repeat(S, 1, 1, 1, 1)

    return item


class MujocoGraphDataset(SequenceDataset):

    def __init__(self, N=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_path = self.dataset.split('-')[0]
        
        state_space_path = open(os.path.join('asserts', 'state_space.json'))
        state_space = json.load(state_space_path)
        self.state_space = state_space[self.dataset]

        xml_path = os.path.join('asserts', self.d_path + '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()

        self.world_body = root.find('worldbody')
        self.actuator = root.find('actuator')
        self.num_node = self._get_num_node()
        self.N = N
        self.discretizer = QuantileDiscretizer(self.joined_raw, N)

    def _set_node_dict(self):
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
        return node_dict

    def _get_num_node(self):
        node_dict = self._set_node_dict()
        return len(node_dict.keys())

    def get_discretizer(self):
        return self.discretizer

    def get_node_features(self, sample):
        sample_node_dict = self._set_node_dict()
        for state_name, value in zip(self.state_space, sample):
            for body, v in sample_node_dict.items():
                if state_name in v['joint']:
                    node = body
            sample_node_dict[node]['node_features'].append(value)

        max_feature_len = max([len(v['node_features']) for v in sample_node_dict.values()])
        node_features = np.zeros((self.num_node, max_feature_len))

        # make graph data
        e_start, e_end = [], []
        for body, v in sample_node_dict.items():
            n_i = v['index']
            n_adj = len([v['direction']])
            if n_adj > 0:
                for adj in v['direction']:
                    e_start.append(n_i)
                    e_end.append(sample_node_dict[adj]['index'])
            n_features = v['node_features']
            node_features[n_i, :len(n_features)] = n_features

        return e_start, e_end, node_features


    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]
        # joined_segmented (1000, 1009, 25) 1000개 에피소드, 각 에피소드 transitions(seq_len) 1009로 패딩, 각 transition 25
        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]
        terminations = self.termination_flags[path_ind, start_ind:end_ind:self.step]
        joined_discrete = self.discretizer.discretize(joined)
        
        ## replace with termination token if the sequence has ended
        assert (joined[terminations] == 0).all(), \
                f'Everything after termination should be 0: {path_ind} | {start_ind} | {end_ind}'
        
        joined_discrete[terminations] = self.N
        joined_discrete = torch.tensor(joined_discrete, device='cpu', dtype=torch.long).contiguous()

        # ----- for states ----- # 
        edege_starts, edege_ends, n_features = [], [], []
        for sample in joined_discrete[:, :self.s_dim]:
            st_node, end_node, n_feature = self.get_node_features(sample)
            edege_starts.append(st_node)
            edege_ends.append(end_node)
            n_features.append(n_feature)

        # (seq_len/steps X n_nodes X each_node_feature_dim)
        node_features = np.stack(n_features)
        x_s = np.array([node_feature for node_feature in node_features])
        batch_x = torch.from_numpy(x_s).long()
        batch_edge_index = torch.tensor([[e_s, e_e] for e_s, e_e in zip(edege_starts, edege_ends)], dtype=torch.long)

        data_graphs = Data(x=batch_x, edge_index=batch_edge_index, y=joined_discrete[:, self.s_dim:])
        data_graphs = preprocess_item(data_graphs)

        return data_graphs


def test():
    dataset = MujocoGraphDataset(
        env='halfcheetah-medium-v2',
        sequence_length=1,
    )
    sample = dataset.__getitem__(0)

# test()