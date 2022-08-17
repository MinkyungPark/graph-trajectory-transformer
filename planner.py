import os
import json
from collections import defaultdict
import xml.etree.ElementTree as ET

import torch
import numpy as np

from dataset import load_environment

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def filter_cdf(logits, threshold):
    batch_inds = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    bins_inds = torch.arange(logits.shape[-1], device=logits.device)
    probs = logits.softmax(dim=-1)
    probs_sorted, _ = torch.sort(probs, dim=-1)
    probs_cum = torch.cumsum(probs_sorted, dim=-1)
    ## get minimum probability p such that the cdf up to p is at least `threshold`
    mask = probs_cum < threshold
    masked_inds = torch.argmax(mask * bins_inds, dim=-1)
    probs_threshold = probs_sorted[batch_inds, masked_inds]
    ## filter
    out = logits.clone()
    logits_mask = probs <= probs_threshold.unsqueeze(dim=-1)
    out[logits_mask] = -1000
    return out

class PlanConfig:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Planner():
    def __init__(self, config):
        self.dataset = config.dataset
        self.device = config.device
        self.model = config.model
        self.model.eval()

        self.discretizer = config.discretizer
        self.percentile = config.percentile
        self.value_fn = lambda x: self.discretizer.value_fn(x, self.percentile)

        self.total_dim = config.total_dim
        self.obs_dim = config.num_node * config.node_feature_dim
        self.action_dim = config.action_dim
        self.arV_dim = config.arV_dim

        self.num_virtual_tokens=config.num_virtual_tokens
        self.num_node=config.num_node
        self.node_feature_dim=config.node_feature_dim

        self.plan_freq = config.plan_freq
        self.prefix_context = config.prefix_context
        self.horizon = config.horizon
        self.beam_width = config.beam_width
        self.n_expand = config.n_expand
        self.max_context_trans = config.max_context_trans
        self.discount = config.discount

        self.k_obs = config.k_obs
        self.k_act = config.k_act
        self.cdf_obs = config.cdf_obs
        self.cdf_act = config.cdf_act

        self.node_dict = self._get_graph()
        self.VALUE_PLACEHOLDER = 1e6


    def _get_graph(self):
        d_path = self.dataset.split('-')[0]
        xml_path = os.path.join('asserts', d_path + '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        world_body = root.find('worldbody')

        node_dict = defaultdict(dict)
        i = 0
        for child in world_body.iter():
            if child.tag == 'body':
                name = child.attrib['name']
                node_dict[name] = {
                    'index': i, 
                    'joint': [joint.attrib['name'] for joint in child.findall('joint')], 
                    'node_features': []
                    }
                i += 1
        assert len(node_dict.keys()) == self.num_node
        return node_dict

    def _push_node_feature(self, obs_discrete):
        # obs_discrete : np (1, obs_dim)
        state_space_path = open(os.path.join('asserts', 'state_space.json'))
        state_space = json.load(state_space_path)[self.dataset]

        tmp = defaultdict(list)
        for state_name, obs in zip(state_space, obs_discrete.reshape(-1)):
            for body, node_values in self.node_dict.items():
                if state_name in node_values['joint']:
                    node = body
            tmp[node].append(obs)
        
        for node, obs in tmp.items():
            n_pad = self.node_feature_dim - len(obs)
            features = obs + [0] * n_pad
            self.node_dict[node]['node_features'].append(features)
    
    def _pop_node_feature(self):
        node_features = []
        
        for body, node_values in self.node_dict.items():
            node_features.append(node_values['node_features'])
        node_features = torch.tensor(node_features, dtype=torch.long, device=self.device)
        # num_node, num_trans, features -> num_trans, num_node, features
        node_features = node_features.permute(1, 0, 2)
        
        return node_features 

    def get_context_x(self, obs):
        obs_dim = obs.size
        obs_discrete = self.discretizer.discretize(obs, subslice=[0, obs_dim]) 
        self._push_node_feature(obs_discrete)
        prefix_x = self._pop_node_feature()[-self.max_context_trans:]
        if not self.prefix_context:
            prefix_x = prefix_x[-1:, :, :]
        
        return prefix_x

    @torch.no_grad()
    def sample_n(self, x, y, sample='action', cdf=None, topk=None, temperature=1.0):
        b, s, n, f = x.size()
        tmp = torch.zeros((b, 1, 1), device=self.device, dtype=torch.long)
        if sample == 'observation':
            sample_dim = sample_iter = n*f
            sampled = (tmp.repeat(1, 1, sample_dim), None, 'observation')
        elif sample == 'action':
            sample_dim = self.arV_dim
            sample_iter = self.action_dim
            sampled = (None, tmp.repeat(1, 1, sample_dim), 'action')
        else:
            sample_dim = sample_iter = self.arV_dim - self.action_dim
            sampled = (None, tmp.repeat(1, 1, sample_dim), 'reward')
        
        sampled_probs = torch.zeros(b, sample_dim, self.model.vocab_size + 1, device=x.device)

        for dim in range(sample_iter):
            logits, _ = self.model(x, y, sampled=sampled, dim=dim)

            logits = logits[:, -1] / temperature
            raw_probs = logits.softmax(dim=-1) # b, vocab+1

            if cdf is not None: # crop logits to only the top `1-cdf` percentile
                logits = filter_cdf(logits, cdf)
            
            if topk is not None: # crop logits to only the most likely `k`
                logits = top_k_logits(logits, topk)

            # probs = logits.log_softmax(dim=-1)
            probs = logits.softmax(dim=-1)
            indices = torch.multinomial(probs, num_samples=1) # (b, 1)

            if sample == 'observation':
                sampled[0][:, :, dim] = indices
            else:
                sampled[1][:, :, dim] = indices

            sampled_probs[:, dim] = raw_probs # (b, sampled_dim, vocab_size+1)

        if sample == 'observation':
            x = torch.concat([x, sampled[0].view(b, 1, n, f)], dim=1)
        elif sample == 'action':
            y = torch.concat([y, sampled[1]], dim=1) if y.size(-1) != 0 else sampled[1]
        else:
            y[:, -1:, -sample_dim:] = sampled[1]

        return x, y, sampled_probs

    @torch.no_grad()
    def beam_plan(self, prefix_x, prefix_y):
        _, n, f = prefix_x.size()
        x = prefix_x.unsqueeze(0).repeat(self.beam_width, 1, 1, 1)
        y = prefix_y.unsqueeze(0).repeat(self.beam_width, 1, 1)

        rewards = torch.zeros(self.beam_width, self.horizon + 1, device=self.device)
        discounts = self.discount ** torch.arange(self.horizon + 1, device=self.device)

        for t in range(self.horizon):
            # b : beam_width * n_expand
            x = x.repeat(self.n_expand, 1, 1, 1) # (b, s, n, f)
            y = y.repeat(self.n_expand, 1, 1) # (b, s, d)-5

            rewards = rewards.repeat(self.n_expand, 1)
            
            # sample actions
            x, y, _ = self.sample_n(x, y, sample='action', cdf=self.cdf_act, topk=self.k_act) # (bs, d)
            
            # sample reward and value estimate
            x, y, r_probs = self.sample_n(x, y, sample='reward')

            # percentile or mean of the reward
            # value distribution indstead of sampled tokens
            r_t, V_t = self.value_fn(r_probs)
            values = (rewards * discounts).sum(dim=-1)
            rewards[:, t], rewards[:, t+1] = r_t, V_t
            values, inds = torch.topk(values, self.beam_width)
            x, y, rewards = x[inds], y[inds], rewards[inds]

            if t != self.horizon - 1:
                # sample next obs
                x, y, _ = self.sample_n(x, y, sample='observation', cdf=self.cdf_obs, topk=self.k_obs)
        
        y = y[:, -self.horizon:]
        argmax = values.argmax()
        best_sequence = y[argmax]
        return best_sequence # (horizon, arV_dim)

    def plan(self):
        env = load_environment(self.dataset)
        observation = env.reset()
        context_y = torch.tensor([[]], dtype=torch.long, device=self.device)
        total_reward = 0.0
        T = env.max_episode_steps

        for t in range(T):
            if t % self.plan_freq == 0:
                context_x = self.get_context_x(observation) # s, n, f
                sampled_a = self.beam_plan(context_x, context_y) # (horizon, arV_dim)
            else:
                sampled_a = sampled_a[1:]
            
            a_seq_recon = self.discretizer.reconstruct(sampled_a, subslice=(-self.arV_dim, None))
            action = a_seq_recon[:, :self.action_dim][0]
            
            next_observation, reward, terminal, _ = env.step(action)
            total_reward += reward
            score = env.get_normalized_score(total_reward)
            
            rew_val = np.array([reward, self.VALUE_PLACEHOLDER])
            new_arV = np.concatenate([action, rew_val])
            new_arV = self.discretizer.discretize(new_arV, subslice=(-self.arV_dim, None))
            new_arV = torch.tensor(new_arV, dtype=torch.long, device=self.device)

            if context_y.size(-1) == 0:
                context_y = new_arV
            else:
                context_y = torch.concat([context_y, new_arV], dim=0)
            context_y = context_y[-self.max_context_trans:]

            print(
                f'[ plan ] timestep: {t} / {T} | reward: {reward:.2f} | total Reward: {total_reward:.2f} | normalized_score: {score:.4f} | \n'
            )

            if terminal: break

            observation = next_observation

        return score, t, total_reward, terminal, context_x, context_y
