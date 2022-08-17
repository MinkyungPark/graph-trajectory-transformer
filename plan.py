import os
import argparse
import dill
import json

import torch
import numpy as np
from gpt import GPT

from utils import check_dir, set_seed, Timer
from planner import PlanConfig, Planner

parser = argparse.ArgumentParser()

parser.add_argument('--loadpath', type=str, default='logs')
parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
parser.add_argument('--model_path', type=str, default='base')
parser.add_argument('--model_epoch', type=int, default=-1)
parser.add_argument('--device', type=str, default='cuda:3')
parser.add_argument('--num_eval', type=int, default=10)

parser.add_argument('--plan_freq', type=int, default=1)
parser.add_argument('--horizon', type=int, default=5) # 15
parser.add_argument('--beam_width', type=int, default=32) # 32
parser.add_argument('--n_expand', type=int, default=2)
parser.add_argument('--k_obs', type=int, default=1)
parser.add_argument('--k_act', type=int, default=None)
parser.add_argument('--cdf_obs', type=int, default=None)
parser.add_argument('--cdf_act', type=int, default=0.6)
parser.add_argument('--percentile', type=str, default='mean')
parser.add_argument('--max_context_trans', type=int, default=5)
parser.add_argument('--prefix_context', type=bool, default=True)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()
loadpath = os.path.join(args.loadpath, args.dataset, f'{args.model_path}_{args.seed}')

# -------------------- get model epoch -------------------- #
file_list = [f for f in os.listdir(loadpath + '/models') if 'state' in f]
epoch_list = []
for f in file_list:
    splited = f.split('_')[1]
    epoch_list.append(int(splited.split('.')[0]))

if args.model_epoch == -1:
        args.model_epoch = sorted(epoch_list)[-1]
elif args.model_epoch == 0:
        args.model_epoch = sorted(epoch_list)[-1] // 2
else:
        args.model_epoch = args.model_epoch
print(f'Dataset : [{args.dataset}] / Model : [{args.model_path}/model_{args.model_epoch}.pt]')

# -------------------- Load Model & Config -------------------- #
mconf = dill.load(open(loadpath + '/model_config.dill', 'rb'))
mconf.dataset = args.dataset
set_seed(mconf.seed)

model = GPT(mconf)
model.load_state_dict(torch.load(loadpath + '/models/state_'+ str(args.model_epoch) +'.pt'))
model.to(torch.device(args.device))

# -------------------- Discretizer -------------------- #
cached_data_name = f'{args.dataset}_graph_dataset.dill'
with open('./asserts/' + cached_data_name, 'rb') as f:
    cached_dataset = dill.load(f)
discount = cached_dataset["discount"]
discretizer = cached_dataset["discretizer"]
del cached_dataset

# -------------------- Planner Config -------------------- #
pconfig = PlanConfig(
        dataset=args.dataset,
        model=model,
        total_dim=mconf.total_dim,
        action_dim=mconf.action_dim,
        arV_dim=mconf.arV_dim,
        num_virtual_tokens=mconf.num_virtual_tokens,
        num_node=mconf.num_node,
        node_feature_dim=mconf.node_feature_dim,
        plan_freq=args.plan_freq,
        discretizer=discretizer,
        prefix_context=args.prefix_context,
        horizon=args.horizon,
        beam_width=args.beam_width,
        n_expand=args.n_expand,
        max_context_trans=args.max_context_trans,
        discount=discount,
        k_obs=args.k_obs, 
        k_act=args.k_act, 
        cdf_obs=args.cdf_obs, 
        cdf_act=args.cdf_act, 
        percentile=args.percentile, 
        device=args.device
)

# -------------------- Plan! -------------------- #
timer = Timer()
results, trajs = [], []
for _ in range(args.num_eval):
        planner = Planner(pconfig)
        score, t, total_reward, terminal, context = planner.plan()
        results.append((score, t, total_reward, terminal))
        trajs.append(context)

# -------------------- Logs! -------------------- #
d_name = f'plan_{args.horizon}_{args.beam_width}_{args.model_epoch}'
result_path = check_dir(os.path.join(loadpath, d_name))

print(f'{args.num_eval} of plan time: {timer():.2f}')

# for histogram
dill.dump(trajs, open(result_path + '/generated_trajectories.dill', 'wb')) 

json_data = {'total_return': [], 'normalized_score': [], 'step': [], 'terminal': [], 'gpt_epoch': [], 
            'reward_mean': 0, 'reward_std': 0, 'score_mean': 0, 'score_std':0}
for (score, t, total_reward, terminal) in results:
    json_data['normalized_score'].append(score)
    json_data['step'].append(t)
    json_data['total_return'].append(total_reward)
    json_data['terminal'].append(terminal)
    json_data['gpt_epoch'].append(args.model_epoch)

reward_mean, reward_std = np.mean(json_data['total_return']), np.std(json_data['total_return'])
score_mean, score_std = np.mean(json_data['normalized_score']), np.std(json_data['normalized_score'])

json_data['reward_mean'] = reward_mean
json_data['reward_std'] = reward_std
json_data['score_mean'] = score_mean
json_data['score_std'] = score_std

print(f"Evalution on {args.dataset}")
print(f"Mean reward: {reward_mean} ± {reward_std}")
print(f"Mean score: {score_mean} ± {score_std}")
json.dump(json_data, open(result_path + '/result.json', 'w'), indent=2, sort_keys=True)

# argument save, just in case
json.dump(vars(args), open(result_path + '/args_info.json', 'w'), indent=2, sort_keys=True)