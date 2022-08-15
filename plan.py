import os
import argparse
import dill
import json

import torch
import numpy as np
from eval_gpt import GPT

from dataset import load_environment
from utils import check_dir, set_seed, Timer
from search import plan

parser = argparse.ArgumentParser()

parser.add_argument('--loadpath', type=str, default='logs')
parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
parser.add_argument('--model_path', type=str, default='base')
parser.add_argument('--model_epoch', type=int, default=0)
parser.add_argument('--device', type=str, default='cpu')
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
parser.add_argument('--max_context_transitions', type=int, default=5)
parser.add_argument('--prefix_context', type=bool, default=True)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()
loadpath = os.path.join(args.loadpath, args.dataset, f'{args.model_path}_{args.seed}')

# get the last model
# file_list = [f for f in os.listdir(loadpath + '/models/') if 'state' in f]
file_list = [f for f in os.listdir(loadpath) if 'state' in f]
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

print(f'Dataset [{args.dataset}] / Model : [{args.model_path}/model_{args.model_epoch}.pt]')

# Load Discretizer & Model
discretizer = torch.load(os.path.join(loadpath, "discretizer.pt"), map_location=args.device)

mconf = dill.load(open(loadpath + '/model_config.dill', 'rb'))
mconf.dataset = args.dataset
set_seed(mconf.seed)

model = GPT(mconf)
# model.load_state_dict(torch.load(loadpath + '/models/state_'+ str(args.model_epoch) +'.pt'))
model.load_state_dict(torch.load(loadpath + '/state_'+ str(args.model_epoch) +'.pt'))
model.to(torch.device(args.device))

env = load_environment(args.dataset)

# s_dim = mconf.obs_dim
s_dim = 8*5
a_dim = mconf.action_dim
mconf.discount = 0.99
timer = Timer()
results, trajs = [], []
for _ in range(args.num_eval):
        score, t, total_reward, terminal, context = plan(
                env, model,
                s_dim, a_dim,
                args.plan_freq, 
                discretizer,
                args.prefix_context, 
                args.horizon, 
                args.beam_width, 
                args.n_expand,
                args.max_context_transitions,
                discount=mconf.discount, 
                k_obs=args.k_obs, 
                k_act=args.k_act, 
                cdf_obs=args.cdf_obs, 
                cdf_act=args.cdf_act, 
                percentile=args.percentile, 
                device=args.device
        )
        results.append((score, t, total_reward, terminal))
        trajs.append(context)

# Logs !
d_name = f'plan_{args.horizon}_{args.beam_width}_{args.model_epoch}' # horizon, beam_width, model_epoch
result_path = check_dir(os.path.join(loadpath, d_name))

print(f'{args.num_eval} of plan time: {timer():.2f}')

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

# argument save
json.dump(vars(args), open(result_path + '/args_info.json', 'w'), indent=2, sort_keys=True)