import os
import gc
import dill
import json
import argparse
import datetime

import torch

from gpt import GPT, GPTConfig
from trainer import Trainer
from dataset import MujocoGraphDataset
from utils import set_seed, check_dir, Timer

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
parser.add_argument('--max_path_length', type=int, default=1000)
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--discount', type=float, default=0.99)
parser.add_argument('--n_layer', type=int, default=4)
parser.add_argument('--n_head', type=int, default=4)

parser.add_argument('--n_epochs_ref', type=int, default=50)
parser.add_argument('--n_saves', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda')

parser.add_argument('--n_embd', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=6e-4)
parser.add_argument('--lr_decay', type=bool, default=True)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--embd_pdrop', type=float, default=0.1)
parser.add_argument('--resid_pdrop', type=float, default=0.1)
parser.add_argument('--attn_pdrop', type=float, default=0.1)

parser.add_argument('--step', type=int, default=1)
parser.add_argument('--subsampled_sequence_length', type=int, default=10)
parser.add_argument('--termination_penalty', type=int, default=-100)

parser.add_argument('--discretizer', type=str, default='QuantileDiscretizer')
parser.add_argument('--action_weight', type=int, default=5)
parser.add_argument('--reward_weight', type=int, default=1)
parser.add_argument('--value_weight', type=int, default=1)

args = parser.parse_args()
set_seed(args.seed)
savepath = check_dir(os.path.join('logs', args.dataset, f'base_{args.seed}'))


################# Dataset #################

sequence_length = args.subsampled_sequence_length * args.step

dataset = MujocoGraphDataset(
    dataset=args.dataset,
    N=args.N,
    penalty=args.termination_penalty,
    sequence_length=sequence_length,
    step=args.step,
    discount=args.discount,
    max_path_length=args.max_path_length,
)
# save discretizer
torch.save(dataset.get_discretizer(), os.path.join(savepath, "discretizer.pt"))

################# Model GPT #################
num_virtual_token=1
num_node = 7
node_feature_dim = 5

trans_dim = (dataset.joined_dim - dataset.s_dim) + (num_node + num_virtual_token)*node_feature_dim
block_size = args.subsampled_sequence_length * trans_dim - 1
mconf = GPTConfig(
    vocab_size=args.N, 
    n_layer=args.n_layer, 
    n_head=args.n_head, 
    n_embd=args.n_embd * args.n_head,
    action_weight=args.action_weight, 
    reward_weight=args.reward_weight, 
    value_weight=args.value_weight,
    embd_pdrop=args.embd_pdrop, 
    resid_pdrop=args.resid_pdrop, 
    attn_pdrop=args.attn_pdrop,
    block_size=block_size, 
    obs_dim=dataset.s_dim, 
    action_dim=dataset.a_dim, 
    trans_dim=trans_dim,
    discount=dataset.discount,
    # for graphormerEncoder
    num_virtual_token=num_virtual_token,
    num_node=num_node,
    node_feature_dim=node_feature_dim,
    )


model = GPT(config=mconf)
mconf.seed = args.seed
dill.dump(mconf, open(savepath + '/model_config.dill', 'wb'))
model.to(args.device)

################# Trainer #################

warmup_tokens = len(dataset) * block_size ## number of tokens seen per epoch
final_tokens = 20 * warmup_tokens

trainer = Trainer(
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    betas=(0.9, 0.95),
    grad_norm_clip=1.0,
    weight_decay=0.1, # only applied on matmul weights
    lr_decay=args.lr_decay,
    warmup_tokens=warmup_tokens,
    final_tokens=final_tokens,
    num_workers=4,
    device=args.device
)

n_epochs = int(1e6 / len(dataset) * args.n_epochs_ref)
save_freq = int(n_epochs // args.n_saves)

################# Training ! #################

timer, tr_stt = Timer(), datetime.datetime.now()
for epoch in range(n_epochs):
    print(f'\nEpoch: {epoch} / {n_epochs} | {args.dataset} ')
    trainer.train(model, dataset)
    gc.collect()
    torch.cuda.empty_cache()
    
    # save_epoch = (epoch + 1) // save_freq * save_freq
    statepath = os.path.join(savepath, f'state_{epoch}.pt')
    state = model.state_dict()
    torch.save(state, statepath)
    print(f'Saving model to {statepath}')


time_log = f'{args.dataset} : training start-time: {tr_stt} | training end-time : {datetime.datetime.now()} | timer: {timer():.2f} \n'
with open('./logs/time_log.txt', 'a') as f: 
    f.write(time_log)

# argument save
json.dump(vars(args), open(savepath + '/args_info.json', 'w'), indent=2, sort_keys=True)