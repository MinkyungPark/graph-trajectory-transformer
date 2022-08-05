import torch
import numpy as np
from utils import to_torch


REWARD_DIM = VALUE_DIM = 1
VALUE_PLACEHOLDER = 1e6

def make_prefix(discretizer, context, obs, device, prefix_context=True):
    observation_dim = obs.size
    obs_discrete = discretizer.discretize(obs, subslice=[0, observation_dim])
    obs_discrete = to_torch(obs_discrete, dtype=torch.long, device=device)
    
    if prefix_context:
        prefix = torch.cat(context + [obs_discrete], dim=-1)
    else:
        prefix = obs_discrete

    return prefix

def extract_actions(x, observation_dim, action_dim, t=None):
    assert x.shape[1] == observation_dim + action_dim + 2
    actions = x[:, observation_dim:observation_dim+action_dim]
    if t is not None:
        return actions[t]
    else:
        return actions

def update_context(context, discretizer, observation, action, reward, max_context_transitions, device):
    '''
        context : list of transitions
            [ tensor( transition_dim ), ... ]
    '''
    ## use a placeholder for value because input values are masked out by model
    rew_val = np.array([reward, VALUE_PLACEHOLDER])
    transition = np.concatenate([observation, action, rew_val])

    ## discretize transition and convert to torch tensor
    transition_discrete = discretizer.discretize(transition)
    transition_discrete = to_torch(transition_discrete, dtype=torch.long, device=device)

    ## add new transition to context
    context.append(transition_discrete)

    ## crop context if necessary
    context = context[-max_context_transitions:]

    return context

@torch.no_grad()
def beam_plan(
    model, value_fn, x,
    n_steps, beam_width, n_expand,
    observation_dim, action_dim,
    discount=0.99, max_context_transitions=None,
    k_obs=None, k_act=None, k_rew=1,
    cdf_obs=None, cdf_act=None, cdf_rew=None,
):
    '''
        x : tensor[ 1 x input_sequence_length ]
    '''

    inp = x.clone()

    # convert max number of transitions to max number of tokens
    transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
    max_block = max_context_transitions * transition_dim - 1 if max_context_transitions else None

    ## pass in max numer of tokens to sample function
    sample_kwargs = {
        'max_block': max_block,
        'crop_increment': transition_dim,
    }

    ## repeat input for search
    x = x.repeat(beam_width, 1)

    ## construct reward and discount tensors for estimating values
    rewards = torch.zeros(beam_width, n_steps + 1, device=x.device)
    discounts = discount ** torch.arange(n_steps + 1, device=x.device)


    for t in range(n_steps):
        ## repeat everything by `n_expand` before we sample actions
        x = x.repeat(n_expand, 1)
        rewards = rewards.repeat(n_expand, 1)

        ## sample actions
        x, _ = sample_n(model, x, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)

        ## sample reward and value estimate
        x, r_probs = sample_n(model, x, REWARD_DIM + VALUE_DIM, topk=k_rew, cdf=cdf_rew, **sample_kwargs)

        ## optionally, use a percentile or mean of the reward and
        ## value distributions instead of sampled tokens
        r_t, V_t = value_fn(r_probs)

        ## update rewards tensor
        rewards[:, t] = r_t
        rewards[:, t+1] = V_t

        ## estimate values using rewards up to `t` and terminal value at `t`
        values = (rewards * discounts).sum(dim=-1)

        ## get `beam_width` best actions
        values, inds = torch.topk(values, beam_width)

        ## index into search candidates to retain `beam_width` highest-reward sequences
        x = x[inds]
        rewards = rewards[inds]

        ## sample next observation (unless we have reached the end of the planning horizon)
        if t < n_steps - 1:
            x, _ = sample_n(model, x, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)

    ## [ batch_size x (n_context + n_steps) x transition_dim ]
    x = x.view(beam_width, -1, transition_dim)

    ## crop out context transitions
    ## [ batch_size x n_steps x transition_dim ]
    x = x[:, -n_steps:]

    ## return best sequence
    argmax = values.argmax()
    best_sequence = x[argmax]

    return best_sequence

#-------------------------------- helper functions --------------------------------#

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

def round_to_multiple(x, N):
    '''
        Rounds `x` up to nearest multiple of `N`.

        x : int
        N : int
    '''
    pad = (N - x % N) % N
    return x + pad

def sort_2d(x):
    '''
        x : [ M x N ]
    '''
    M, N = x.shape
    x = x.view(-1)
    x_sort, inds = torch.sort(x, descending=True)

    rows = inds // N
    cols = inds % N

    return x_sort, rows, cols


#-------------------------------- forward pass --------------------------------#

def forward(model, x, max_block=None, allow_crop=True, crop_increment=None, **kwargs):
    '''
        A wrapper around a single forward pass of the transformer.
        Crops the input if the sequence is too long.

        x : tensor[ batch_size x sequence_length ]
    '''
    model.eval()

    block_size = min(model.get_block_size(), max_block or np.inf)

    if x.shape[1] > block_size:
        assert allow_crop, (
            f'[ search/sampling ] input size is {x.shape} and block size is {block_size}, '
            'but cropping not allowed')

        ## crop out entire transition at a time so that the first token is always s_t^0
        n_crop = round_to_multiple(x.shape[1] - block_size, crop_increment)
        assert n_crop % crop_increment == 0
        x = x[:, n_crop:]

    logits, _ = model(x, **kwargs)

    return logits

def get_logp(model, x, temperature=1.0, topk=None, cdf=None, **forward_kwargs):
    '''
        x : tensor[ batch_size x sequence_length ]
    '''
    ## [ batch_size x sequence_length x vocab_size ]
    logits = forward(model, x, **forward_kwargs)

    ## pluck the logits at the final step and scale by temperature
    ## [ batch_size x vocab_size ]
    logits = logits[:, -1] / temperature

    ## optionally crop logits to only the top `1 - cdf` percentile
    if cdf is not None:
        logits = filter_cdf(logits, cdf)

    ## optionally crop logits to only the most likely `k` options
    if topk is not None:
        logits = top_k_logits(logits, topk)

    ## apply softmax to convert to probabilities
    logp = logits.log_softmax(dim=-1)

    return logp


#-------------------------------- sampling --------------------------------#

def sample(model, x, temperature=1.0, topk=None, cdf=None, **forward_kwargs):
    '''
        Samples from the distribution parameterized by `model(x)`.

        x : tensor[ batch_size x sequence_length ]
    '''
    ## [ batch_size x sequence_length x vocab_size ]
    logits = forward(model, x, **forward_kwargs)

    ## pluck the logits at the final step and scale by temperature
    ## [ batch_size x vocab_size ]
    logits = logits[:, -1] / temperature

    ## keep track of probabilities before modifying logits
    raw_probs = logits.softmax(dim=-1)

    ## optionally crop logits to only the top `1 - cdf` percentile
    if cdf is not None:
        logits = filter_cdf(logits, cdf)

    ## optionally crop logits to only the most likely `k` options
    if topk is not None:
        logits = top_k_logits(logits, topk)

    ## apply softmax to convert to probabilities
    probs = logits.softmax(dim=-1)

    ## sample from the distribution
    ## [ batch_size x 1 ]
    indices = torch.multinomial(probs, num_samples=1)

    return indices, raw_probs

@torch.no_grad()
def sample_n(model, x, N, **sample_kwargs):
    batch_size = len(x)

    ## keep track of probabilities from each step;
    ## `vocab_size + 1` accounts for termination token
    probs = torch.zeros(batch_size, N, model.vocab_size + 1, device=x.device)

    for n in range(N):
        indices, p = sample(model, x, **sample_kwargs)

        ## append to the sequence and continue
        ## [ batch_size x (sequence_length + n) ]
        x = torch.cat((x, indices), dim=1)

        probs[:, n] = p

    return x, probs


def plan(
    env, model,
    s_dim, a_dim,
    plan_freq, 
    discretizer,
    prefix_context, 
    horizon, 
    beam_width, 
    n_expand,
    max_context_transitions,
    discount=0.99, 
    k_obs=1, k_act=None, 
    cdf_obs=None, cdf_act=0.6, 
    percentile='mean',
    device='cpu'
    ):

    observation = env.reset()
    total_reward = 0.0
    context = []
    value_fn = lambda x: discretizer.value_fn(x, percentile)
    T = env.max_episode_steps

    for t in range(T):
        if t % plan_freq == 0:
            ## concatenate previous transitions and current observations to input to model
            prefix = make_prefix(discretizer, context, observation, device, prefix_context)

            ## sample sequence from model beginning with `prefix`
            sequence = beam_plan(
                model, value_fn, prefix,
                horizon, beam_width, n_expand, s_dim, a_dim,
                discount, max_context_transitions,
                k_obs=k_obs, k_act=k_act, cdf_obs=cdf_obs, cdf_act=cdf_act,
            )
        else:
            sequence = sequence[1:]

        ## [ horizon x transition_dim ] convert sampled tokens to continuous trajectory
        sequence_recon = discretizer.reconstruct(sequence)

        ## [ action_dim ] index into sampled trajectory to grab first action
        action = extract_actions(sequence_recon, s_dim, a_dim, t=0)

        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)

        ## update return
        total_reward += reward
        score = env.get_normalized_score(total_reward)

        ## update rollout observations and context transitions
        context = update_context(context, discretizer, observation, action, reward, max_context_transitions, device)

        print(
            f'[ plan ] timestep: {t} / {T} | reward: {reward:.2f} | total Reward: {total_reward:.2f} | normalized_score: {score:.4f} | \n'
        )

        if terminal: break

        observation = next_observation

    return score, t, total_reward, terminal, context