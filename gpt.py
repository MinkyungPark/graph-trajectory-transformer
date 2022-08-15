import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from encoder import GraphormerEncoder


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        ## mask previous value estimates
        self.mask.squeeze()[:, config.total_dim-1::config.total_dim] = 0
        self.n_head = config.n_head
        self.n_vt = config.num_virtual_tokens
        self.n_node = config.num_node
        self.f_dim = config.node_feature_dim

    def forward(self, x, attn_bias=None, layer_past=None):
        B, T, C = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        ## [ B x n_heads x T x head_dim ]
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        ## [ B x n_heads x T x T ]
        padded_attn_bias = torch.zeros((B, self.n_head, T, T), device=x.device).contiguous()

        if attn_bias is not None:
            # (nh, N+1, N+1)
            nh, n = attn_bias.size()[:2]
            attn_bias = torch.repeat_interleave(attn_bias, self.f_dim)
            attn_bias = attn_bias.view(nh, -1, n*self.f_dim)
            attn_bias = attn_bias.repeat(1, 1, self.f_dim)
            attn_bias = attn_bias.view(nh, n*self.f_dim, n*self.f_dim)
            nh, d = attn_bias.size()[:2]
            padded_attn_bias[:, :nh, :d, :d] =  attn_bias

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) + padded_attn_bias
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        self._attn_map = att.clone()
        att = self.attn_drop(att)
        ## [ B x n_heads x T x head_size ]
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        ## [ B x T x embedding_dim ]
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attn_bias=None):
        x = x + self.attn(self.ln1(x), attn_bias)
        x = x + self.mlp(self.ln2(x))
        return x


class EinLinear(nn.Module):

    def __init__(self, n_models, in_features, out_features, bias):
        super().__init__()
        self.n_models = n_models
        self.out_features = out_features
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(n_models, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_models, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.n_models):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, input):
        """
            input : [ B x n_models x input_dim ]
        """
        ## [ B x n_models x output_dim ]
        output = torch.einsum('eoi,bei->beo', self.weight, input)
        if self.bias is not None:
            raise RuntimeError()
        return output

    def extra_repr(self):
        return 'n_models={}, in_features={}, out_features={}, bias={}'.format(
            self.n_models, self.in_features, self.out_features, self.bias is not None
        )


class GPTConfig:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem (+1 for stop token)
        self.grp_encoder = GraphormerEncoder(config)
        self.tok_emb = nn.Embedding(
            config.vocab_size * (config.arV_dim + config.num_node) + 1, config.n_embd
        )

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        # self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head = EinLinear(config.total_dim, config.n_embd, config.vocab_size + 1, bias=False)

        self.embd_dim = config.n_embd
        self.n_head = config.n_head
        self.apply(self._init_weights)

        self.vocab_size = config.vocab_size
        self.stop_token = config.vocab_size * config.total_dim
        self.block_size = config.block_size
        
        self.action_weight = config.action_weight
        self.reward_weight = config.reward_weight
        self.value_weight = config.value_weight

        self.action_dim = config.action_dim
        self.total_trans_dim = config.total_dim
        self.num_virtual_tokens = config.num_virtual_tokens
        self.num_node = config.num_node
        self.node_feature_dim = config.node_feature_dim
        self.graph_attr = config.graph_attr
        

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, EinLinear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def state_offset(self, x):
        bs, n, f = x.size()
        state_offsets = torch.arange(n) * self.vocab_size
        state_offsets = state_offsets[None, :, None]
        state_offsets = state_offsets.repeat(bs, 1, f).to(x.device)
        offset_x = x + state_offsets
        return offset_x

    def arV_offset(self, y):
        bs, d = y.size()
        trans_dim = self.num_node + d
        stop_token = trans_dim * self.vocab_size
        arV_offsets = torch.arange(self.num_node, trans_dim) * self.vocab_size
        arV_offsets = arV_offsets[None, :].repeat(bs, 1).to(y.device)
        offset_y = y + arV_offsets
        offset_y[y == self.vocab_size] = stop_token
        return offset_y

    def pad_to_full_observation(self, x, verify=False):
        b, t, _ = x.shape
        n_pad = (self.total_trans_dim - t % self.total_trans_dim) % self.total_trans_dim
        padding = torch.zeros(b, n_pad, self.embd_dim, device=x.device)
        ## [ B x T' x embd_dim ]
        x_pad = torch.cat([x, padding], dim=1)
        ## [ (B * T' / transition_dim) x transition_dim x embd_dim ]
        x_pad = x_pad.view(-1, self.total_trans_dim, self.embd_dim)
        if verify:
            self.verify(x, x_pad)
        return x_pad, n_pad

    def verify(self, x, x_pad):
        b, t, embedding_dim = x.shape
        n_states = int(np.ceil(t / self.total_trans_dim))
        inds = torch.arange(0, self.total_trans_dim).repeat(n_states)[:t]
        for i in range(self.total_trans_dim):
            x_ = x[:,inds == i]
            t_ = x_.shape[1]
            x_pad_ = x_pad[:,i].view(b, n_states, embedding_dim)[:,:t_]
            print(i, x_.shape, x_pad_.shape)
            assert (x_ == x_pad_).all()

    def forward(self, x, y, target=False):
        """
            x : ( B, S, N_node, F )
            y : ( B, S, arV_dim )
            S : subsample * step
        """
        b, s, n, f = x.size()
        assert s <= self.block_size, "Cannot forward, model block size is exhausted."

        offset_x = self.state_offset(x.view(b*s, n, f))
        # (BS, (N+1)*F, n_embd), (nH, N+1, N+1)
        state_embd, graph_attn_bias = self.grp_encoder(offset_x)

        offset_y = self.arV_offset(y.view(b*s, -1))
        # (BS, arV_dim, n_embd)
        arV_embd = self.tok_emb(offset_y)

        state_embd = state_embd.view(b, -1, self.embd_dim)
        arV_embd = arV_embd.view(b, -1, self.embd_dim)
        token_embeddings = torch.cat([state_embd, arV_embd], dim=1)

        t = token_embeddings.size(1) # t = S * (N_node + arV_dim) 
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x, attn_bias=graph_attn_bias)
            # x = block(x)
        x = self.ln_f(x)

        # S' : n_pad * S
        # ( B*S', total_dim, N_embd )
        x_pad, n_pad = self.pad_to_full_observation(x)
        # ( B*S', total_dim, vocab_size + 1 )
        logits = self.head(x_pad)
        # ( B, S' * total_dim, vocab_size + 1 )
        logits = logits.reshape(b, t + n_pad, self.vocab_size + 1)
        # ( B, S' * total_dim - 1, vocab_size + 1 ) 512, 159, 101
        logits = logits[:,:t]

        # logit 256, 470, 101
        # target 2560, 47
        if target is not None:
            y_hat = logits.reshape(-1, logits.size(-1))
            y_true = target.contiguous().view(-1)
            loss = F.cross_entropy(y_hat, y_true, reduction='none')
            if self.action_weight != 1 or self.reward_weight != 1 or self.value_weight != 1:
                #### make weights
                n_states = int(np.ceil(t / self.total_trans_dim))
                weights = torch.cat([
                    torch.ones((n+1) * f, device=x.device),
                    torch.ones(self.action_dim, device=x.device) * self.action_weight,
                    torch.ones(1, device=x.device) * self.reward_weight,
                    torch.ones(1, device=x.device) * self.value_weight,
                ])
                ## [ t + 1]
                weights = weights[1:].repeat(n_states)
                ## [ b x t ]
                weights = weights.repeat(b, 1)
                loss = loss * weights.view(-1)
            # loss = (loss * mask.view(-1)).mean()
        else:
            loss = None

        return logits, loss