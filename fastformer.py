import einops
from einops import rearrange
import torch
import torch.nn as nn


class FastformerAttention(nn.Module):
    def __init__(self, dim=3, decode_dim=16):
        super(FastformerAttention, self).__init__()
        # Generate weight for Wquery、Wkey and Wvalue
        self.to_qkv = nn.Linear(dim, decode_dim * 3, bias=False)
        self.weight_q = nn.Linear(dim, decode_dim, bias=False)
        self.weight_k = nn.Linear(dim, decode_dim, bias=False)
        self.weight_v = nn.Linear(dim, decode_dim, bias=False)
        self.weight_r = nn.Linear(decode_dim, decode_dim, bias=False)
        self.weight_alpha = nn.Parameter(torch.randn(decode_dim))
        self.weight_beta = nn.Parameter(torch.randn(decode_dim))
        self.scale_factor = decode_dim ** -0.5

    def forward(self, x, attn_bias=None, mask=None):
        query = self.weight_q(x)
        key = self.weight_k(x)
        value = self.weight_v(x)
        b, n, d = query.shape

        # mask_value = torch.finfo(x.dtype).min
        # mask = rearrange(mask, "b n -> b () n")

        # Caculate the global query
        alpha_weight = torch.softmax(
            torch.mul(query, self.weight_alpha) * self.scale_factor, dim=-1
        )
        global_query = query * alpha_weight
        # global_query = global_query.masked_fill(~mask, mask_value)
        global_query = torch.einsum("b n d -> b d", global_query)

        # Model the interaction between global query vector and the key vector
        repeat_global_query = einops.repeat(global_query, "b d -> b copy d", copy=n)
        # attn_bias = attn_bias.sum(dim=-1).permute(0,2,1)
        # if attn_bias is not None:
        #     p = (repeat_global_query + attn_bias) * key
        # else:
        p = repeat_global_query * key
        beta_weight = torch.softmax(
            torch.mul(p, self.weight_beta) * self.scale_factor, dim=-1
        )
        global_key = p * beta_weight
        global_key = torch.einsum("b n d -> b d", global_key)

        # key-value
        key_value_interaction = torch.einsum("b j, b n j -> b n j", global_key, value)
        key_value_interaction_out = self.weight_r(key_value_interaction)
        result = key_value_interaction_out + query
        return result


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = FastformerAttention(
            dim=hidden_size, decode_dim=hidden_size
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, mask=None):

        y = self.self_attention_norm(x)
        y = self.self_attention(y)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return 