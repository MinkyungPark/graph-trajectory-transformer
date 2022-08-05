"""
    graph
    DataBatch(x=[512, 7, 5], edge_index=[1, 2, 3072], y=[512, 8], 
    adj=[512, 8, 8], attn_bias=[512, 8, 8], attn_edge_type=[512, 7, 7, 1], 
    rel_pos=[512, 7, 7], in_degree=[512, 7], out_degree=[512, 7], 
    edge_input=[512, 7, 7, 100, 1], batch=[512], ptr=[513])
"""
import torch
import torch.nn as nn 
import numpy as np

def convert_to_single_emb(x, feature_num, offset=100, s=0):
    feature_offset = torch.arange(s, feature_num * offset, offset, dtype=torch.long, device=x.device)
    x = x + feature_offset
    return x

class GraphormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.trans_dim = config.trans_dim
        self.vocab_size = config.vocab_size
        
        self.n_vt = config.num_virtual_token # num virtual tokens
        self.n_node = config.num_node
        self.feature_dim = config.node_feature_dim

        self.rel_pos_encoder = nn.Embedding(self.vocab_size + 1, self.n_head)
        self.graph_token_virtual_distance = nn. Embedding(self.n_vt, self.n_head)
        self.edge_encoder = nn.Embedding(self.n_node, self.n_head)
        self.atom_encoder = nn.Embedding(self.vocab_size * self.feature_dim * self.vocab_size, self.n_embd)
        self.in_degree_encoder = nn.Embedding(self.n_node, self.n_embd)
        self.out_degree_encoder = nn.Embedding(self.n_node, self.n_embd)
        self.graph_token = nn.Embedding(self.n_vt, self.n_embd)

    def forward(self, graph, perturb=None):
        b, n = graph.x.size()[:2] # batch_size, num_nodes

        x, y, attn_bias, rel_pos = (graph.x, graph.y, graph.attn_bias, graph.rel_pos)
        x = convert_to_single_emb(x, x.size(-1), offset=self.vocab_size)
        in_degree, out_degree = (graph.in_degree, graph.out_degree)
        edge_input, attn_edge_type = (graph.edge_input, graph.attn_edge_type)

        graph_attn_bias = attn_bias.clone()
        # [B, nH, N+1, N+1]
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        
        # rel_pos [B, N, N, nH] -> [B, nH, N, N]
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, self.n_vt:, self.n_vt:] = (
            graph_attn_bias[:, :, self.n_vt:, self.n_vt:] + rel_pos_bias
        )

        # reset rel pos
        t = self.graph_token_virtual_distance.weight.view(1, self.n_head, self.n_vt).unsqueeze(-2)
        graph_attn_bias[:, :, self.n_vt:, :self.n_vt] = (
            graph_attn_bias[:, :, self.n_vt:, :self.n_vt] + t
        )

        edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)
        
        graph_attn_bias[:, :, self.n_vt:, self.n_vt:] = (
            graph_attn_bias[:, :, self.n_vt:, self.n_vt:] + edge_input
        )
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1) # reset

        # node_feature + graph token
        # [B, N, n_embd]
        # node_feature = self.atom_encoder(x).sum(dim=-2)
        node_feature = self.atom_encoder(x)
        b, n, f, e = node_feature.size()
        node_feature = node_feature.view(b, n*f, e)
        if perturb is not None:
            node_feature += perturb

        in_degree = torch.repeat_interleave(in_degree, f).view(b, -1)
        out_degree = torch.repeat_interleave(out_degree, f).view(b, -1)
        in_degree_feature = self.in_degree_encoder(in_degree)
        out_degree_feature = self.out_degree_encoder(out_degree)
        node_feature = ( # same as x dim
            node_feature + in_degree_feature + out_degree_feature
        )
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(b, f, 1)
        graph_node_feature = torch.cat([node_feature, graph_token_feature], dim=1)

        # (B, (N + 1) * F , n_embd), (B, nH, N+1, N+1)
        return graph_node_feature, graph_attn_bias
