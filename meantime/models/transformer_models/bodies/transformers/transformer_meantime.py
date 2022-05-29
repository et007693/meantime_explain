from meantime.models.transformer_models.utils import PositionwiseFeedForward, SublayerConnection

import torch
from torch import nn as nn
from torch.nn import functional as F


class TransformerMeantimeBlock(nn.Module):
    def __init__(self, args, La, Lr):
        super().__init__()


        hidden = args.hidden_units
        feed_forward_hidden = hidden * 4
        dropout = args.dropout
        self.attention = MixedAttention(args, La, Lr)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(args=args, size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(args=args, size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, abs_kernel, rel_kernel, layer, info):
        # x : B x T x H
        # abs_kernel : La of [B x T x H]
        # rel_kernel : Lr of [B x T x T x H]
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask, abs_kernel, rel_kernel, layer, info))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class MixedAttention(nn.Module):
    def __init__(self, args, La, Lr):
        # La : absolute embedding layer
        # Lr : relative embedding layer
        super().__init__()
        # d_model : embedding 차원
        d_model = args.hidden_units
        dropout = args.dropout
        h = La + Lr  # num_heads
        self.La = La
        self.Lr = Lr
        # key dimension
        self.d_k = d_model // h
        self.h = h
        self.scale = 1 / (self.d_k ** 0.5)
        ## TODO
        # moduleist : nn.Module의 list를 input으로
        # nn.Linear(input_dim,output_dim)
        self.content_linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        # embedding -> key dimension
        self.abs_position_query_linear_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(La)])
        self.abs_position_key_linear_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(La)])
        self.rel_position_key_linear_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(Lr)])

        self.rel_position_bias = nn.Parameter(torch.FloatTensor(1, self.Lr, 1, self.d_k))
        ## OUTPUT
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask, abs_kernel, rel_kernel, layer, info):
        # q, k, v : B x T x H
        # abs_kernel : La of [B x T x H]
        # rel_kernel : Lr of [B x T x T x H]
        batch_size, T = query.size(0), query.size(1)

        # q, k, v, kernel_q, kernel_k : B x n x T x d
        query, key, value = \
            [l(x).view(batch_size, T, self.h, self.d_k).transpose(1, 2) # view : reshape
             for l, x in zip(self.content_linear_layers, (query, key, value))] #item-item matrix에서 query, key, value

        scores = torch.zeros(batch_size, self.h, T, T).to(query)
        if self.La > 0:
            Xq = query[:, :self.La]  # B x La x T x d
            Xk = key[:, :self.La]  # B x La x T x d
            # stack : concat
            # dim : 0 -> 열(세로)
            # dim -> 1 -> 행(가로)
            Pq = torch.stack([l(x) for l, x in zip(self.abs_position_query_linear_layers, abs_kernel)], dim=1)  # B x La x T x d
            Pk = torch.stack([l(x) for l, x in zip(self.abs_position_key_linear_layers, abs_kernel)], dim=1)  # B x La x T x d
            abs_scores = torch.einsum('blid,bljd->blij', Xq + Pq, Xk + Pk)  # B x La x T x T
            scores[:, :self.La] += abs_scores

        if self.Lr > 0:
            Xq = query[:, self.La:]  # B x Lr x T x d
            Xk = key[:, self.La:]  # B x Lr x T x d
            R = torch.stack([l(x) for l, x in zip(self.rel_position_key_linear_layers, rel_kernel)], dim=1)  # B x Lr x T x T x d
            # rel_scores = torch.einsum('blid,bljd->blij', Xq + self.content_bias, Xk)  # B x Lr x T x T
            rel_scores = torch.einsum('blid,bljd->blij', Xq, Xk)  # B x Lr x T x T
            rel_scores += torch.einsum('blid,blijd->blij', Xq + self.rel_position_bias, R)  # B x Lr x T x T
            scores[:, self.La:] += rel_scores

        scores = scores * self.scale

        scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)  # B x n x T x T
        p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value)  # B x n x T x d

        if info is not None:
            info['attn_{}'.format(layer)] = p_attn

        x = x.transpose(1, 2).contiguous().view(batch_size, T, self.h * self.d_k)  # B x T x H

        x = self.output_linear(x)  #  B x T x H
        return x