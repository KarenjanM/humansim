import numpy as np
import torch
from torch import nn


def _sequence_mask(X, valid_len, value: float = 0):
    maxlen = X.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def _masked_softmax(X, valid_len):
    if valid_len is None:
        return nn.functional.softmax(X, dim=-1)
    shape = X.shape
    if valid_len.dim() == 1:
        valid_len = torch.repeat_interleave(valid_len, repeats=shape[1], dim=0)
    else:
        valid_len = valid_len.reshape(-1)
    X = _sequence_mask(X.reshape(-1, shape[-1]), valid_len, -1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)


class SelfAttentionLayer(nn.Module):
    def __init__(self, in_channels, global_graph_width, need_scale=False):
        super().__init__()
        self.in_channels = in_channels
        self.q_lin = nn.Linear(in_channels, global_graph_width)
        self.k_lin = nn.Linear(in_channels, global_graph_width)
        self.v_lin = nn.Linear(in_channels, global_graph_width)
        self.scale_factor_d = global_graph_width

    def forward(self, x, valid_len):
        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)
        scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.scale_factor_d)
        attention_weights = _masked_softmax(scores, valid_len)
        return torch.bmm(attention_weights, value)
