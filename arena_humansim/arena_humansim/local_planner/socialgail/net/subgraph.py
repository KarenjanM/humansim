import copy

import torch
from torch import nn
from torch_geometric.nn import MessagePassing, max_pool


class GraphLayerProp(MessagePassing):
    def __init__(self, in_channels, hidden_unit=64):
        super().__init__(aggr="max")
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, in_channels),
        )

    def forward(self, x, edge_index):
        x = self.mlp(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):  # pyright: ignore[reportIncompatibleMethodOverride]
        return torch.cat([x, aggr_out], dim=1)


class SubGraph(nn.Module):
    def __init__(self, in_channels, num_subgraph_layres=3, hidden_unit=64):
        super().__init__()
        self.num_subgraph_layres = num_subgraph_layres
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layres):
            self.layer_seq.add_module(f"glp_{i}", GraphLayerProp(in_channels, hidden_unit))
            in_channels *= 2

    def forward(self, data):
        sub_data = copy.deepcopy(data)
        x, edge_index = sub_data.x, sub_data.edge_index
        for _name, layer in self.layer_seq.named_modules():
            if isinstance(layer, GraphLayerProp):
                x = layer(x, edge_index)
        sub_data.x = x
        out_data = max_pool(sub_data.cluster, sub_data)
        assert out_data.x is not None
        assert out_data.x.shape[0] % int(sub_data.time_step_len[0]) == 0
        return out_data
