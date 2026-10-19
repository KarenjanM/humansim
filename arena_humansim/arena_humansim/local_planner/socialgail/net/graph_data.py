from torch_geometric.data import Data


class GraphData(Data):
    """Polyline-graph Data with cluster offsetting for batch concatenation."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            assert self.x is not None
            return self.x.size(0)
        if key == "cluster":
            return int(self.cluster.max().item()) + 1
        return 0
