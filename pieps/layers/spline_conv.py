"""DAGr-derived sparse spline layer used by the PIE encoder."""
import torch
from torch_geometric.nn.conv import SplineConv
from torch_geometric.data import Data
from torch_geometric.transforms.to_sparse_tensor import ToSparseTensor


class MySplineConv(SplineConv):
    def __init__(self, in_channels, out_channels, args, bias=False, degree=1, **kwargs):
        self.reproducible = False
        self.to_sparse_tensor = ToSparseTensor(attr="edge_attr", remove_edge_index=False)
        super().__init__(in_channels=in_channels, out_channels=out_channels, bias=bias, degree=degree,
                         dim=args.edge_attr_dim, aggr=args.aggr, kernel_size=args.kernel_size)


    def forward(self, data: Data)->Data:
        if self.reproducible:
            # first check we already computed the adjacency matrix
            if not hasattr(data, "adj_t"):
                data.edge_attr = data.edge_attr[:,:self.dim]
                data = self.to_sparse_tensor(data)
            data.x = self._forward(data.x,
                                  edge_index=data.adj_t)
        else:
            data.x = self._forward(data.x,
                                  edge_index=data.edge_index,
                                  edge_attr=data.edge_attr[:, :self.dim],
                                  size=(data.x.shape[0], data.x.shape[0]))
        return data

    def _forward(self, x, edge_index, edge_attr=None, size=None):
        """"""
        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        if edge_index.numel() > 0:
            out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr, size=size)
        else:
            out = torch.zeros((x.size(0), self.out_channels), dtype=x.dtype, device=x.device)

        if x is not None and self.root_weight:
            out += self.lin(x)

        if self.bias is not None:
            out += self.bias

        return out
