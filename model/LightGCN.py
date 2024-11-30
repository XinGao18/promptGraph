import torch
from torch import Tensor
from typing import Optional, Union
from torch_geometric.utils import spmm
from torch.nn import Embedding, ModuleList
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import is_sparse, to_edge_index
from torch_geometric.typing import Adj, OptTensor, SparseTensor


class LightGCNConv(MessagePassing):
    r"""The Light Graph Convolution (LGC) operator from the `"LightGCN:
    Simplifying and Powering Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper.

    Args:
        normalize (bool, optional): If set to :obj:`False`, output features
            will not be normalized via symmetric normalization.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Contents:
        - **input:**
          node features,
          edge indices (COO format),
          edge weights **(optional)**
        - **output:** node features
    """

    def __init__(self, normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.normalize = normalize

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.normalize and isinstance(edge_index, Tensor):
            out = gcn_norm(edge_index, edge_weight, x.size(self.node_dim),
                           add_self_loops=False, flow=self.flow, dtype=x.dtype)
            edge_index, edge_weight = out
        elif self.normalize and isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, None, x.size(self.node_dim),
                                  add_self_loops=False, flow=self.flow,
                                  dtype=x.dtype)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def edge_update(self) -> Tensor:
        pass


class LightGCN(torch.nn.Module):
    r"""The LightGCN model from the `"LightGCN: Simplifying and Powering
    Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper.

    :class:`LightGCN` learns embeddings by linearly propagating them on the underlying graph, and uses the weighted sum of the
    embeddings learned at all layers as the final embedding

    .. math::
        \textbf{x}_i = \sum_{l=0}^{L} \alpha_l \textbf{x}^{(l)}_i,

    where each layer's embedding is computed as

    .. math::
        \mathbf{x}^{(l+1)}_i = \sum_{j \in \mathcal{N}(i)}
        \frac{1}{\sqrt{\deg(i)\deg(j)}}\mathbf{x}^{(l)}_j.

    Args:
        num_nodes (int): The number of nodes in the graph.
        embedding_dim (int): The dimensionality of node embeddings.
        num_layers (int): The number of LightGCNConv layers.
        prompt: The prompt of the model.
        alpha (float or torch.Tensor, optional): The scalar or vector
            specifying the re-weighting coefficients for aggregating the final
            embedding. If set to :obj:`None`, the uniform initialization of
            :obj:`1 / (num_layers + 1)` is used. (default: :obj:`None`)
        **kwargs (optional)
    """

    def __init__(
            self,
            num_nodes: int,
            embedding_dim: int,
            num_layers: int,
            alpha: Optional[Union[float, Tensor]] = None,
            **kwargs,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)
        self.embedding = Embedding(num_nodes, embedding_dim)  # declare embedding size is [num_nodes, embedding_dim]
        self.convs = ModuleList([LightGCNConv(**kwargs) for _ in range(num_layers)])
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(
            self,
            edge_index: Adj,
            edge_weight: OptTensor = None,
            prompt=None,
    ) -> Tensor:
        r"""Returns the embedding of nodes in the graph."""
        x = self.embedding.weight
        if prompt is not None:
            x = prompt.add(x)
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            out = out + x * self.alpha[i + 1]

        return out

    def forward(
            self,
            edge_index: Adj,
            edge_label_index: OptTensor = None,
            edge_weight: OptTensor = None,
            prompt=None,
    ) -> Tensor:
        r"""Computes rankings for pairs of nodes.

        Args:
            edge_index (torch.Tensor or SparseTensor): Edge tensor specifying
                the connectivity of the graph.
            edge_label_index (torch.Tensor, optional): Edge tensor specifying
                the node pairs for which to compute rankings or probabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used instead. (default: :obj:`None`)
            edge_weight (torch.Tensor, optional): The weight of each edge in
                :obj:`edge_index`. (default: :obj:`None`)
            prompt: The prompt of the model.
        :returns:
            torch.Tensor: The ranking of nodes in :obj:`edge_label_index`.
        """
        if edge_label_index is None:
            if is_sparse(edge_index):
                edge_label_index, _ = to_edge_index(edge_index)
            else:
                edge_label_index = edge_index

        out = self.get_embedding(edge_index, edge_weight, prompt)

        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]

        return (out_src * out_dst).sum(dim=-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'{self.embedding_dim}, num_layers={self.num_layers})')