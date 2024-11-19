import torch_geometric as pyg
import torch
from torch_geometric.graphgym.register import register_layer
from torch_geometric.graphgym.models.layer import LayerConfig

@register_layer('gcnconv_with_edgeweight')
class GCNConv(torch.nn.Module):
    r"""A Graph Convolutional Network (GCN) layer."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GCNConv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias,
        )

    def forward(self, batch):
        # check if edge_weight is in the batch
        if 'edge_weight' not in batch:
            batch.x = self.model(batch.x, batch.edge_index)
        else:
            batch.x = self.model(batch.x, batch.edge_index, edge_weight=batch.edge_weight)
        return batch