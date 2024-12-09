import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
from overrides import overrides
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch_geometric.data.lightning.datamodule import LightningDataModule

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import network_dict, register_network, register_train
from src.custom_graphgym.loader.protein import ProteinDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.data.lightning import LightningNodeData
from torch_geometric.graphgym.loader import create_dataset


class GraphsageGraphGymModule(GraphGymModule):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__(dim_in, dim_out, cfg)

    def training_step(self, batch, *args, **kwargs):
        logits, true = self(batch)
        batch_mask = torch.cat([torch.ones(batch.batch_size), torch.zeros(len(batch.y) - batch.batch_size)], dim=0).bool()
        logits, true = logits[batch_mask & ~batch.unlabeled_mask], true[batch_mask & ~batch.unlabeled_mask]
        loss, pred_score = compute_loss(logits, true)
        step_end_time = time.time()
        return dict(loss=loss, true=true, pred_score=pred_score.detach(),
                    step_end_time=step_end_time)


    def validation_step(self, batch, *args, **kwargs):
        logits, true = self(batch)
        # create mask to filter the original mini-batch nodes
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/neighbor_loader.html
        batch_mask = torch.cat([torch.ones(batch.batch_size), torch.zeros(len(batch.y) - batch.batch_size)], dim=0).bool()
        # for each batch, only use test nodes in the original mini-batch nodes
        logits, true = logits[batch_mask & ~batch.unlabeled_mask], true[batch_mask & ~batch.unlabeled_mask]
        loss, pred_score = compute_loss(logits, true)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        step_end_time = time.time()
        return dict(loss=loss, true=true, pred_score=pred_score.detach(),
                    step_end_time=step_end_time)


    def test_step(self, batch, *args, **kwargs):
        logits, true = self(batch)
        # create mask to filter the original mini-batch nodes
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/neighbor_loader.html
        batch_mask = torch.cat([torch.ones(batch.batch_size), torch.zeros(len(batch.y) - batch.batch_size)], dim=0).bool()
        # for each batch, only use test nodes in the original mini-batch nodes
        logits, true = logits[batch_mask & ~batch.unlabeled_mask], true[batch_mask & ~batch.unlabeled_mask]
        loss, pred_score = compute_loss(logits, true)

        step_end_time = time.time()
        return dict(loss=loss, true=true, pred_score=pred_score.detach(),
                    step_end_time=step_end_time)

    def predict_step(self, batch, *args, **kwargs):
        # only predict on unlabeled nodes
        logits, true = self(batch)
        index = np.arange(batch.x.shape[0])
        unlabeled = batch.unlabeled_mask
        logits = logits.squeeze(-1)
        pred_prob = torch.nn.functional.sigmoid(logits)
        return dict(pred_prob=pred_prob.squeeze().detach().cpu().numpy(),
                    true=true.detach().cpu().numpy(),
                    index=index,
                    unlabeled=unlabeled.detach().cpu().numpy())

    @overrides
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        dataset = ProteinDataset(root=cfg.dataset.dir, rebuild=True,
                                 node_numeric_cols=cfg.dataset.node_numeric_cols,
                                 label_col=cfg.dataset.label_col,
                                 include_seq_embedding=cfg.dataset.include_seq_embedding,
                                 num_val=0, num_test=0)

        return dataset

class NeighborNodeGraphGymDataModule(LightningDataModule):
    r"""Create datamodule for graph machine learning.
    """
    def __init__(self):
        super().__init__(has_val=True, has_test=True)
        data = create_dataset()
        datamodule = LightningNodeData(data[0],
                                       input_train_nodes=data.train_mask,
                                       input_val_nodes=data.val_mask,
                                       input_test_nodes=data.test_mask,
                                       loader='neighbor',
                                       num_neighbors=cfg.train.neighbor_sizes,
                                       batch_size=cfg.train.batch_size,
                                       num_workers=cfg.num_workers)
        self.loaders = [datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()]

    def train_dataloader(self):
        return self.loaders[0]

    def val_dataloader(self):
        return self.loaders[1]

    def test_dataloader(self):
        return self.loaders[2]





@register_train("sageconv_create_model")
def create_model(to_device=True, dim_in=None, dim_out=None) -> GraphGymModule:
    r"""Create model for graph machine learning.

    Args:
        to_device (bool, optional): Whether to transfer the model to the
            specified device. (default: :obj:`True`)
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    dim_out = cfg.share.dim_out if dim_out is None else dim_out
    # binary classification, output dim = 1
    if 'classification' == cfg.dataset.task_type and dim_out == 2:
        dim_out = 1

    model = GraphsageGraphGymModule(dim_in, dim_out, cfg)
    if to_device:
        model.to(torch.device(cfg.accelerator))
    return model
