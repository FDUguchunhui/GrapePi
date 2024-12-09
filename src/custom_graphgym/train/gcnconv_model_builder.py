import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch_geometric.data.lightning.datamodule import LightningDataModule
from torch_geometric.graphgym import create_loader
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.loader import DataLoader

from src.custom_graphgym.loader.protein import ProteinDataset


class GCNConvGymModule(GraphGymModule):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__(dim_in, dim_out, cfg)

    def training_step(self, batch, *args, **kwargs):
        logits, true = self(batch)
        logits, true = logits[batch.train_mask & ~batch.unlabeled_mask], true[batch.train_mask & ~batch.unlabeled_mask]
        loss, pred_score = compute_loss(logits, true)
        step_end_time = time.time()
        return dict(loss=loss, true=true, pred_score=pred_score.detach(),
                    step_end_time=step_end_time)


    def validation_step(self, batch, *args, **kwargs):
        logits, true = self(batch)
        logits, true = logits[batch.val_mask & ~batch.unlabeled_mask], true[batch.val_mask & ~batch.unlabeled_mask]
        loss, pred_score = compute_loss(logits, true)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        step_end_time = time.time()
        return dict(loss=loss, true=true, pred_score=pred_score.detach(),
                    step_end_time=step_end_time)


    def test_step(self, batch, *args, **kwargs):
        logits, true = self(batch)
        logits, true = logits[batch.test_mask & ~batch.unlabeled_mask], true[batch.test_mask & ~batch.unlabeled_mask]
        loss, pred_score = compute_loss(logits, true)
        step_end_time = time.time()
        return dict(loss=loss, true=true, pred_score=pred_score.detach(),
                    step_end_time=step_end_time)


    def predict_step(self, batch, *args, **kwargs):
        # only predict on unlabeled nodes
        logits, true = self(batch)
        index = np.arange(batch.x.shape[0])
        unlabeled = batch.unlabeled_mask
        pred_prob = torch.nn.functional.sigmoid(logits)
        return dict(pred_prob=pred_prob.squeeze().detach().cpu().numpy(),
                    true=true.detach().cpu().numpy(),
                    index=index,
                    unlabeled=unlabeled.detach().cpu().numpy())

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        dataset = ProteinDataset(root=cfg.dataset.dir, rebuild=True,
                                 node_numeric_cols=cfg.dataset.node_numeric_cols,
                                 label_col=cfg.dataset.label_col,
                                 include_seq_embedding=cfg.dataset.include_seq_embedding,
                                 num_val=0, num_test=0)

        return dataset

class GraphGymDataModule(LightningDataModule):
    r"""A :class:`pytorch_lightning.LightningDataModule` for handling data
    loading routines in GraphGym.

    This class provides data loaders for training, validation, and testing, and
    can be accessed through the :meth:`train_dataloader`,
    :meth:`val_dataloader`, and :meth:`test_dataloader` methods, respectively.
    """
    def __init__(self):
        self.loaders = create_loader()
        super().__init__(has_val=True, has_test=True)

    def train_dataloader(self) -> DataLoader:
        return self.loaders[0]

    def val_dataloader(self) -> DataLoader:
        # better way would be to test after fit.
        # First call trainer.fit(...) then trainer.test(...)
        return self.loaders[1]

    def test_dataloader(self) -> DataLoader:
        return self.loaders[2]

@register_train("gcnconv_create_model")
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

    model = GCNConvGymModule(dim_in, dim_out, cfg)
    if to_device:
        model.to(torch.device(cfg.accelerator))
    return model

