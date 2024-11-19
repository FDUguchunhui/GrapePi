
import logging
import os
import src.custom_graphgym # noqa
from yacs.config import CfgNode
import torch
from torch_geometric import seed_everything
from torch_geometric.graphgym import set_printing, register
from torch_geometric.graphgym.config import (
    cfg,
    set_run_dir,
)
from src.utils.config import dump_cfg
from torch_geometric.graphgym.register import train_dict
from torch_geometric.graphgym.train import GraphGymDataModule
from src.utils.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from omegaconf import DictConfig, OmegaConf
import hydra
from src.custom_graphgym.train.graphsage_model_builder import NeighborNodeGraphGymDataModule


@hydra.main(version_base='1.3', config_path='configs/protein', config_name='grapepi')
def main(cfg_: DictConfig):
    # Load cmd line args
    # load additional custom config
    register.config_dict.get('grape_pi')(cfg)
    cfg.merge_from_other_cfg(CfgNode(OmegaConf.to_container(cfg_)))
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    # Repeat for different random seeds
    for i in range(cfg.run.repeat):
        set_run_dir(cfg.output_dir)
        set_printing()
        # Set configurations for each run
        cfg.seed = cfg.seed + 1
        seed_everything(cfg.seed)
        auto_select_device()
        # Set machine learning pipeline

        # use the right customized datamodule and graphgymmodule
        if cfg.gnn.layer_type == 'sageconv':
            model = train_dict['sageconv_create_model']()
            datamodule = NeighborNodeGraphGymDataModule()
        elif cfg.gnn.layer_type == 'gcnconv' or cfg.gnn.layer_type == 'gcnconv_with_edgeweight':
            model = train_dict['gcnconv_create_model']()
            datamodule = GraphGymDataModule()
        elif cfg.gnn.layer_type == 'gatconv':
            model = train_dict['gcnconv_create_model']()
            datamodule = GraphGymDataModule()
        else:
            raise ValueError(f'Unsupported layer type {cfg.gnn.layer_type}. current support sageconv, gcnconv, gatconv, gcnconv_with_edgeweight')

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        # Print model info
        logger.info(model)
        logger.info(cfg)
        model(next(iter(datamodule.loaders[0])))
        cfg.params = params_count(model)
        logger.info('Num parameters: %s', cfg.params)
        # num_edges = datamodule.loaders[0].dataset.data.num_edges // 2
        # logger.info(f'Num of interactions: {num_edges}')
        train(model, datamodule, logger=True)

    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if cfg.run.mark_done:
        os.rename(hydra.job.config_name, f'{hydra.job.config_name}_done')

if __name__ == '__main__':
    main()