import os
import shlex

from omegaconf import DictConfig as CN, OmegaConf
from omegaconf import OmegaConf as OC
from torch_geometric.graphgym import register
import os.path as osp

from torch_geometric.graphgym.config import assert_cfg
from yacs.config import CfgNode
from hydra.core.hydra_config import HydraConfig

def add_cfg_if_not_exists(cfg):
    r"""This function adds a configuration if it doesn't exist.

    1) Note that for an experiment, only part of the arguments will be used.
       The remaining unused arguments won't affect anything.
       So feel free to register any argument in graphgym.contrib.config
    2) We support *at most* two levels of configs, *e.g.*,
       :obj:`cfg.dataset.name`.

    :return: Configuration used by the experiment.
    """
    if cfg is None:
        return cfg

    # ----------------------------------------------------------------------- #
    # Basic options
    # ----------------------------------------------------------------------- #

    # Set print destination: stdout / file / both
    cfg.setdefault('print', 'both')

    # Select device: 'cpu', 'cuda', 'auto'
    cfg.setdefault('accelerator', 'auto')

    # number of devices: eg. for 2 GPU set cfg.devices=2
    cfg.setdefault('devices', 1)

    # Output directory
    cfg.setdefault('out_dir', 'results')

    # Config name (in out_dir)
    cfg.setdefault('cfg_dest', 'config.yaml')

    # Names of registered custom metric funcs to be used (use defaults if none)
    cfg.setdefault('custom_metrics', [])

    # Random seed
    cfg.setdefault('seed', 0)

    # Print rounding
    cfg.setdefault('round', 4)

    # Tensorboard support for each run
    cfg.setdefault('tensorboard_each_run', False)

    # Tensorboard support for aggregated results
    cfg.setdefault('tensorboard_agg', True)

    # Additional num of worker for data loading
    cfg.setdefault('num_workers', 0)

    # Max threads used by PyTorch
    cfg.setdefault('num_threads', 6)

    # The metric for selecting the best epoch for each run
    cfg.setdefault('metric_best', 'auto')

    # argmax or argmin in aggregating results
    cfg.setdefault('metric_agg', 'argmax')

    # If visualize embedding.
    cfg.setdefault('view_emb', False)

    # If get GPU usage
    cfg.setdefault('gpu_mem', False)

    # If do benchmark analysis
    cfg.setdefault('benchmark', False)

    # ----------------------------------------------------------------------- #
    # Globally shared variables:
    # These variables will be set dynamically based on the input dataset
    # Do not directly set them here or in .yaml files
    # ----------------------------------------------------------------------- #

    cfg.setdefault('share',  OmegaConf.create({}))
    cfg.share.setdefault('dim_in', 1)
    cfg.share.setdefault('dim_out', 1)
    cfg.share.setdefault('num_splits', 1)

    # ----------------------------------------------------------------------- #
    # Dataset options
    # ----------------------------------------------------------------------- #
    cfg.setdefault('dataset',  OmegaConf.create({}))
    cfg.dataset.setdefault('name', 'Cora')
    cfg.dataset.setdefault('format', 'PyG')
    cfg.dataset.setdefault('dir', './datasets')
    cfg.dataset.setdefault('task', 'node')
    cfg.dataset.setdefault('task_type', 'classification')
    cfg.dataset.setdefault('transductive', True)
    cfg.dataset.setdefault('split', [0.8, 0.1, 0.1])
    cfg.dataset.setdefault('shuffle_split', True)
    cfg.dataset.setdefault('split_mode', 'random')
    cfg.dataset.setdefault('encoder', True)
    cfg.dataset.setdefault('encoder_name', 'db')
    cfg.dataset.setdefault('encoder_bn', True)
    cfg.dataset.setdefault('node_encoder', False)
    cfg.dataset.setdefault('node_encoder_name', 'Atom')
    cfg.dataset.setdefault('node_encoder_bn', True)
    cfg.dataset.setdefault('edge_encoder', False)
    cfg.dataset.setdefault('edge_encoder_name', 'Bond')
    cfg.dataset.setdefault('edge_encoder_bn', True)
    cfg.dataset.setdefault('encoder_dim', 128)
    cfg.dataset.setdefault('edge_dim', 128)
    cfg.dataset.setdefault('edge_train_mode', 'all')
    cfg.dataset.setdefault('edge_message_ratio', 0.8)
    cfg.dataset.setdefault('edge_negative_sampling_ratio', 1.0)
    cfg.dataset.setdefault('resample_disjoint', False)
    cfg.dataset.setdefault('resample_negative', False)
    cfg.dataset.setdefault('transform', 'none')
    cfg.dataset.setdefault('cache_save', False)
    cfg.dataset.setdefault('cache_load', False)
    cfg.dataset.setdefault('remove_feature', False)
    cfg.dataset.setdefault('tu_simple', True)
    cfg.dataset.setdefault('to_undirected', False)
    cfg.dataset.setdefault('location', 'local')
    cfg.dataset.setdefault('label_table', 'none')
    cfg.dataset.setdefault('label_column', 'none')

    # ----------------------------------------------------------------------- #
    # Training options
    # ----------------------------------------------------------------------- #
    cfg.setdefault('train',  OmegaConf.create({}))
    cfg.train.setdefault('batch_size', 16)
    cfg.train.setdefault('sampler', 'full_batch')
    cfg.train.setdefault('sample_node', False)
    cfg.train.setdefault('node_per_graph', 32)
    cfg.train.setdefault('radius', 'extend')
    cfg.train.setdefault('eval_period', 10)
    cfg.train.setdefault('skip_train_eval', False)
    cfg.train.setdefault('ckpt_period', 100)
    cfg.train.setdefault('enable_ckpt', True)
    cfg.train.setdefault('auto_resume', False)
    cfg.train.setdefault('epoch_resume', -1)
    cfg.train.setdefault('ckpt_clean', True)
    cfg.train.setdefault('iter_per_epoch', 32)
    cfg.train.setdefault('walk_length', 4)
    cfg.train.setdefault('neighbor_sizes', [20, 15, 10, 5])

    # ----------------------------------------------------------------------- #
    # Validation options
    # ----------------------------------------------------------------------- #
    cfg.setdefault('val', OmegaConf.create({}))
    cfg.val.setdefault('sample_node', False)
    cfg.val.setdefault('sampler', 'full_batch')
    cfg.val.setdefault('node_per_graph', 32)
    cfg.val.setdefault('radius', 'extend')

    # ----------------------------------------------------------------------- #
    # Model options
    # ----------------------------------------------------------------------- #
    cfg.setdefault('model',  OmegaConf.create({}))
    cfg.model.setdefault('type', 'gnn')
    cfg.model.setdefault('match_upper', True)
    cfg.model.setdefault('loss_fun', 'cross_entropy')
    cfg.model.setdefault('size_average', 'mean')
    cfg.model.setdefault('thresh', 0.5)
    cfg.model.setdefault('edge_decoding', 'dot')
    cfg.model.setdefault('graph_pooling', 'add')

    # ----------------------------------------------------------------------- #
    # GNN options
    # ----------------------------------------------------------------------- #
    cfg.setdefault('gnn',  OmegaConf.create({}))
    cfg.gnn.setdefault('head', 'default')
    cfg.gnn.setdefault('layers_pre_mp', 0)
    cfg.gnn.setdefault('layers_mp', 2)
    cfg.gnn.setdefault('layers_post_mp', 0)
    cfg.gnn.setdefault('dim_inner', 16)
    cfg.gnn.setdefault('layer_type', 'generalconv')
    cfg.gnn.setdefault('stage_type', 'stack')
    cfg.gnn.setdefault('skip_every', 1)
    cfg.gnn.setdefault('batchnorm', True)
    cfg.gnn.setdefault('act', 'relu')
    cfg.gnn.setdefault('dropout', 0.0)
    cfg.gnn.setdefault('agg', 'add')
    cfg.gnn.setdefault('normalize_adj', False)
    cfg.gnn.setdefault('msg_direction', 'single')
    cfg.gnn.setdefault('self_msg', 'concat')
    cfg.gnn.setdefault('att_heads', 1)
    cfg.gnn.setdefault('att_final_linear', False)
    cfg.gnn.setdefault('att_final_linear_bn', False)
    cfg.gnn.setdefault('l2norm', True)
    cfg.gnn.setdefault('keep_edge', 0.5)
    cfg.gnn.setdefault('clear_feature', True)

    # ----------------------------------------------------------------------- #
    # Optimizer options
    # ----------------------------------------------------------------------- #
    cfg.setdefault('optim',  OmegaConf.create({}))
    cfg.optim.setdefault('optimizer', 'adam')
    cfg.optim.setdefault('base_lr', 0.01)
    cfg.optim.setdefault('weight_decay', 5e-4)
    cfg.optim.setdefault('momentum', 0.9)
    cfg.optim.setdefault('scheduler', 'cos')
    cfg.optim.setdefault('steps', [30, 60, 90])
    cfg.optim.setdefault('lr_decay', 0.1)
    cfg.optim.setdefault('max_epoch', 200)

    # ----------------------------------------------------------------------- #
    # Batch norm options
    # ----------------------------------------------------------------------- #
    cfg.setdefault('bn',  OmegaConf.create({}))
    cfg.bn.setdefault('eps', 1e-5)
    cfg.bn.setdefault('mom', 0.1)

    # ----------------------------------------------------------------------- #
    # Memory options
    # ----------------------------------------------------------------------- #
    cfg.setdefault('mem',  OmegaConf.create({}))
    cfg.mem.setdefault('inplace', False)

    # Set user customized cfgs
    for func in register.config_dict.values():
        func(cfg)

    return cfg

def dump_cfg(cfg):
    r"""Dumps the config to the output directory specified in
    :obj:`cfg.out_dir`.

    Args:
        cfg: Configuration node
    """
    if cfg.run.name is not None:
        output_dir = os.path.join(cfg.out_dir, cfg.run.name)
    else:
        output_dir = os.path.join(cfg.out_dir,  get_fname(HydraConfig.get().job.config_name))
    cfg.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    cfg_file = osp.join(output_dir, cfg.cfg_dest)
    # remove the last part of the path of cfg.out_dir
    # this is bug in original graphgym since its change cfg make the saved config file cannot be used to rerun the experiment
    with open(cfg_file, 'w') as f:
        cfg.dump(stream=f)



def load_cfg(cfg, args):
    r"""Load configurations from file system and command line.

    Args:
        cfg (CfgNode): Configuration node
        args (ArgumentParser): Command argument parser
    """
    cfg.merge_from_file(args.cfg_file)
    # todo: this is a temporary fix without changing code in torch_geometric/graphgym
    # Split the override arguments into a list
    override_args = shlex.split(args.override) if args.override else []
    cfg.merge_from_other_cfg(CfgNode(OmegaConf.to_container(OmegaConf.from_dotlist(override_args))))
    assert_cfg(cfg)


def get_fname(fname):
    r"""Extract filename from file name path.

    Args:
        fname (str): Filename for the yaml format configuration file
    """
    fname = osp.basename(fname)
    if fname.endswith('.yaml'):
        fname = fname[:-5]
    elif fname.endswith('.yml'):
        fname = fname[:-4]
    return fname
