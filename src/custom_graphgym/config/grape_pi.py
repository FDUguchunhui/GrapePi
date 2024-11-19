from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

@register_config('grape_pi')
def set_cfg(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #
    cfg.run_dir = None
    cfg.dataset.node_numeric_cols = None
    cfg.dataset.edge_numeric_cols = None
    cfg.dataset.label_col = None
    cfg.dataset.rebuild = False
    cfg.dataset.interaction_weight_col = None
    cfg.dataset.interaction_conf_col = None
    cfg.dataset.interaction_conf_thresh = 700.0
    cfg.train.pos_weight = -1.0
    cfg.train.early_stop = False
    cfg.train.early_stop_patience = 50
    cfg.run = CN()
    cfg.run.repeat = 1
    cfg.run.name = None
    cfg.run.mark_done = False
