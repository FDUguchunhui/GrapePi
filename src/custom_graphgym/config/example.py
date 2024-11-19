from omegaconf import DictConfig, OmegaConf
from torch_geometric.graphgym.register import register_config


@register_config('example')
def set_cfg_example(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # example argument
    cfg.example_arg = 'example'

    # example argument group
    cfg.setdefault('example_group', OmegaConf.create({}))

    # then argument can be specified within the group
    cfg.example_group.example_arg = 'example'



