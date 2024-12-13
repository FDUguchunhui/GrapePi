import glob
import logging
import os
import shlex
import subprocess
import warnings

import pytorch_lightning as pl
import torch_geometric.graphgym.register as register
from torch_geometric.data import DataLoader
from torch_geometric.loader import NeighborLoader

import torch
from torch_geometric import seed_everything
import argparse
from torch_geometric.graphgym.config import (
    cfg,
)
from src.utils.config import load_cfg
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device

from src.custom_graphgym import logger
import pandas as pd
from src.custom_graphgym.loader.protein import ProteinDataset
from src.custom_graphgym.logger import set_printing


def main():
    ''''
    This code is very similar to code in main.py. The reason to have two code with similar functionality is that: the
    '''
    parser = argparse.ArgumentParser(description='GrapePi: Promote unconfident proteins based on prediction probability.')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('--checkpoint', dest= 'checkpoint', type=str, default=None,
                        help='The checkpoint file path.')
    parser.add_argument('--threshold', dest='threshold', type=float, default=0.9,
                        help='The threshold to determine the unconfident proteins based on raw protein probability.')
    parser.add_argument('--labeled', dest='labeled', action='store_true',
                        help='Whether to apply promotion to labeled data as well. '
                             'if not provided, the promotion will only apply to the unlabeled proteins only')
    parser.add_argument('--num-promoted', dest='num_promoted', type=int, default=100,
                        help='The number, N, of proteins to be promoted '
                             'Top N Proteins will be selected based on the prediction probability from unconfident'
                             'proteins.')
    parser.add_argument('--output', dest='output', type=str, default=None,
                        help='The output file path to save the promoted proteins.')
    parser.add_argument('--override', dest='override', type=str, default=None,
                        help='Override the configurations from the configuration file. '
                             'This argument must be a single string with the format "key1=value1,key2=value2". '
                             'For overriding config of list type, use the format it has been additional nested by single quotation marker \'key1=[value1,value2]\'. '
                             'Example: "dataset.dir=data/yeast-LCQ \'dataset.node_numeric_cols=[protein_probability, mRNA]\' run.name=yeast_LCQ_sageconv dataset.label_col=y" ')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Load cmd line args
    args = parser.parse_args()
    # Load config file
    register.config_dict.get('grape_pi')(cfg) # update with custom config
    load_cfg(cfg, args)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for different random seeds
    set_printing()

    seed_everything(cfg.seed)
    auto_select_device()
    # Set machine learning pipeline

    # use the right customized datamodule and graphgymmodule
    # use the right customized datamodule and graphgymmodule
    if cfg.gnn.layer_type == 'sageconv':
        model = register.train_dict['sageconv_create_model']()
    elif cfg.gnn.layer_type == 'gcnconv' or cfg.gnn.layer_type == 'gcnconv_with_edgeweight':
        model = register.train_dict['gcnconv_create_model']()
    elif cfg.gnn.layer_type == 'gatconv':
        model = register.train_dict['gcnconv_create_model']()
    else:
        raise ValueError(f'Unsupported layer type {cfg.gnn.layer_type}. current support sageconv, gcnconv, gatconv, gcnconv_with_edgeweight')


    # Print model info
    logging.info(model)
    logging.info(cfg)

    if args.checkpoint is not None:
        print(f'Loading checkpoint from {args.checkpoint}')
        model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
        # ../saved_results/gastric-graphsage/epoch=199-step=4800.ckpt'
    else:
        # if no checkpoint is provided, retrain the model
        # run the graphgym.py script to train the model with provided configuration
        # create a directory to store the results that using the same as configuration file
        # get path part of the configuration file args.cfg_file
        cfg_path = os.path.dirname(args.cfg_file)
        # get file name part of the configuration file args.cfg_file
        cfg_file = os.path.basename(args.cfg_file)
        override_args = shlex.split(args.override) if args.override else []
        subprocess.run(['python', 'main.py', '--config-path', cfg_path, '--config-name', cfg_file, 'train.ckpt_clean=True'] + override_args)

        # load the last checkpoint
        # the path is output_path/0/ckpt/*.ckpt
        if cfg.run.name is None:
            output_path = os.path.join(cfg.out_dir, args.cfg_file.split('/')[-1].replace('.yaml', ''))
        else:
            output_path = os.path.join(cfg.out_dir, cfg.run.name)
        checkpoint_dir = os.path.join(output_path,  str(cfg.seed), 'ckpt')
        checkpoint_file = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))[0]
        model.load_state_dict(torch.load(checkpoint_file)['state_dict'])

    # prediction on the unconfident proteins
    warnings.filterwarnings('ignore', '.*use `CSVLogger` as the default.*')

    trainer = pl.Trainer(
        enable_checkpointing=cfg.train.enable_ckpt,
        default_root_dir=cfg.out_dir,
        max_epochs=cfg.optim.max_epoch,
        accelerator=cfg.accelerator,
        devices='auto' if not torch.cuda.is_available() else cfg.devices,
    )


    #  torch lightning put a return value in a list [] with just one element
    prediction = trainer.predict(model)
    all_pred_prob = prediction[0]['pred_prob']
    index = prediction[0]['index']

    # get the dictionary mapping from global node index to original protein accession
    # get the file in subdirectory "raw" of args.data and has "mapping" in the name
    # Use glob to get all files in the directory that have "mapping" in the name
    mapping = pd.read_csv(glob.glob(os.path.join(cfg.dataset.dir, '*mapping*'))[0])
    mapping = dict(zip(mapping['integer_id'], mapping['protein_id']))

    # convert the node index to original protein ID
    accession = [mapping[key] for key in index]

    # create a dataframe to store the prediction probability
    # accession is the original protein ID and not necessarily UniProt accession and can be any ID
    all_proteins_df = pd.DataFrame({'accession': accession, 'pred_prob': all_pred_prob})

    # read the original protein data
    # get the file in subdirectory "raw" of args.data and has "protein" in the name
    # Construct the path to the "raw/protein" subdirectory of args.data
    raw_dir = os.path.join(cfg.dataset.dir, 'raw/protein')
    # list all files in the directory
    dat = pd.read_csv(glob.glob(f'{raw_dir}/[!.]*')[0])
    # get the name of first column
    protein_id_col = dat.columns[0]
    # combine the test_proteins_df with the original protein data
    all_proteins_df = all_proteins_df.merge(dat, left_on='accession', right_on=protein_id_col, how='inner')

    # save the prediction probability of all proteins
    output_dir = os.path.join(args.output or cfg.dataset.dir, 'all_proteins.csv')
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    logger.info(f'Saving all proteins prediction probability to {output_dir}')
    all_proteins_df.to_csv(output_dir, index=False)

    # get the unconfident proteins and their prediction probability
    all_proteins_df = all_proteins_df.loc[:, [protein_id_col, 'pred_prob', cfg.dataset.label_col]]
    all_proteins_df.columns = ['accession', 'pred_prob','label']
    # slice unlabeled protein if cfg.dataset.label_column is NaN
    if not args.labeled:
        all_proteins_df = all_proteins_df[pd.isnull(all_proteins_df['label'])]

    unconfident_protein = all_proteins_df[(all_proteins_df.iloc[:, 1] < args.threshold)]

    # get the top N proteins to be promoted
    output_dir = os.path.join(args.output or cfg.dataset.dir, 'promoted_proteins.csv')
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    promoted_proteins = unconfident_protein.sort_values(by='pred_prob', ascending=False).head(args.num_promoted)
    promoted_proteins.to_csv(output_dir, index=False)
    # output message that the process completed and promoted proteins are saved
    logger.info(f'Top {args.num_promoted} proteins are saved in {output_dir}')


if __name__ == '__main__':
    main()
