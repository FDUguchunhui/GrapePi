import sys
from typing import List, Dict, Iterable

import h5py
import numpy as np
import pandas as pd
import os.path as osp
import os

import torch
from overrides import overrides

import warnings
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.graphgym.register import register_loader
import torch_geometric.transforms as T
from typing import Any, Callable, List, Optional, Tuple, Union
from torch_geometric.data.dataset import to_list, _repr
from torch_geometric.graphgym.config import cfg
from torch_geometric.data.dataset import files_exist

from torch_geometric import  seed_everything

@register_loader('grape_pi')
def load_dataset_protein(format: object, name: object, dataset_dir: object) -> object:
    if format == 'PyG':
        if name == 'protein':
            dataset_raw = ProteinDataset(cfg.dataset.dir, node_numeric_cols=cfg.dataset.node_numeric_cols,
                                         label_col=cfg.dataset.label_col, rebuild=cfg.dataset.rebuild,
                                         interaction_weight_col=cfg.dataset.interaction_weight_col,
                                         interaction_conf_col=cfg.dataset.interaction_conf_col,
                                         interaction_conf_thresh=cfg.dataset.interaction_conf_thresh,
                                         num_val=cfg.dataset.split[1], num_test=cfg.dataset.split[2])
            # calculate pos_weight
            if cfg.train.pos_weight == -1:
                labeled = dataset_raw.y[~dataset_raw.unlabeled_mask]
                positive = labeled.sum()
                negative = len(labeled) - positive
                cfg.train.pos_weight = (negative / positive).detach().cpu().item()
            return dataset_raw


class ProteinDataset(InMemoryDataset):
    '''
    Args:
        root: the root directory to look for files
        node_numeric_cols: the numeric columns in the protein file used as feature
        label_col: the label column in the protein file used as label. If not provided,
        a reference folder contains "positive.txt" and "negative.txt" is required in "raw" folder
        remove_unlabeled_data: whether to remove unlabeled data
        rebuild: whether to rebuild the dataset even if the processed file already exist
        transform: the transform function to apply to the dataset
        pre_transform: the pre_transform function to apply to the dataset
        pre_filter: the pre_filter function to apply to the dataset
    '''

    def __init__(self, root,
                 node_numeric_cols: List[str],
                 edge_numeric_cols: Union[None, List[str]]=None,
                 label_col: Union[str, None]=None,
                 node_encoders=None,
                 edge_encoders=None,
                 include_seq_embedding=False, rebuild=False,
                 interaction_weight_col=None,
                 interaction_conf_col=None, interaction_conf_thresh=0.7,
                 transform=None, pre_transform=None, pre_filter=None,
                 num_val=0.2, num_test=0.1):

        self.include_seq_embedding = include_seq_embedding
        self.rebuild = rebuild
        self.node_encoders = node_encoders
        self.edge_encoders = edge_encoders
        if node_numeric_cols is None:
            raise ValueError('numeric_params is required for ProteinDataset')
        self.node_numeric_cols = node_numeric_cols
        self.edge_numeric_cols = edge_numeric_cols
        self.label_col = label_col
        self.interaction_weight_col = interaction_weight_col
        self.interaction_conf_col = interaction_conf_col
        self.interaction_conf_thresh = interaction_conf_thresh
        self.num_val = num_val
        self.num_test = num_test

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

        # seed_everything(seed)

    @property
    @overrides
    def raw_file_names(self):
        # the files required for this dataset will be handled in raw_paths function
        pass

    @property
    @overrides
    def processed_file_names(self):
        return 'data.pt'

    @overrides
    def process(self):
        # read data into data list
        data_list = [self._process_protein_file(protein_file) for protein_file in self.raw_paths['protein']]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # combine list of data into a big data object
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @overrides
    def download(self):
        raise Exception('Download is not supported for this type of dataset')

    def _process_protein_file(self, protein_filename: str) -> InMemoryDataset:
        '''
        Helper function for parse a single protein file to a torch.geometry.dataset using
        the given interaction file
        Args:
            protein_filename: the absolute path to the protein file

        Returns:
            torch.geometry.dataset
        '''

        positive_reference_list, negative_reference_list = None, None
        if self.label_col is None:
            # get positive/and negative reference
            with open(self.raw_paths['positive_reference']) as f:
                self.positive_protein_reference = f.read().splitlines()
            with open(self.raw_paths['negative_reference']) as f:
                self.negative_protein_reference = f.read().splitlines()

        # x is feature tensor for nodes
        x, mapping, y, unlabeled_mask = self._load_node_csv(path=protein_filename)

        # add protein sequence embedding
        if  self.include_seq_embedding:
            # create an empty tensor to store sequence embedding
            seq_embedding = torch.zeros((len(mapping), 1024))
            with h5py.File(self.raw_paths['embedding'], 'r') as file:
                # iterate through the file to create
                for accession, index  in mapping.items():
                    # if the protein is not in the embedding file, skip
                    # it will have an embedding vector of all zeros
                    if accession not in file:
                        continue
                    seq_embedding[index] = torch.from_numpy(np.array(file[accession]))
            x = torch.hstack([x, seq_embedding])

        # read protein-protein-interaction data (the last file from self.raw_file_names)
        edge_index, edge_attr, edge_weight = self._load_edge_csv(path=self.raw_paths['interaction'], mapping=mapping)
        if edge_weight is not None:
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, split=1, edge_attr=edge_attr, y=y,
                    unlabeled_mask=unlabeled_mask)
        else:
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, split=1, y=y, unlabeled_mask=unlabeled_mask)

        # split data into train, val, and test set
        # rename the attributes to match name convention in GraphGym
        split_transformer = T.RandomNodeSplit(split='train_rest', num_splits=1,
                                              num_val=self.num_val,
                                              num_test=self.num_test)
        data = split_transformer(data)

        # store mapping information for translate back protein integer ID back to string ID
        # had better save mapping somewhere else
        base_name = os.path.basename(protein_filename)
        name_without_suffix = os.path.splitext(base_name)[0]
        mapping_df =  pd.DataFrame(mapping.items(), columns=['protein_id', 'integer_id'])
        mapping_df['train_mask'] = data.train_mask.to(torch.int)
        mapping_df['val_mask'] = data.val_mask.to(torch.int)
        mapping_df['test_mask'] = data.test_mask.to(torch.int)
        mapping_df.to_csv(
            os.path.join(os.path.dirname(self.raw_dir), name_without_suffix + '_mapping.csv'), index=False)

        return data


    def _load_node_csv(self, path: str,
                       **kwargs) -> tuple['x', 'mapping', 'y', 'unlabeled_mask']:

        # file can be either csv or tsv
        if 'tsv' in os.path.basename(path):
            df = pd.read_csv(path, index_col=0, sep='\t', **kwargs)
        else:
            df = pd.read_csv(path, index_col=0, **kwargs)

        # if label is provided, use label column to create y
        y = df[self.label_col].to_numpy()

        # if choose to keep unlabeled data, create a mask for unlabeled data
        # make the unlabeled data to have label 0. This is because the code in
        # package torch_geometric only accept label in range 0 to num_classes - 1

        # create a mask for proteins without label
        unlabeled_mask = pd.isnull(y)
        # fill the unlabeled proteins with 0 since graphgym module only accept label in range 0 to num_classes - 1
        y = np.nan_to_num(y, nan=0)
        unlabeled_mask = torch.tensor(unlabeled_mask, dtype=torch.bool).view(-1)

        # after no NaN in y, convert y to integer
        y = y.astype(int)


        # create mapping from protein ID to integer ID
        # the mapping dict is needed to convert results back to protein ID
        mapping = {index: i for i, index in enumerate(df.index.unique())}

        # convert protein ID to integer ID
        x = torch.tensor(df.loc[:, self.node_numeric_cols].values, dtype=torch.float)
        if self.node_encoders is not None:
            xs = [encoder(df[col]) for col, encoder in self.node_encoders.items()]
            x2 = torch.cat(xs, dim=-1).view(-1, 1)
            x = torch.hstack([x, x2])
        # reshape to [n, 1] if x is 1D tensor
        if x.dim() == 1:
            x = x.view(-1, 1)

        # remove last dimension in y to make it a 1D tensor
        y = torch.tensor(y).view(-1).to(dtype=torch.int)

        return x, mapping, y, unlabeled_mask

    def _load_edge_csv(self, path: str, mapping: dict):
        if 'tsv' in os.path.basename(path):
            df = pd.read_csv(path, sep='\t')
        else:
            df = pd.read_csv(path)

        if self.interaction_conf_col is not None:
            df = df[df[self.interaction_conf_col] >= self.interaction_conf_thresh]
        # only keep interactions related to proteins that in the protein dataset (i.e. in mapping keys)
        protein_data_acc = mapping.keys()
        df = df[df.iloc[:, 0].isin(protein_data_acc) & df.iloc[:, 1].isin(protein_data_acc)]
        # check if df is empty
        if df.empty:
            raise Exception('No interaction data left after thresh filtering or overlapping with protein data')

        #   convert protein ID to integer ID
        src = [mapping[index] for index in df.iloc[:, 0]]
        dst = [mapping[index] for index in df.iloc[:, 1]]
        edge_index = torch.tensor([src, dst])

        edge_weight = None
        if self.interaction_weight_col is not None:
            edge_weight = torch.tensor(df.loc[:, self.interaction_weight_col].values, dtype=torch.float)

        edge_attr = None
        if self.edge_numeric_cols is not None:
            edge_attr = torch.tensor(df.loc[:, self.numeric_cols].values, dtype=torch.float)

        if self.edge_encoders is not None:
            edge_attrs = [encoder(df[col]) for col, encoder in self.edge_encoders.items()]
            edge_attr = torch.cat(edge_attrs, dim=-1)

        return edge_index, edge_attr, edge_weight

    @property
    def raw_paths(self) -> Dict[str, List[str]]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        # our data folder have special structure the original raw_paths will only used for check if the files exist
        raw_paths_dict = {}

        # generate necessary file paths
        raw_protein_dir = osp.join(self.root, 'raw/protein')
        file_names = [f for f in os.listdir(raw_protein_dir) if not f.startswith('.')]
        if len(file_names) == 0:
            raise Exception('no protein file detected!')
        else:
            protein_file_paths = [os.path.abspath(os.path.join(raw_protein_dir, file_name))
                                  for file_name in file_names]
        raw_paths_dict['protein'] = protein_file_paths

        raw_interaction_dir = osp.join(self.root, 'raw/interaction')
        file_names = [f for f in os.listdir(raw_interaction_dir) if not f.startswith('.')]
        if len(file_names) != 1:
            raise Exception('Wrong number of interaction file detected! Expecting exactly one file.')
        else:
            interaction_file_path = os.path.abspath(os.path.join(raw_interaction_dir, file_names[0]))
        raw_paths_dict['interaction'] = interaction_file_path

        if self.include_seq_embedding:
            raw_embedding_dir = osp.join(self.root, 'raw/embedding')
            file_names = [f for f in os.listdir(raw_embedding_dir) if not f.startswith('.')]
            if len(file_names) != 1:
                raise Exception('Wrong number of embedding file detected! Expecting exactly one file.')
            else:
                embedding_file_path = os.path.abspath(os.path.join(raw_embedding_dir, file_names[0]))
            raw_paths_dict['embedding'] = embedding_file_path

        return raw_paths_dict

    @overrides
    def _download(self):
        # check if the protein files exist otherwise raise exception
        if not files_exist(self.raw_paths['protein']):
            raise Exception('Protein file not found! Not supported for automatic download.')
        # check interaction file exist otherwise download from STRING database
        # elf.raw_paths['interaction'] is a str so len is 0 and files_exist will return False
        if not osp.exists(self.raw_paths['interaction']):
            os.makedirs(os.path.dirname(self.raw_paths['interaction']), exist_ok=True)
            self.download()

    @overrides
    def _process(self):
        f = osp.join(self.processed_dir, 'pre_transform.pt')
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first")

        f = osp.join(self.processed_dir, 'pre_filter.pt')
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, make sure to "
                "delete '{self.processed_dir}' first")


        if files_exist(self.processed_paths) and not self.rebuild:  # pragma: no cover
            return

        if self.log and 'pytest' not in sys.modules:
            if self.rebuild:
                print('Rebuilding...', file=sys.stderr)
            else:
                print('Processing...', file=sys.stderr)

        if self.rebuild:
            os.makedirs(self.processed_dir, exist_ok=True)
        else:
            os.makedirs(self.processed_dir)

        self.process()

        path = osp.join(self.processed_dir, 'pre_transform.pt')
        torch.save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, 'pre_filter.pt')
        torch.save(_repr(self.pre_filter), path)

        if self.log and 'pytest' not in sys.modules:
            print('Done!', file=sys.stderr)



# for testing purpose
if __name__ == '__main__':
    import argparse
    from torch_geometric.loader import DataLoader

    parser = argparse.ArgumentParser(
        prog='ImportProteinData')

    parser.add_argument('-r', '--root', required=True,
                        help='the root directory to look for files')
    parser.add_argument('-n', '--numeric-columns', nargs='+', required=True,
                        help='the numeric columns in the protein file used as feature')
    parser.add_argument('-l', '--label-col', required=False, default=None,
                        help='the label column in the protein file used as label. If not provided, '
                             'a reference folder contains "positive.txt" and "negative.txt" is required in "raw" folder')
    parser.add_argument('-i', '--include-seq-embedding', required=False, default=False, action='store_true',
                        help='whether to include protein sequence embedding')
    parser.add_argument('-b', '--rebuild', required=False, default=False, action='store_true',
                        help='whether to rebuild the dataset even if the processed file already exist')

    args = parser.parse_args()

    protein_dataset = ProteinDataset(root=args.root,
                                     node_numeric_cols=args.numeric_columns,
                                     label_col=args.label_col,
                                     include_seq_embedding=args.include_seq_embedding,
                                     rebuild=args.rebuild)
    # loader = DataLoader(protein_dataset)
    # for data in loader:
    #     data
    print('Successfully run')
