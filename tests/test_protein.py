from unittest.mock import Mock

import pytest
import torch

from src.custom_graphgym.loader.protein import ProteinDataset


def test_protein_dataset_using_reference():
    data = ProteinDataset('data/yeast-ORBI', node_numeric_cols=['protein_probability', 'mRNA'], rebuild=True)
    assert data is not None

def test_protein_dataset_using_label():
    data = ProteinDataset('data/yeast-ORBI', node_numeric_cols=['protein_probability', 'mRNA'],
                          label_col='y', rebuild=True)
    assert data is not None

def test_protein_dataset_using_label():
    data = ProteinDataset('data/yeast-ORBI', node_numeric_cols=['protein_probability', 'mRNA'],
                          label_col='y', rebuild=True, )
    assert data is not None

def test_protein_dataset_with_only_one_numeric_col():
    data = ProteinDataset('data/yeast-ORBI', node_numeric_cols=['protein_probability'], rebuild=True)
    assert data.x.shape[1] == 1
    # test is tensor.float
    assert data.x.dtype == torch.float32

def test_protein_dataset_with_edge_weight():
    data = ProteinDataset('data/yeast-ORBI', node_numeric_cols=['protein_probability'], rebuild=True, interaction_weight_col='combined_score')
    assert data.edge_weight is not None
    # test edge_weight is tensor.float
    assert data.edge_weight.dtype == torch.float32