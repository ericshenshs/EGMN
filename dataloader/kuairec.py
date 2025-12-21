"""dataloader/kuairec.py.

This file is part of the watch-time prediction codebase.
Primary role: dataloader.
Defines dataset/dataloader wrappers that produce feature dicts and labels for training.
"""

import numpy as np
import pandas as pd
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader


class KUAIRECDataset(Dataset):
    """KUAIRECDataset.
    
    Dataset/DataLoader wrapper for the preprocessed KuaiRec pickle artifact.
    Responsible for: dtype casting (embedding indices vs floats), device placement, and
    returning (features_dict, label) pairs compatible with model.forward().
    """
    # Notes:
    # - Defines the features_dict structure consumed by all models; changing keys/dtypes breaks model.forward().
    # - Moves tensors to the requested device up front to avoid per-batch device transfers.
    # - Casts per schema so embedding layers get integer indices and masks/ctn features are float32.
    def __init__(self, dataset_name, df, description, device):
        """__init__.
        
        Materializes a preprocessed pandas DataFrame into device tensors.
        
        Args:
        - dataset_name: dataset identifier (used for bookkeeping/logging).
        - df: pandas DataFrame containing already-encoded features and labels.
        - description: schema list of (name, size, type) defining feature roles and dtypes.
        - device: torch device where tensors should live (CPU or GPU).
        
        Notes:
        - The DataFrame is assumed to already be encoded by preprocessing (integer IDs for categories,
          padded sequences + masks, normalized continuous features).
        """
        # Notes:
        # - Materializes each DataFrame column into a tensor once (avoids per-sample conversion overhead).
        # - Calls format() to enforce schema-driven dtypes before any model sees the data.
        super(KUAIRECDataset, self).__init__()
        self.dataset_name = dataset_name
        self.df = df
        self.length = len(df)
        # Materialize each column as a tensor on target device for fast __getitem__.
        self.name2array = {name: torch.from_numpy(np.array(list(df[name])).reshape([self.length, -1])).to(device) \
                                        for name in df.columns}
        self.format(description, device)
        self.features = [name for name, size, type in description if type != 'label']
        self.label = 'play_time'

    def format(self, description, device):
        """format.
        
        Casts each stored column tensor to the dtype expected by downstream models.
        - spr/seq: torch.long (embedding indices).
        - ctn/seqm/other: torch.float32 (linear features / masks).
        """
        # Notes:
        # - Schema-driven dtype casting prevents embedding lookup errors and silent float->long issues.
        # - Masks are float32 so they can be multiplied with embeddings and summed safely.
        for name, size, type in description:
            if type == 'spr' or type == 'seq':
                self.name2array[name] = self.name2array[name].to(torch.long)
            elif type == 'ctn' or type == 'seqm' or type == 'other':
                self.name2array[name] = self.name2array[name].to(torch.float32)
            elif type == 'label':
                pass
            else:
                raise ValueError('unkwon type {}'.format(type))
                
    def __getitem__(self, index):
        """__getitem__.
        
        Returns one sample as (features_dict, label).
        features_dict contains all non-label feature tensors for a single row.
        label is the normalized play_time scalar for that row.
        """
        # Notes:
        # - Returns a dict of feature tensors for a single row; models assume this dict is complete.
        # - Label is the normalized play_time scalar (squeezed).
        return {name: self.name2array[name][index] for name in self.features}, \
                self.name2array[self.label][index].squeeze()

    def __len__(self):
        """__len__.
        
        Function defined in dataloader/kuairec.py.
        Args: self.
        """
        # Notes:
        # - Length equals number of rows in the underlying DataFrame.
        return self.length


class KUAIRECDataLoader(object):
    """KUAIRECDataLoader.
    
    Dataset/DataLoader wrapper for the preprocessed KuaiRec pickle artifact.
    Responsible for: dtype casting (embedding indices vs floats), device placement, and
    returning (features_dict, label) pairs compatible with model.forward().
    """
    # Notes:
    # - Defines the features_dict structure consumed by all models; changing keys/dtypes breaks model.forward().
    # - Moves tensors to the requested device up front to avoid per-batch device transfers.
    # - Casts per schema so embedding layers get integer indices and masks/ctn features are float32.

    def __init__(self, dataset_name, dataset_path, device, bsz=32):
        """__init__.
        
        Loads the serialized dataset artifact and builds split-specific DataLoader objects.
        
        Args:
        - dataset_name: dataset identifier (e.g., \"kuairec\").
        - dataset_path: path to the pickle produced by preprocessing (contains train/test/description).
        - device: torch device where batches should be placed.
        - bsz: batch size for each split DataLoader.
        
        Side effects:
        - Reads the pickle file from disk.
        - Materializes per-split tensors on the given device via KUAIRECDataset.
        """
        # Notes:
        # - Materializes each DataFrame column into a tensor once (avoids per-sample conversion overhead).
        # - Calls format() to enforce schema-driven dtypes before any model sees the data.
        # Fail early with a clear message if preprocessing artifacts are missing.
        assert os.path.exists(dataset_path), '{} does not exist'.format(dataset_path)
        with open(dataset_path, 'rb+') as f:
            data = pickle.load(f)
        self.dataset_name = dataset_name
        self.dataloaders = {}
        self.description = data['description']
        for key, df in data.items():
            if key == 'description':
                continue
            # Keep shuffle=False to make evaluation deterministic; training scripts can shuffle if desired.
            self.dataloaders[key] = DataLoader(
                KUAIRECDataset(dataset_name, df, self.description, device),
                batch_size=bsz,
                shuffle=False,
            )
        self.keys = list(self.dataloaders.keys())
                                

    def __getitem__(self, name):
        """__getitem__.
        
        Returns one sample as (features_dict, label).
        features_dict contains all non-label feature tensors for a single row.
        label is the normalized play_time scalar for that row.
        """
        # Notes:
        # - Returns a dict of feature tensors for a single row; models assume this dict is complete.
        # - Label is the normalized play_time scalar (squeezed).
        assert name in self.keys, '{} not in keys of datasets'.format(name)
        return self.dataloaders[name]