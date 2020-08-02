#!/usr/bin/env python3

from torch.utils.data import Dataset
import numpy as np

class NYUDepthv2(Dataset):
    def __init__(self, npz_file, **info):
        """
        info is for storing random metadata about this particular run,
        e.g. train or test.
        """
        super().__init__()
        self.info = info
        npz = np.load(npz_file)
        self.images = npz['images']
        self.rawDepths = npz['rawDepths']
        self.depths = npz['depths']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        return {
            'image': self.images[i, ...],
            'rawDepth': self.rawDepths[i, ...],
            'depth': self.depths[i, ...],
        }
