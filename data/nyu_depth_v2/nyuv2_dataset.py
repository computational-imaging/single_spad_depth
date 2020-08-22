#!/usr/bin/env python3

from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from ...experiment import ex

SPLIT_FILES = {
    'train': Path(__file__).parent/"processed"/"nyuv2_train.npz",
    'test': Path(__file__).parent/"processed"/"nyuv2_test.npz"
}

TRANSIENT_FILES = {
    'train': Path(__file__).parent/"processed"/"counts_train.npz",
    'test': Path(__file__).parent/"processed"/"counts_test.npz"
}

NYUV2_CROP = (20, 460, 24, 616)

@ex.entity('NYUDepthv2')
class NYUDepthv2(Dataset):
    def __init__(self, split, transform=None, **info):
        """
        info is for storing random metadata about this particular run,
        e.g. train or test.
        """
        super().__init__()
        self.split = split
        self.info = info
        npz = np.load(SPLIT_FILES[split])
        self.images = npz['images']
        self.rawDepths = npz['rawDepths']
        self.depths = npz['depths']
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        data = {
            'image': self.images[i, ...],
            'rawDepth': self.rawDepths[i, ...],
            'depth': self.depths[i, ...],
        }
        if self.transform is None:
            return data
        else:
            return self.transform(data)

@ex.entity('NYUDepthv2Transient')
class NYUDepthv2Transient(Dataset):
    def __init__(self, split, sbr, transform=None, **info):
        super().__init__()
        self.nyuv2 = NYUDepthv2(split, transform=transform, **info)
        self.info = info
        npz = np.load(TRANSIENT_FILES[split], allow_pickle=True)
        self.transient = npz[str(sbr)]
        self.transform = transform

    def __len__(self):
        return len(self.nyuv2)

    def __getitem__(self, i):
        data = self.nyuv2[i]
        data['transient'] = self.transient[i, ...]
        if self.transform is None:
            return data
        else:
            return self.transform(data)

@ex.transform('crop_image_and_depth')
def crop_image_and_depth(data, crop=NYUV2_CROP):
    for k in ['image', 'depth', 'rawDepth']:
        data[f'{k}_cropped'] = data[k][crop[0]:crop[1], crop[2]:crop[3]]
    return data

if __name__ == "__main__":
    test_dataloader = DataLoader(NYUDepthv2("test", transform=crop_image_and_depth),
                                 batch_size=1)
    print("No transient")
    data = iter(test_dataloader).next()
    for k, v in data.items():
        print(k, v.shape)
    test_dataloader = DataLoader(NYUDepthv2Transient("test", sbr=5., transform=crop_image_and_depth),
                                 batch_size=1)
    print("With transient")
    data = iter(test_dataloader).next()
    for k, v in data.items():
        print(k, v.shape)
    print("done.")
