#!/usr/bin/env python3

from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

SPLIT_FILES ={
    'train': Path(__file__).parent/"processed"/"nyuv2_train.npz",
    'test': Path(__file__).parent/"processed"/"nyuv2_test.npz"
}

NYUV2_CROP = (20, 460, 24, 616)

class NYUDepthv2(Dataset):
    def __init__(self, npz_file, transform=None, **info):
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


def crop_image_and_depth(data, crop=NYUV2_CROP):
    for k in ['image', 'depth', 'rawDepth']:
        data[f'{k}_cropped'] = data[k][crop[0]:crop[1], crop[2]:crop[3]]
    return data


def get_dataloader(split, transform=None):
    """
    split should be either "train" or "test"
    """
    assert split in ('train', 'test')
    dataset = NYUDepthv2(SPLIT_FILES[split], transform=transform,
                         split=split)
    return DataLoader(dataset, batch_size=1)


if __name__ == "__main__":
    test_dataloader = get_dataloader("test", transform=crop_image_and_depth)
    for i, data in enumerate(test_dataloader):
        print(i)
        for k, v in data.items():
            print(f"{k}: {v.shape}")
        break
    print("done.")
