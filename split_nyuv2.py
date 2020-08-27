#!/usr/bin/env python3

from configargparse import ArgumentParser
from pdb import set_trace
import h5py
import os
from scipy.io import loadmat
import numpy as np
import skimage.io as sio
from pathlib import Path

nyuv2_dir = Path(__file__).parent/'data'/'nyu_depth_v2'
parser = ArgumentParser()
parser.add_argument('--main-file',
                    default=nyuv2_dir/'raw'/'nyu_depth_v2_labeled.mat')
parser.add_argument('--split-file',
                    default=nyuv2_dir/'raw'/'splits.mat')
parser.add_argument('--output-dir',
                    default=nyuv2_dir/'processed')


if __name__ == '__main__':
    params = parser.parse_args()
    # Main file
    print(f"Loading NYUv2 from {params.main_file}...")
    f = h5py.File(params.main_file, 'r')
    depths = np.array(f['depths'])
    rawDepths = np.array(f['rawDepths'])
    images = np.array(f['images'])

    # Split file
    print(f"Loading split file from {params.split_file}...")
    splits = loadmat(params.split_file)
    # Subtract 1 because MATLAB is 1-indexed
    trainNdxs = splits['trainNdxs'].squeeze() - 1
    testNdxs = splits['testNdxs'].squeeze() - 1

    def split_and_write(Ndxs, filepath):
        data = {
            'depths': depths[Ndxs, ...].transpose(0, 2, 1),
            'rawDepths': rawDepths[Ndxs, ...].transpose(0, 2, 1),
            'images': images[Ndxs, ...].transpose(0, 3, 2, 1),
            }
        output_dir = os.path.dirname(filepath)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        np.savez(filepath, **data)
        return
    print("Splitting and writing train set...")
    split_and_write(trainNdxs, Path(params.output_dir)/"nyuv2_train.npz")
    print("Spliting and writing test set...")
    split_and_write(testNdxs, Path(params.output_dir)/"nyuv2_test.npz")
    print("Done.")
