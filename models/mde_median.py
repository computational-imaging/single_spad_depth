#!/usr/bin/env python3

import torch
import numpy as np
import configargparse
from pathlib import Path
from pdb import set_trace

# MDEs
from .mde import MDE

from data.nyu_depth_v2.nyuv2_dataset import NYUV2_CROP
from core.experiment import ex


@ex.add_arguments('median')
def cfg():
    parser = configargparse.get_argument_parser()
    group = parser.add_argument_group('MDEMedian', 'MDE+Median method params.')
    group.add('--median-gt-key', default='depth_cropped')
    # args, _ = parser.parse_known_args()
    # return vars(args)

@ex.setup('median')
def setup(config):
    mde_model = ex.get_and_configure('mde')
    return MDEMedian(mde_model, gt_key=config['median_gt_key'])

@ex.entity
class MDEMedian:
    def __init__(self, mde_model, gt_key, crop=NYUV2_CROP):
        self.mde_model = mde_model
        self.gt_key = gt_key
        self.crop = crop

    def __call__(self, data):
        init = self.mde_model(data)
        if self.crop is not None:
            # Set median based on crop (makes a big difference)
            init_median = torch.median(init[...,
                                       self.crop[0]:self.crop[1],
                                       self.crop[2]:self.crop[3]])
        else:
            init_median = torch.median(init)
        init_median = init_median.to(data[self.gt_key].device)
        init = init.to(data[self.gt_key].device)
        pred = init * (torch.median(data[self.gt_key])/init_median)
        return pred

if __name__ == '__main__':
    from pdb import set_trace
    model = ex.get_and_configure('MDEMedian')
