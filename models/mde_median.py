#!/usr/bin/env python3

import numpy as np
import configargparse
from pathlib import Path

# MDEs
from .mde import MDE

from ..experiment import ex


@ex.config('MDEMedian')
def cfg():
    parser = configargparse.ArgParser(default_config_files=[str(Path(__file__).parent/'mde_median.cfg')])
    parser.add('--gt-key')
    args, _ = parser.parse_known_args()
    return vars(args)

@ex.setup('MDEMedian')
def setup(config):
    mde_model = ex.get_and_configure('MDE')
    return MDEMedian(mde_model, gt_key=config['gt_key'])

@ex.entity
class MDEMedian:
    def __init__(self, mde_model, gt_key):
        self.mde_model = mde_model
        self.gt_key = gt_key

    def __call__(self, data):
        init = self.mde_model(data)
        pred = init * (np.median(data[self.gt_key])/np.median(pred))
        return pred

if __name__ == '__main__':
    from pdb import set_trace
    model = ex.get_and_configure('MDEMedian')
