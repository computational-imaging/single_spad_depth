#!/usr/bin/env python3

import numpy as np
import configargparse
from pathlib import Path
from pdb import set_trace

# MDEs
from .mde import MDE

from ..experiment import ex

@ex.config('MDE')


@ex.entity
class MDEGTHist:
    def __init__(self, mde_model, gt_key, crop=NYUV2_CROP):
        self.mde_model = mde_model
        self.gt_key = gt_key
        self.crop = crop

    def __call__(self, data):
        init = self.mde_model(data)

    @staticmethod
    def hist_match():
        pass
