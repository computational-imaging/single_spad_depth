#!/usr/bin/env python3

import torch
import numpy as np

import os
from pathlib import Path
import configargparse
from .densedepth_backend.model import create_model
from .densedepth_backend.utils import scale_up, predict

from ..experiment import ex

@ex.config('DenseDepth')
def cfg():
    backend = Path(__file__).parent/'densedepth_backend'
    parser = configargparse.ArgParser(default_config_files=[str(backend/'densedepth.cfg')])
    parser.add('--densedepth-weights', default=str(backend/'nyu.h5'))
    config, _ = parser.parse_known_args()
    return vars(config)

@ex.setup('DenseDepth')
def setup(config):
    return DenseDepth(weights_file=config['densedepth_weights'])

@ex.entity
class DenseDepth:
    """
    DenseDepth Network

    https://github.com/ialhashim/DenseDepth

    Meant to be run as a part of a larger network.

    Only works in eval mode.

    Thin wrapper around the Keras implementation.
    """
    def __init__(self, weights_file=Path(__file__).parent/'densedepth_backend'/'nyu.h5'):
        self.model = create_model(weights_file)

    def __call__(self, image):
        """
        image is a numpy array in NHWC order
        pred_final is a numpy array in NHWC order
        """
        pred = scale_up(2, predict(self.model, image,
                                   minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0
        pred_flip = scale_up(2, predict(self.model, image[...,::-1,:],
                                        minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0
        pred_final = 0.5*pred + 0.5*pred_flip[:,:,::-1]
        return pred_final[..., np.newaxis]
