#!/usr/bin/env python3

import torch
import numpy as np

import os
from pathlib import Path
import configargparse
from .densedepth_backend.model import create_model
from .densedepth_backend.utils import scale_up, predict

from core.experiment import ex

@ex.add_arguments
def cfg():
    backend = Path(__file__).parent/'densedepth_backend'
    parser = configargparse.get_argument_parser()
    group = parser.add_argument_group('DenseDepth', 'DenseDepth-specific params.')
    group.add('--densedepth-path', default=str(backend/'nyu.h5'))
    group.add('--gpu', type=str)
    # config, _ = parser.parse_known_args()
    # return vars(config)

@ex.setup('DenseDepth')
def setup(config):
    if config['gpu'] is not None and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
        print(f"Using gpu {config['gpu']} (CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}).")
    else:
        print("Using cpu.")
    return DenseDepth(weights_file=config['densedepth_path'])

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
