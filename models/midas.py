#!/usr/bin/env python3

import torch
import numpy as np
import configargparse
from pdb import set_trace
from pathlib import Path
from .midas_backend.monodepth_net import MonoDepthNet
from .midas_backend.utils import resize_depth, resize_image
import os

from core.experiment import ex

@ex.add_arguments
def cfg():
    backend = Path(__file__).parent/'midas_backend'
    parser = configargparse.get_argument_parser()
    group = parser.add_argument_group('MiDaS', 'MiDaS-specific params.')
    group.add('--midas-path', default=str(backend/'model.pt'))
    group.add('--midas-full-width', type=int, default=480)
    group.add('--midas-full-height', type=int, default=640)
    group.add('--midas-min-depth', type=float, default=0.1)
    group.add('--midas-max-depth', type=float, default=10.)
    group.add('--gpu', type=str)
    # args, _ = parser.parse_known_args()
    # return vars(args)

@ex.setup('MiDaS')
def setup(config):
    midas = MiDaS(model_path=config['midas_path'],
                  full_width=config['midas_full_width'],
                  full_height=config['midas_full_height'],
                  min_depth=config['midas_min_depth'],
                  max_depth=config['midas_max_depth'])
    if config['gpu'] is not None and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
        midas.model.to('cuda')
        midas.device='cuda'
        print(f"Using gpu {config['gpu']} (CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}).")
    else:
        print("Using cpu.")
    return midas

@ex.transform('midas_preprocess')
def midas_preprocess(data):
    """data['image'] is HWC"""
    data['midas_image'] = resize_image(data['image'])
    return data


@ex.entity
class MiDaS:
    def __init__(self, model_path, full_width, full_height, min_depth, max_depth, device='cpu'):
        self.model = MonoDepthNet(model_path)
        self.model.eval()
        self.full_shape = (full_height, full_width)
        self.depth_range = (min_depth, max_depth)
        self.device = device

    def __call__(self, image):
        """
        Predict depth from RGB
        :param model: a midas model
        :param img: numpy RGB image in 0-1 in NHWC order
        :param depth_range: Range of output depths (e.g. (0.1, 10) for NYUv2)
        :param device: torch.device object to run on
        :return: Inverse Depth image, same size as RGB image, scaled to be in the output range.
        """
        depth_range = self.depth_range

        with torch.no_grad():
            idepth = self.model(image)

        idepth = resize_depth(idepth, self.full_shape[0], self.full_shape[1])

        idepth_min = idepth.min()
        idepth_max = idepth.max()

        # Arbitrarily cap at 1./depth_range[1] + 10 for "infinite depth"
        idepth_range = (1. / depth_range[1],
                        1. / depth_range[0] if depth_range[0] > 0 else 1. / depth_range[1] + 10.)
        idepth_scaled = (idepth_range[1] - idepth_range[0]) * \
                        (idepth - idepth_min) / (idepth_max - idepth_min) + \
                        idepth_range[0]
        depth_scaled = 1. / idepth_scaled
        return depth_scaled[np.newaxis, ..., np.newaxis]
