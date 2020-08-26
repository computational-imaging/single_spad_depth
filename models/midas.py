#!/usr/bin/env python3

import torch
import numpy as np
import configargparse
from pdb import set_trace
from pathlib import Path
from .midas_backend.monodepth_net import MonoDepthNet
from .midas_backend.utils import resize_depth, resize_image

from ..experiment import ex

@ex.config('MiDaS')
def cfg():
    backend = Path(__file__).parent/'midas_backend'
    parser = configargparse.ArgParser(default_config_files=[str(backend/'midas.cfg')])
    parser.add('--midas-path', default=str(backend/'model.pt'))
    parser.add('--midas-min-depth', type=float, default=0.1)
    parser.add('--midas-max-depth', type=float, default=10.)
    args, _ = parser.parse_known_args()
    return vars(args)

@ex.setup('MiDaS')
def setup(config):
    return MiDaS(model_path=config['midas_path'],
                 min_depth=config['midas_min_depth'],
                 max_depth=config['midas_max_depth'])

@ex.entity
class MiDaS:
    def __init__(self, model_path, min_depth, max_depth):
        self.model = MonoDepthNet(model_path)
        self.model.eval()
        self.depth_range = (min_depth, max_depth)

    def __call__(self, image):
        """
        Predict depth from RGB
        :param model: a midas model
        :param img: numpy RGB image in 0-1 in NHWC order
        :param depth_range: Range of output depths (e.g. (0.1, 10) for NYUv2)
        :param device: torch.device object to run on
        :return: Inverse Depth image, same size as RGB image, scaled to be in the output range.
        """
        img = image.squeeze()
        depth_range = self.depth_range
        img_input = resize_image(img)

        # compute

        with torch.no_grad():
            idepth = self.model(img_input)

        idepth = resize_depth(idepth, img.shape[1], img.shape[0])

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
