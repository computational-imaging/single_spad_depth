#!/usr/bin/env python3

import torch
import numpy as np
import configargparse
from pathlib import Path
from pdb import set_trace

# MDEs
from models.dorn import DORN
from models.densedepth import DenseDepth
from models.midas import MiDaS

from core.metrics import get_depth_metrics
from core.experiment import ex

@ex.add_arguments('mde')
def cfg():
    parser = configargparse.get_argument_parser()
    group = parser.add_argument_group('MDE', 'MDE Wrapper config')
    group.add('--mde', choices = ['DORN', 'DenseDepth', 'MiDaS'], default='DORN')
    group.add('--img-key', default='dorn_image', help='Key in data corresponding to image')
    group.add('--in-type', choices=['torch', 'numpy'], default='torch')
    group.add('--in-order', choices=['nchw', 'nhwc'], default='nchw')
    group.add('--out-type', choices=['torch', 'numpy'], default='torch')
    group.add('--out-order', choices=['nchw', 'nhwc'], default='nchw')
    # args, _ = parser.parse_known_args()
    # return vars(args)

@ex.setup('mde')
def setup(config):
    mde = ex.get_and_configure(config['mde'])
    # if mde == 'DORN':
    #     img_key = 'dorn_image'
    #     in_type = 'torch'
    #     in_order = 'nchw'
    #     out_type = 'torch'
    #     out_order = 'nchw'
    # elif mde == 'DenseDepth':
    #     img_key = 'image'
    #     in_type = 'numpy'
    #     in_order = 'nhwc'
    #     out_type = 'numpy'
    #     out_order = 'nhwc'
    # elif mde == 'MiDaS':
    #     img_key = 'midas_image'
    #     in_type = 'torch'
    #     in_order = 'nhwc'
    #     out_type = 'numpy'
    #     out_order = 'nhwc'
    return MDE(mde,
               key=config['img_key'],
               in_type=config['in_type'],
               in_order=config['in_order'],
               out_type=config['out_type'],
               out_order=config['out_order'])

@ex.entity
class MDE:
    """
    Wrapper for an MDE model, handles channels and input/output datatypes.
    """
    def __init__(self, mde, key,
                 in_type='torch', in_order='nchw',
                 out_type='torch', out_order='nchw'):
        self.mde = mde
        self.key = key
        self.in_type = in_type # Input type required by mde
        self.in_order = in_order  # Input channel location required by mde
        self.out_type = out_type # Output type of mde
        self.out_order = out_order # Output channel location of mde

    def __call__(self, data):
        """
        Wraps the mde so that it takes torch input and produces numpy output
        image_tensor is a torch float tensor in [0, 1] in NCHW format from the torch dataloader
        out is a torch tensor in NCHW format
        """
        img = data[self.key]
        if self.in_order == 'nhwc':
            img = img.permute(0, 2, 3, 1)
        if self.in_type == 'numpy':
            img = img.cpu().numpy()
        out = self.mde(img)
        if self.out_type == 'numpy':
            out = torch.from_numpy(out)
        if self.out_order == 'nhwc':
            out = out.permute(0, 3, 1, 2)
        return out

if __name__ == '__main__':
    from pdb import set_trace
    mde_model = ex.get_and_configure('MDE')
    test = {'image': torch.randn(1, 3, 480, 640)}
    output = mde_model(test)
    print(output.shape)
    print(type(output))
