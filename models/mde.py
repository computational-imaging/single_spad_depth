#!/usr/bin/env python3

import torch
import numpy as np
import configargparse
from pathlib import Path
from pdb import set_trace

# MDEs
from .dorn import DORN
from .densedepth import DenseDepth

from ..experiment import ex

@ex.config('MDE')
def cfg():
    parser = configargparse.ArgParser(default_config_files=[str(Path(__file__).parent/'dorn_mde.cfg')])
    parser.add('--mde-config', is_config_file=True)
    parser.add('--mde', choices = ['DORN', 'DenseDepth', 'MiDaS'], required=True)
    parser.add('--img-key', required=True, default='image', help='Key in data corresponding to image')
    parser.add('--in-type', choices=['torch', 'numpy'], required=True, default='torch')
    parser.add('--in-order', choices=['nchw', 'nhwc'], required=True, default='nchw')
    parser.add('--out-type', choices=['torch', 'numpy'], required=True, default='torch')
    parser.add('--out-order', choices=['nchw', 'nhwc'], required=True, default='nchw')
    args, _ = parser.parse_known_args()
    return vars(args)

@ex.setup('MDE')
def setup(config):
    mde = ex.get_and_configure(config['mde'])
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
        out is a numpy array in NCHW format
        """
        img = data[self.key]
        if self.in_order == 'nhwc':
            img = img.permute(0, 2, 3, 1)
        if self.in_type == 'numpy':
            img = img.numpy()
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
