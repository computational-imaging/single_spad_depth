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

@ex.add_arguments('gt_hist')
def cfg():
    parser = configargparse.get_argument_parser()
    group = parser.add_argument_group('MDEGTHist', 'MDE+GT Hist matching params.')
    group.add('--gt-hist-gt-key', default='depth_cropped')
    # args, _ = parser.parse_known_args()
    # return vars(args)

@ex.setup('gt_hist')
def setup(config):
    mde_model = ex.get_and_configure('mde')
    return MDEGTHist(mde_model, config['gt_hist_gt_key'])

@ex.entity
class MDEGTHist:
    def __init__(self, mde_model, gt_key, crop=NYUV2_CROP):
        self.mde_model = mde_model
        self.gt_key = gt_key
        self.crop = crop

    def __call__(self, data):
        init = self.mde_model(data).cpu().numpy().squeeze()
        if self.crop is not None:
            init = init[...,
                        self.crop[0]:self.crop[1],
                        self.crop[2]:self.crop[3]]
        template = data[self.gt_key].cpu().numpy().squeeze()
        pred = self.hist_match(init, template)
        return torch.from_numpy(pred)

    @staticmethod
    def hist_match(source, template):
        """
        From https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x

        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image

        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the flattened
                array
            template: np.ndarray
                Template image; can have different dimensions to source
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
        """

        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)
