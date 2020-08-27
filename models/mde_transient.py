#!/usr/bin/env python3

import configargparse
import numpy as np
import torch
from pathlib import Path
from pdb import set_trace
from models.discretize import SI, rescale_hist, Uniform
from data.nyu_depth_v2.nyuv2_dataset import NYUV2_CROP

from core.experiment import ex

@ex.config('MDETransient')
def cfg():
    parser = configargparse.ArgParser(default_config_files=[str(Path(__file__).parent/'dorn_transient.cfg')])
    parser.add('--mde-transient-config', is_config_file=True)
    parser.add('--refl-est', choices=['gray', 'red'], required=True)
    parser.add('--source-n-sid-bins', type=int, default=68)
    parser.add('--n-ambient-bins', type=int, default=60)
    parser.add('--edge-coeff', type=float, default=5.)
    parser.add('--n-std', type=int, default=1)
    parser.add('--min-depth', type=float, default=0.)
    parser.add('--max-depth', type=float, default=10.)
    parser.add('--source-alpha', type=float, default=0.6569154266167957)
    parser.add('--source-beta', type=float, default=9.972175646365525)
    parser.add('--source-offset', type=float, default=0.)
    args, _ = parser.parse_known_args()
    return vars(args)

@ex.setup('MDETransient')
def setup(config):
    mde_model = ex.get_and_configure('MDE')
    preproc = TransientPreprocessor(config['source_n_sid_bins'],
                                    config['n_ambient_bins'],
                                    config['edge_coeff'],
                                    config['n_std'])
    if config['refl_est'] == 'gray':
        refl_est = refl_est_gray
    elif config['refl_est'] == 'red':
        refl_est = refl_est_red
    source_disc = SI(config['source_n_sid_bins'],
                     config['source_alpha'],
                     config['source_beta'],
                     config['source_offset'])
    return MDETransient(mde_model=mde_model,
                        preproc=preproc,
                        refl_est=refl_est,
                        source_disc=source_disc,
                        min_depth=config['min_depth'],
                        max_depth=config['max_depth'])

@ex.entity
class MDETransient:
    def __init__(self, mde_model, preproc, refl_est, source_disc,
                 min_depth, max_depth, crop=NYUV2_CROP,
                 image_key='image', transient_key='transient'):
        self.mde_model = mde_model       # MDE function, torch (channels first) -> torch (single channel)
        self.refl_est = refl_est         # Reflectance estimator, torch -> torch
        self.preproc = preproc           # Preprocessing function, np -> np
        self.source_disc = source_disc   # Discretization for reflectance-weighted depth hist
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.crop = crop
        self.image_key = image_key         # Key in data for RGB image
        self.transient_key = transient_key # Key in data for transient

    def apply_crop(self, img):
        return img[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]

    def __call__(self, data):
        depth_init = self.mde_model(data).cpu().numpy().squeeze()
        reflectance_est = self.refl_est(data[self.image_key]).cpu().numpy().squeeze()
        transient = data[self.transient_key].cpu().numpy().squeeze()
        # Pre-crop
        depth_init = self.apply_crop(depth_init)
        reflectance_est = self.apply_crop(reflectance_est)

        # compute weighted depth hist
        source_hist = self.weighted_histogram(depth_init, reflectance_est)

        counts_disc = Uniform(len(transient), self.min_depth, self.max_depth)
        target_hist, target_disc = self.preproc(transient, counts_disc)
        depth_final = \
            self.hist_match(depth_init,
                            source_hist, self.source_disc,
                            target_hist, target_disc)
        depth_final = torch.from_numpy(depth_final)
        return depth_final

    def weighted_histogram(self, depth, weights):
        depth_hist, _ = np.histogram(depth, weights=weights,
                                     bins=self.source_disc.bin_edges)
        return depth_hist

    def hist_match(self, depth_init,
                   source_hist, source_disc,
                   target_hist, target_disc):
        movement = self.find_movement(source_hist, target_hist)
        depth_final = self.move_pixels(depth_init, movement,
                                       source_disc, target_disc)
        return depth_final

    def find_movement(self, source_hist, target_hist):
        """Gives the movements from source_hist (column sum)
        to hist_to (row sum).

        Based on Morovic et. al 2002 A fast, non-iterative, and exact histogram
        matching algorithm.
        """
        # Normalize to same sum
        source_hist /= np.sum(source_hist)
        target_hist /= np.sum(target_hist)

        movement = np.zeros((len(source_hist), len(target_hist)))
        for row in range(len(source_hist)):
            for col in range(len(target_hist)):
                pixels_rem = source_hist[row] - np.sum(movement[row, :col])
                pixels_req = target_hist[col] - np.sum(movement[:row, col])
                movement[row, col] = np.clip(np.minimum(pixels_rem, pixels_req), a_min=0., a_max=None)
        return movement

    def move_pixels(self, depth_init, movement, source_disc, target_disc):
        index_init = source_disc.get_sid_index_from_value(depth_init)
        cpfs = np.cumsum(movement, axis=1)  # Cumulative sum across target dimension
        pixel_cpfs = cpfs[index_init, :] # [H, W, cpf dim]
        p = np.random.uniform(0., pixel_cpfs[..., -1], size=index_init.shape)
        # Use argmax trick to get first k where p[i,j] < pixel_cpfs[i,j,k]
        index_pred = (np.expand_dims(p, 2) < pixel_cpfs).argmax(axis=2)
        depth_pred = target_disc.get_value_from_sid_index(index_pred)
        return depth_pred


@ex.entity
class TransientPreprocessor:
    def __init__(self, n_sid_bins, n_ambient_bins, edge_coeff, n_std):
        self.n_sid_bins = n_sid_bins                # SID bins to discretize transient to
        self.n_ambient_bins = n_ambient_bins        # Number of bins to use to estimate ambient
        self.edge_coeff = edge_coeff                # Coefficient that controls edge detection
        self.n_std = n_std                          # Number of std devs above ambient to cut

    def __call__(self, raw_counts, counts_disc):
        """
        Apply preprocessing to raw transient data.
        """
        ambient = self.estimate_ambient(raw_counts)
        processed_counts = self.remove_ambient(raw_counts,
                                      ambient=ambient,
                                      grad_th=self.edge_coeff*np.sqrt(2*ambient))
        # set_trace()
        processed_counts = self.correct_falloff(processed_counts, counts_disc)

        # Scale SID object to maximize bin utilization
        min_depth_bin = np.min(np.nonzero(processed_counts))
        max_depth_bin = np.max(np.nonzero(processed_counts))
        min_depth = counts_disc.bin_edges[min_depth_bin]
        max_depth = counts_disc.bin_edges[max_depth_bin+1]
        sid_disc = SI(n_bins=self.n_sid_bins,
                      alpha=min_depth,
                      beta=max_depth,
                      offset=0.)
        processed_counts = rescale_hist(processed_counts, counts_disc, sid_disc)
        return processed_counts, sid_disc

    def estimate_ambient(self, counts):
        return np.mean(counts[:self.n_ambient_bins])

    def remove_ambient(self, counts, ambient, grad_th):
        """
        Create a "bounding box" that is bounded on the left and the right
        by using the gradients and below by using the ambient estimate.
        """
        # Detect edges:
        edges = np.abs(np.diff(counts)) > grad_th
        first = np.nonzero(edges)[0][1] + 1  # Right side of the first edge
        last = np.nonzero(edges)[0][-1]      # Left side of the second edge
        threshold = ambient + self.n_std * np.sqrt(ambient)
        # Walk first and last backward and forward to get below the threshold
        while first >= 0 and counts[first] > threshold:
            first -= 1
        while last < len(counts) and counts[last] > threshold:
            last += 1
        counts[:first] = 0.
        counts[last+1:] = 0.
        return np.clip(counts - ambient, a_min=0., a_max=None)

    def correct_falloff(self, counts, counts_disc):
        return counts * counts_disc.bin_values ** 2


def rgb2gray(img):
    """Requires (N)HWC tensor"""
    return 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]

@ex.entity
def refl_est_gray(nchw_tensor):
    """
    tensor should be a NCHW torch tensor in RGB channel order and in range [0, 1]
    returns a nhw reflectance map torch tensor
    """
    assert len(nchw_tensor.shape) == 4
    nhwc = nchw_tensor.permute(0, 2, 3, 1)
    refl_est = rgb2gray(nhwc)
    return refl_est

@ex.entity
def refl_est_red(nchw_tensor):
    """
    tensor should be a NCHW torch tensor in RGB channel order and in range [0, 1]
    returns a nhw reflectance map torch tensor
    """
    assert len(nchw_tensor.shape) == 4
    nhwc = nchw_tensor.permute(0, 2, 3, 1)
    refl_est = nhwc[..., 0]
    return refl_est
