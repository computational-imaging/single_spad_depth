#!/usr/bin/env python3
import numpy as np
from scipy.signal import fftconvolve
import configargparse

parser = configargparse.ArgParser(default_config_files=['spad.cfg'])
parser.add('split', type=str)
parser.add('-c', is_config_file=True)
parser.add('--n-bins', type=int, required=True)
parser.add('--min-depth', type=float, required=True)
parser.add('--max-depth', type=float, required=True)
parser.add('--laser-fwhm-ps', type=float, required=True)
parser.add('--signal-count', type=float, required=True)
parser.add('--sbr', type=float, nargs='+', required=True)
parser.add('--output-dir', required=True)
parser.add('--seed', type=int)


class SingleSPADWithLaser:
    def __init__(self, n_bins, min_depth, max_depth, laser_fwhm_ps, seed=0):
        """
        :param n_bins: Number of SPAD bins to capture
        :param min_depth, max_depth: Range of depths to discretize
        :param laser_fwhm_ps: Full-width-half-maximum of the laser pulse.
        :param seed: Random seed
        """
        assert max_depth >= min_depth

        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.laser_fwhm_ps = laser_fwhm_ps
        self.seed = seed
        np.random.seed(self.seed)


        # Derived quantitites
        self.bin_width_m = float(max_depth - min_depth) / n_bins
        # ps/bin, speed of light = 3e8, x2 because light needs to travel there and back.
        self.bin_width_ps = 2 * self.bin_width_m * 1e12/ 3e8

    def simulate(self, depth, intensity, mask, signal_count, sbr):
        # Per-pixel intensity and falloff
        weights = mask * intensity
        weights = weights / (depth ** 2 + 1e-6)
        depth_hist, _ = np.histogram(depth, bins=self.n_bins,
                                     range=(self.min_depth, self.max_depth),
                                     weights=weights)

        # Scale by number of photons
        counts = depth_hist * (signal_count / np.sum(depth_hist))

        # Add ambient/dark counts (dc_count) w/ Gaussian noise
        dc_count = signal_count / sbr
        counts += np.ones(len(counts)) * (dc_count / self.n_bins)

        # Convolve with PSF
        counts = self.apply_jitter(counts)

        # Apply Poisson noise
        counts = np.random.poisson(counts)
        return counts

    @staticmethod
    def make_gaussian_psf(size, fwhm):
        """ Make a gaussian kernel.
        size is the length of the array
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """

        x = np.arange(0, size, 1, float)
        x0 = size // 2
        return np.roll(np.exp(-4 * np.log(2) * ((x - x0) ** 2) / fwhm ** 2), len(x) - x0)

    def apply_jitter(self, counts):
        fwhm_bin = self.laser_fwhm_ps / self.bin_width_ps
        psf = self.make_gaussian_psf(len(counts), fwhm=fwhm_bin)
        counts = fftconvolve(psf, counts)[:int(self.n_bins)]
        counts = np.clip(counts, a_min=0., a_max=None)
        return counts


def rgb2gray(img):
    return 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]


if __name__ == "__main__":
    from tqdm import tqdm
    from pathlib import Path
    from pdb import set_trace
    args = parser.parse_args()
    spad = SingleSPADWithLaser(args.n_bins, args.min_depth,
                               args.max_depth, args.laser_fwhm_ps)

    # Load a sample depth map
    from dataloader import get_dataloader
    dataloader = get_dataloader(args.split)
    out = {}
    for sbr in args.sbr:
        print(f"Simulating with SBR = {sbr}")
        all_counts = np.zeros((len(dataloader.dataset), args.n_bins))
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader.dataset)):
            intensity = rgb2gray(data['image_cropped'].numpy()/255.)
            depth_truth = data['depth_cropped'].numpy()
            counts = spad.simulate(depth=depth_truth,
                                intensity=intensity,
                                mask=np.ones_like(depth_truth),
                                signal_count=args.signal_count,
                                sbr=sbr)
            all_counts[i, :] = counts
        out[str(sbr)] = all_counts
    output_path = Path(args.output_dir)/f"counts_{args.split}.npz"
    np.savez(output_path, **out)
