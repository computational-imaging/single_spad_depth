#!/usr/bin/env python3

import numpy as np
from discretize import rescale_hist, SI

class TransientPreprocessor:
    def __init__(self, n_sid_bins, N, beta, n_std):
        self.n_sid_bins = n_sid_bins # SID bins to discretize transient to
        self.N = N                   # Number of bins to use to estimate ambient
        self.beta = beta             # Gradient threshold for edge detection
        self.n_std = n_std           # Number of std devs above ambient to cut


    def preprocess_transient(self, raw_counts, counts_disc):
        """
        Apply preprocessing to raw transient data.
        """
        ambient = self.estimate_ambient(raw_counts)
        processed_counts = self.remove_ambient(raw_counts,
                                      ambient=ambient,
                                      grad_th=self.beta*np.sqrt(2*ambient))
        processed_counts = self.correct_falloff(processed_counts, counts_disc)

        # Scale SID object to maximize bin utilization
        min_depth_bin = np.min(np.nonzero(processed_counts))
        max_depth_bin = np.max(np.nonzero(processed_counts))
        min_depth = counts_disc.bin_edges[min_depth_bin]
        max_depth = counts_disc.bin_edges[max_depth_bin+1]
        sid_disc = SI(n_bins=self.n_sid_bins,
                      alpha=min_depth_pred,
                      beta=max_depth_pred,
                      offset=0.)
        processed_counts = rescale_hist(processed_counts, counts_disc, sid_disc)
        return processed_counts

    def estimate_ambient(self, counts):
        return np.mean(counts[:self.N])

    def remove_ambient(self, counts, ambient, grad_th):
        """
        Create a "bounding box" that is bounded on the left and the right
        by using the gradients and below by using the ambient estimate.
        """
        # Detect edges:
        assert len(counts.shape) == 1
        edges = np.abs(np.diff(counts)) > grad_th
        first = np.nonzero(edges)[0][1] + 1  # Right side of the first edge
        last = np.nonzero(edges)[0][-1]      # Left side of the second edge
        threshold = ambient + self.n_std * np.sqrt(ambient)
        # Walk first and last backward and forward to get below the threshold
        while first >= 0 and spad[first] > threshold:
            first -= 1
        while last < len(spad) and spad[last] > threshold:
            last += 1
        counts[:first] = 0.
        counts[last+1:] = 0.
        return np.clip(counts - ambient, a_min=0., a_max=None)

    def correct_falloff(self, counts, counts_disc):
        return counts * counts_disc.bin_values ** 2


