#!/usr/bin/env python3

import numpy as np
from pdb import set_trace

class Discretization:
    def __init__(self):
        assert hasattr(self, 'bin_edges')
        assert hasattr(self, 'bin_values')


class Uniform(Discretization):
    """
    Uniform discretization
    """
    def __init__(self, n_bins, min_val, max_val):
        self.n_bins = n_bins
        self.min_val = min_val  # Lower end of smallest bin
        self.max_val = max_val  # Upper end of largest bin

        self.bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        self.bin_values = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2
        super().__init__()

    def __repr__(self):
        return self.__class__.__name__ + \
               repr((self.n_bins, self.min_val, self.max_val))


class SI(Discretization):
    """
    Implements Spacing-Increasing Discretization as described in the DORN paper.

    Discretizes the region [alpha, beta]
    Offset controls spacing even further by discretizing [alpha + offset, beta +
    offset] and then subtracting offset from all bin edges.

    Works in numpy.
    """
    def __init__(self, n_bins, alpha, beta, offset=0.):
        self.n_bins = n_bins
        self.alpha = alpha
        self.beta = beta
        self.offset = offset

        # Derived quantities
        self.alpha_star = self.alpha + offset
        self.beta_star = self.beta + offset
        bin_edges = np.array(range(n_bins + 1)).astype(np.float32)
        self.bin_edges = \
            np.array(np.exp( \
                np.log(self.alpha_star) + \
                bin_edges / n_bins *
                             np.log(self.beta_star / self.alpha_star)))
        self.bin_values = \
            (self.bin_edges[:-1] + self.bin_edges[1:]) / 2 - self.offset
        super().__init__()

    def get_sid_index_from_value(self, arr):
        """
        Given an array of values in the range [alpha, beta], return the
        indices of the bins they correspond to
        :param arr: The array to turn into indices.
        :return: The array of indices.
        """
        sid_index = np.floor(
            self.n_bins *
            (np.log(arr + self.offset) - np.log(self.alpha_star)) /
            (np.log(self.beta_star) - np.log(self.alpha_star))).astype(np.int32)
        sid_index = np.clip(sid_index, a_min=0, a_max=self.n_bins-1)
        return sid_index

    def get_value_from_sid_index(self, sid_index):
        """
        Given an array of indices in the range [0,...,sid_bins-1]
        return the representative value of the selected bin.
        :param sid_index: The array of indices.
        :return: The array of values correspondding to those indices
        """
        return np.take(self.bin_values, sid_index)

    def __repr__(self):
        return self.__class__.__name__ + \
               repr((self.n_bins, self.alpha, self.beta, self.offset))


def rescale_hist(from_hist, from_disc, to_disc):
    """
    Works in Numpy
    :param spad_counts: The histogram of spad counts to rescale.
    :param min_depth: The minimum depth of the histogram.
    :param max_depth: The maximum depth of the histogram.
    :param sid_obj: An object representing a SID.
    :return: A rescaled histogram in time to be according to the SID

    Assign photons to sid bins proportionally according to the amount of overlap
    between the sid bin range and the spad_count bin.
    """

    to_edges = to_disc.bin_edges
    from_edges = from_disc.bin_edges
    to_bin = 0
    to_hist = np.zeros(to_disc.n_bins)
    for from_bin in range(from_disc.n_bins):
        from_left = from_edges[from_bin]
        from_right = from_edges[from_bin+1]
        for to_bin, to_left, to_right in \
            gen_overlapping_bins(to_bin, to_edges, from_left, from_right):
            left = max(to_left, from_left)
            right = min(to_right, from_right)
            frac = (right - left) / (from_right - from_left)
            to_hist[to_bin] += frac * from_hist[from_bin]
    return to_hist

def gen_overlapping_bins(to_bin, to_edges, from_left, from_right):
    to_left = to_edges[to_bin]
    to_right = to_edges[to_bin+1]
    # Find a to_bin that overlaps with [from_left, from_right]
    while to_right < from_left:
        to_bin += 1
        if to_bin >= len(to_edges) - 1:
            return  # No more bins left
        to_left = to_edges[to_bin]
        to_right = to_edges[to_bin+1]
    # to_right >= from_left
    while from_right > to_left:
        # intervals overlap
        yield to_bin, to_left, to_right
        to_bin += 1
        if to_bin >= len(to_edges) - 1:
            break
        to_left = to_edges[to_bin]
        to_right = to_edges[to_bin+1]
