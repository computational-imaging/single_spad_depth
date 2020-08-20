#!/usr/bin/env python3

import configargparse
from ..experiment import Experiment

ex = Experiment('mde_transient')

@ex.config('mde_transient')
def cfg():
    parser = configargparse.ArgParser(default_config_files=['dorn_transient.cfg'])
    parser.add('--mde', choices=['DORN', 'DenseDepth', 'MiDaS'], required=True)
    parser.add('--refl_est', choices=['grey', 'r', 'g', 'b'], required=True)
    parser.add()


@ex.entity('mde_transient')
class MDETransient:
    def __init__(self, mde, preproc, refl_est, source_disc):
        self.mde = mde                   # MDE function
        self.refl_est = refl_est         # Reflectance estimation function
        self.preproc = preproc           # Preprocessing function
        self.source_disc = source_disc   # Discretization for reflectance-weighted depth hist

    def __call__(self, data):
        return self.predict(data['image'], data['transient'])

    def predict(self, image, transient):
        depth_init = self.mde(image)
        reflectance_est = self.refl_est(image)

        # compute weighted depth hist
        source_hist = self.weighted_histogram(depth_init, reflectance_est)

        target_hist, target_disc = self.preproc(transient)
        depth_final = \
            self.hist_match(depth_init,
                            source_hist, source_disc,
                            target_hist, target_disc)
        return depth_final

    def weighted_histogram(self, depth, weights):
        depth_hist, _ = np.histogram(depth, weights=weights,
                                    bins=self.source_disc.bin_edges)
        return depth_hist

    def hist_match(self, depth_init,
                   source_hist, source_disc,
                   target_hist, target_disc):
        movement = self.find_movement(source_hist, target_hist, normalize_to)
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
                pixels_req = hist_to[col] - np.sum(movement[:row, col])
                movement[row, col] = np.clip(np.minimum(pixels_rem, pixels_req), a_min=0., a_max=None)
        return movement

    def move_pixels(self, depth_init, movement, source_disc, target_disc):
        index_init = source_disc.get_sid_index_from_value(depth_init)
        cpfs = np.cumsum(T, axis=1)  # Cumulative sum across target dimension
        pixel_cpfs = cpfs[init_index, :] # [H, W, cpf dim]
        p = np.random.uniform(0., pixel_cpfs[..., -1], size=init_index.shape)
        # Use argmax trick to get first k where p[i,j] < pixel_cpfs[i,j,k]
        index_pred = (np.expand_dims(p, 2) < pixel_cpfs).argmax(axis=2)
        depth_pred = target_disc.get_value_from_sid_index(index_pred)
        return depth_pred
