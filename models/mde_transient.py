#!/usr/bin/env python3

import configargparse
from ..discretize import SI, rescale_hist
from ..experiment import ex

@ex.config('transient')
def cfg():
    parser = configargparse.ArgParser(default_config_files=['dorn_transient.cfg'])
    parser.add('--mde', choices=['DORN', 'DenseDepth', 'MiDaS'], required=True)
    parser.add('--refl-est', choices=['grey', 'r'], required=True)
    parser.add('--n-sid-bins', type=int, default=68)
    parser.add('--ambient-bins', type=int, default=100)
    parser.add('--beta', type=float, default=5.)
    parser.add('--n-std', type=int, default=1)
    args, _ = parser.parse_known_args()
    return vars(args)


@ex.entity('MDETransient')
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


@ex.entity('TransientPreprocessor')
class TransientPreprocessor:
    def __init__(self, n_sid_bins, n_ambient_bins, beta, n_std):
        self.n_sid_bins = n_sid_bins                # SID bins to discretize transient to
        self.n_ambient_bins = n_ambient_bins        # Number of bins to use to estimate ambient
        self.beta = beta                            # Gradient threshold for edge detection
        self.n_std = n_std                          # Number of std devs above ambient to cut

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
        return np.mean(counts[:self.n_ambient_bins])

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
