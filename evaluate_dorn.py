#!/usr/bin/env python3

import os
import numpy as np
import torch
from pathlib import Path
from pdb import set_trace

def image_histogram_match_variable_bin(init, gt_hist, weights, sid_obj_init, sid_obj_pred, vectorized=True):
    weights = weights * (np.sum(gt_hist) / np.sum(weights))
    init_index = np.clip(sid_obj_init.get_sid_index_from_value(init),
                         a_min=0, a_max=sid_obj_init.sid_bins - 1)
    init_hist, _ = np.histogram(init_index, weights=weights, bins=range(sid_obj_init.sid_bins + 1))
    if (gt_hist < 0).any():
        print("Negative values in gt_hist")
        raise Exception()
    T_count = find_movement(init_hist, gt_hist)
    if vectorized:
        pred_index = move_pixels_vectorized(T_count, init_index, weights)
    # pred_index = move_pixels_vectorized_weighted(T_count, init_index, weights)
    else:
        pred_index = move_pixels(T_count, init_index, weights)
    pred = sid_obj_pred.get_value_from_sid_index(pred_index)
    pred_hist, _ = np.histogram(pred_index, weights=weights, bins=range(len(gt_hist) + 1))
    return pred, (init_index, init_hist, pred_index, pred_hist, T_count)


def image_histogram_match_lin(init, gt_hist, weights, min_depth, max_depth):
    weights = weights * (np.sum(gt_hist) / np.sum(weights))
    n_bins = len(gt_hist)
    bin_edges = np.linspace(min_depth, max_depth, n_bins + 1)
    bin_values = (bin_edges[1:] + bin_edges[:-1])/2
    init_index = np.clip(np.floor((init - min_depth)*n_bins/(max_depth - min_depth)).astype('int'),
                         a_min=0., a_max=n_bins-1)
    init_hist, _ = np.histogram(init_index, weights=weights, bins=range(n_bins+1))
    if (gt_hist < 0).any():
        print("Negative values in gt_hist")
        raise Exception()
    T_count = find_movement(init_hist, gt_hist)
#     pred_index = move_pixels_raster(T_count, init_index, weights)
#     pred_index = move_pixels(T_count, init_index, weights)
#     pred_index = move_pixels_better(T_count, init_index, weights)
#     pred_index = move_pixels_vectorized(T_count, init_index, weights)
    pred_index = move_pixels_weighted(T_count, init_index, weights)
    pred = np.take(bin_values, pred_index)
    pred_hist, _ = np.histogram(pred_index, weights=weights, bins=range(n_bins+1))
    return pred, (init_index, init_hist, pred_index, pred_hist, T_count)


class SID:
    """
    Implements Spacing-Increasing Discretization as described in the DORN paper.

    Discretizes the region [alpha, beta]
    Offset controls spacing even further by discretizing [alpha + offset, beta + offset] and then
    subtracting offset from all bin edges.

    Bonus: Includes support for when the index is -1 (in which case the value should be alpha)
    and when it is sid_bins (in which case the value should be beta).

    Works in numpy.
    """
    def __init__(self, sid_bins, alpha, beta, offset):
        self.sid_bins = sid_bins
        self.alpha = alpha
        self.beta = beta
        self.offset = offset

        # Derived quantities
        self.alpha_star = self.alpha + offset
        self.beta_star = self.beta + offset
        bin_edges = np.array(range(sid_bins + 1)).astype(np.float32)
        self.sid_bin_edges = np.array(np.exp(np.log(self.alpha_star) +
                                             bin_edges / self.sid_bins * np.log(self.beta_star / self.alpha_star)))
        self.sid_bin_values = (self.sid_bin_edges[:-1] + self.sid_bin_edges[1:]) / 2 - self.offset
        self.sid_bin_values = np.append(self.sid_bin_values, [self.alpha, self.beta])
        # Do the above so that:
        # self.sid_bin_values[-1] = self.alpha < self.sid_bin_values[0]
        # and
        # self.sid_bin_values[sid_bins] = self.beta > self.sid_bin_values[sid_bins-1]

    def get_sid_index_from_value(self, arr):
        """
        Given an array of values in the range [alpha, beta], return the
        indices of the bins they correspond to
        :param arr: The array to turn into indices.
        :return: The array of indices.
        """
        sid_index = np.floor(self.sid_bins * (np.log(arr + self.offset) - np.log(self.alpha_star)) /
                                             (np.log(self.beta_star) - np.log(self.alpha_star))).astype(np.int32)
        sid_index = np.clip(sid_index, a_min=-1, a_max=self.sid_bins)
        # An index of -1 indicates alpha, while self.sid_bins indicates beta
        return sid_index

    def get_value_from_sid_index(self, sid_index):
        """
        Given an array of indices in the range {-1, 0,...,sid_bins},
        return the representative value of the selected bin.
        :param sid_index: The array of indices.
        :return: The array of values correspondding to those indices
        """
        return np.take(self.sid_bin_values, sid_index)

    def __repr__(self):
        return repr((self.sid_bins, self.alpha, self.beta, self.offset))


def rescale_bins(spad_counts, min_depth, max_depth, sid_obj):
    """
    Works in Numpy
    :param spad_counts: The histogram of spad counts to rescale.
    :param min_depth: The minimum depth of the histogram.
    :param max_depth: The maximum depth of the histogram.
    :param sid_obj: An object representing a SID.
    :return: A rescaled histogram in time to be according to the SID

    Assign photons to sid bins proportionally according to the amount of overlap between
    the sid bin range and the spad_count bin.
    """

    sid_bin_edges_m = sid_obj.sid_bin_edges - sid_obj.offset
#     print(sid_bin_edges_m)
    # Convert sid_bin_edges_m into units of spad bins
    sid_bin_edges_bin = (sid_bin_edges_m - min_depth) * len(spad_counts) / (max_depth - min_depth)
    # Fail if the SID bins lie outside the range
    # if np.min(sid_bin_edges_bin) < 0 or np.max(sid_bin_edges_bin) > len(spad_counts):
    #     print("spad range: [{}, {}]".format(0, len(spad_counts)))
    #     print("SID range: [{}, {}]".format(np.min(sid_bin_edges_bin),
    #                                        np.max(sid_bin_edges_bin)))
    #     raise ValueError("SID range exceeds spad range.")


    # sid_bin_edges_bin -= sid_bin_edges_bin[0]  # Start at 0
    # sid_bin_edges_bin[-1] = np.floor(sid_bin_edges_bin[-1])
    # print(sid_bin_edges_bin[-1])
    # Map spad_counts onto sid_bin indices
    # print(sid_bin_edges_bin)
    sid_counts = np.zeros(sid_obj.sid_bins)
    for i, (left, right) in enumerate(zip(sid_bin_edges_bin[:-1], sid_bin_edges_bin[1:])):
        # print("left: ", left)
        # print("right: ", right)
        curr = left
        while curr < right and int(np.floor(left)) < len(spad_counts)       :
            curr = np.min([right, np.floor(left + 1.)])  # Don't go across spad bins - stop at integers
            sid_counts[i] += (curr - left) * spad_counts[int(np.floor(left))]
            # Update window
            left = curr
        # print(sid_counts[i])
    return sid_counts

def remove_dc_from_spad_edge(spad, ambient, grad_th=1e3, n_std=1.):
    """
    Create a "bounding box" that is bounded on the left and the right
    by using the gradients and below by using the ambient estimate.
    """
    # Detect edges:
    assert len(spad.shape) == 1
    edges = np.abs(np.diff(spad)) > grad_th
    print(np.abs(np.diff(spad)))
    first = np.nonzero(edges)[0][1] + 1  # Want the right side of the first edge
    last = np.nonzero(edges)[0][-1]      # Want the left side of the second edge
    below = ambient + n_std*np.sqrt(ambient)
    # Walk first and last backward and forward until we encounter a value below the threshold
    while first >= 0 and spad[first] > below:
        first -= 1
    while last < len(spad) and spad[last] > below:
        last += 1
    spad[:first] = 0.
    spad[last+1:] = 0.
    print(first)
    print(last)
    return np.clip(spad - ambient, a_min=0., a_max=None)

# from nyuv2_labeled_dataset import nyuv2_labeled_ingredient, load_data

# from models.loss import get_depth_metrics


data_dir = Path(__file__).parent/'data'/'nyu_depth_v2'/'processed'
dataset_type = 'test'
use_intensity = True
use_squared_falloff = True
dc_count = 1e4
use_jitter = True
use_poisson = True
hyper_string = "{}_int_{}_fall_{}_lamb_False_dc_{}_jit_{}_poiss_{}".format(
    dataset_type,
    use_intensity,
    use_squared_falloff,
    dc_count,
    use_jitter,
    use_poisson)
spad_file = data_dir/'{}_spad.npy'.format(hyper_string)
dorn_depth_file = data_dir/'dorn_{}_outputs.npy'.format(dataset_type)

# SID params
sid_bins = 68
bin_edges = np.array(range(sid_bins + 1)).astype(np.float32)
dorn_decode = np.exp((bin_edges - 1) / 25 - 0.36)
d0 = dorn_decode[0]
d1 = dorn_decode[1]
# Algebra stuff to make the depth bins work out exactly like in the
# original DORN code.
alpha = (2 * d0 ** 2) / (d1 + d0)
beta = alpha * np.exp(sid_bins * np.log(2 * d0 / alpha - 1))
del bin_edges, dorn_decode, d0, d1
offset = 0.

# SPAD Denoising params
lam = 3e2
eps_rel = 1e-5

entry = None
save_outputs = True
small_run = 0
output_dir = "results"

if __name__ == '__main__':
    # Load all the data:
    print("Loading SPAD data from {}".format(spad_file))
    spad_dict = np.load(spad_file, allow_pickle=True).item()
    spad_data = spad_dict["spad"]
    intensity_data = spad_dict["intensity"]
    spad_config = spad_dict["config"]
    print("Loading depth data from {}".format(dorn_depth_file))
    depth_data = np.load(dorn_depth_file)
    # dataset = load_data(channels_first=True, dataset_type=dataset_type)

    # Read SPAD config and determine proper course of action
    # dc_count = spad_config["dc_count"]
    true_ambient = spad_config["dc_count"]/spad_config["spad_bins"]
    # use_intensity = spad_config["use_intensity"]
    # use_squared_falloff = spad_config["use_squared_falloff"]
    # lambertian = spad_config["lambertian"]
    # min_depth = spad_config["min_depth"]
    # max_depth = spad_config["max_depth"]

    print("dc_count: ", dc_count)
    print("use_intensity: ", use_intensity)
    print("use_squared_falloff:", use_squared_falloff)

    print("spad_data.shape", spad_data.shape)
    print("depth_data.shape", depth_data.shape)
    print("intensity_data.shape", intensity_data.shape)

    sid_obj_init = SID(sid_bins, alpha, beta, offset)
    if entry is None:
        metric_list = ["delta1", "delta2", "delta3", "rel_abs_diff", "rmse", "mse", "log10", "weight"]
        metrics = np.zeros((654, len(metric_list)))
        entry_list = []
        outputs = []
        for i in range(depth_data.shape[0]):
            if small_run and i == small_run:
                break
            entry_list.append(i)

            print("Evaluating {}[{}]".format(dataset_type, i))
            spad = spad_data[i,...]
            ambient = np.mean(spad[:60])
            ambient_all = np.mean(spad_data[:, :60])
            weights = np.ones_like(depth_data[i, 0, ...])
            if use_intensity:
                weights = intensity_data[i, 0, ...]
            if dc_count > 0.:
                spad = remove_dc_from_spad_edge(spad,
                                                ambient=ambient,
                                                grad_th=5*np.sqrt(2*ambient))
            set_trace()
            bin_edges = np.linspace(min_depth, max_depth, len(spad) + 1)
            bin_values = (bin_edges[1:] + bin_edges[:-1]) / 2
            if use_squared_falloff:
                if lambertian:
                    spad = spad * bin_values ** 4
                else:
                    spad = spad * bin_values ** 2
            # Scale SID object to maximize bin utilization
            nonzeros = np.nonzero(spad)[0]
            if nonzeros.size > 0:
                min_depth_bin = np.min(nonzeros)
                max_depth_bin = np.max(nonzeros) + 1
                if max_depth_bin > len(bin_edges) - 2:
                    max_depth_bin = len(bin_edges) - 2
            else:
                min_depth_bin = 0
                max_depth_bin = len(bin_edges) - 2
            min_depth_pred = np.clip(bin_edges[min_depth_bin], a_min=1e-2, a_max=None)
            max_depth_pred = np.clip(bin_edges[max_depth_bin+1], a_min=1e-2, a_max=None)
            sid_obj_pred = SID(sid_bins=sid_obj_init.sid_bins,
                               alpha=min_depth_pred,
                               beta=max_depth_pred,
                               offset=0.)
            spad_rescaled = rescale_bins(spad[min_depth_bin:max_depth_bin+1],
                                         min_depth_pred, max_depth_pred, sid_obj_pred)
            pred, t = image_histogram_match_variable_bin(depth_data[i, 0, ...], spad_rescaled, weights,
                                                         sid_obj_init, sid_obj_pred)
            # break
            # break
            # Calculate metrics
            gt = dataset[i]["depth_cropped"].unsqueeze(0)
            # print(gt.dtype)
            # print(pred.shape)

            pred_metrics = get_depth_metrics(torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float(),
                                             gt,
                                             torch.ones_like(gt))

            for j, metric_name in enumerate(metric_list[:-1]):
                metrics[i, j] = pred_metrics[metric_name]

            metrics[i, -1] = np.size(pred)
            # Option to save outputs:
            if save_outputs:
                outputs.append(pred)
            print("\tAvg RMSE = {}".format(np.mean(metrics[:i + 1, metric_list.index("rmse")])))

        if save_outputs:
            np.save(os.path.join(output_dir, "dorn_{}_outputs.npy".format(hyper_string)), np.array(outputs))

        # Save metrics using pandas
        metrics_df = pd.DataFrame(data=metrics, index=entry_list, columns=metric_list)
        metrics_df.to_pickle(path=os.path.join(output_dir, "dorn_{}_metrics.pkl".format(hyper_string)))
        # Compute weighted averages:
        average_metrics = np.average(metrics_df.ix[:, :-1], weights=metrics_df.weight, axis=0)
        average_df = pd.Series(data=average_metrics, index=metric_list[:-1])
        average_df.to_csv(os.path.join(output_dir, "dorn_{}_avg_metrics.csv".format(hyper_string)), header=True)
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('d1', 'd2', 'd3', 'rel', 'rms', 'log_10'))
        print(
            "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(average_metrics[0],
                                                                                average_metrics[1],
                                                                                average_metrics[2],
                                                                                average_metrics[3],
                                                                                average_metrics[4],
                                                                                average_metrics[6]))


        print("wrote results to {} ({})".format(output_dir, hyper_string))

    else:
        input_unbatched = dataset.get_item_by_id(entry)
        # for key in ["rgb", "albedo", "rawdepth", "spad", "mask", "rawdepth_orig", "mask_orig", "albedo_orig"]:
        #     input_[key] = input_[key].unsqueeze(0)
        from torch.utils.data._utils.collate import default_collate

        data = default_collate([input_unbatched])

        # Checks
        entry = data["entry"][0]
        i = int(entry)
        entry = entry if isinstance(entry, str) else entry.item()
        print("Evaluating {}[{}]".format(dataset_type, i))
        # Rescale SPAD
        spad_rescaled = rescale_bins(spad_data[i, ...], min_depth, max_depth, sid_obj)
        weights = np.ones_like(depth_data[i, 0, ...])
        if use_intensity:
            weights = intensity_data[i, 0, ...]
        spad_rescaled = preprocess_spad(spad_rescaled, sid_obj, use_squared_falloff, dc_count > 0.,
                                        lam=lam, eps_rel=eps_rel)

        pred, _ = image_histogram_match(depth_data[i, 0, ...], spad_rescaled, weights, sid_obj)
        # break
        # Calculate metrics
        gt = data["depth_cropped"]
        print(gt.shape)
        print(pred.shape)

        pred_metrics = get_depth_metrics(torch.from_numpy(pred).unsqueeze(0).unsqueeze(0),
                                         gt,
                                         torch.ones_like(gt))
        if save_outputs:
            np.save(os.path.join(output_dir, "dorn_{}[{}]_{}_out.npy".format(dataset_type, entry, hyper_string)))
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('d1', 'd2', 'd3', 'rel', 'rms', 'log_10'))
        print(
            "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(pred_metrics["delta1"],
                                                                                pred_metrics["delta2"],
                                                                                pred_metrics["delta3"],
                                                                                pred_metrics["rel_abs_diff"],
                                                                                pred_metrics["rms"],
                                                                                pred_metrics["log10"]))
