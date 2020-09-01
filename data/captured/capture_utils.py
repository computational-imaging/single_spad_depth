#!/usr/bin/env python3

import os
import h5py
import cv2
import numpy as np
from remove_dc_from_spad import remove_dc_from_spad_edge
import matplotlib.pyplot as plt

from models.data.data_utils.sid_utils import SID


def loadmat_h5py(file):
    output = {}
    with h5py.File(file, 'r') as f:
        for k, v in f.items():
            output[k] = np.array(v)
    return output

fc_kinect = [1053.622, 1047.508]
fc_spad = [758.2466, 791.2153]


def z_to_r(z, fc):
    yy, xx = np.meshgrid(range(z.shape[0]), range(z.shape[1]), indexing="ij")
    x = (xx * z) / fc[0]
    y = (yy * z) / fc[1]
    r = np.sqrt(x**2 + y**2 + z**2)
    return r


def r_to_z(r, fc):
    yy, xx = np.meshgrid(range(r.shape[0]), range(r.shape[1]), indexing="ij")
    z = r / np.sqrt((xx/fc[0])**2 + (yy/fc[1])**2 + 1)
    return z


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
    sid_bin_edges_bin = sid_bin_edges_m * len(spad_counts) / (max_depth - min_depth)
    sid_bin_edges_bin -= sid_bin_edges_bin[0]  # Start at 0
    sid_bin_edges_bin[-1] = np.floor(sid_bin_edges_bin[-1])
    # print(sid_bin_edges_bin[-1])
    # Map spad_counts onto sid_bin indices
    # print(sid_bin_edges_bin)
    sid_counts = np.zeros(sid_obj.sid_bins)
    for i in range(sid_obj.sid_bins):
#         print(i)
        left = sid_bin_edges_bin[i]
        right = sid_bin_edges_bin[i + 1]
        curr = left
        while curr != right:
#             print(curr)
            curr = np.min([right, np.floor(left + 1.)])  # Don't go across spad bins - stop at integers
            sid_counts[i] += (curr - left) * spad_counts[int(np.floor(left))]
            # Update window
            left = curr

    return sid_counts


def normals_from_depth(z):
    """
    Compute surface normals of a depth map.
    """
    # Get GT x and y coords
    fc = [758.2466, 791.2153]  # Focal length of SPAD in pixels
    yy, xx = np.meshgrid(range(z.shape[0]), range(z.shape[1]), indexing="ij")
    x = (xx * z) / fc[0]
    y = (yy * z) / fc[1]
    dzdx = (z[:, 2:] - z[:, :-2])/(x[:, 2:] - x[:, :-2])
    dzdy = (z[:-2, :] - z[2:, :])/(y[:-2, :] - y[2:, :])
    n = np.array((-dzdx[1:-1, :], -dzdy[:, 1:-1], np.ones_like(dzdx[1:-1,:])))
    n /= np.sqrt(np.sum(n**2, axis=0)) # Normalize each vector
    return n


def get_closer_to_mod(lower, upper, mod):
    """
    Adjusts lower and upper so that their difference is 0 (mod mod)
    :param lower: smaller number
    :param upper: larger number
    :param mod: the modulus
    :return: pair (lower_modified, upper_modified) such that upper_modified - lower_modified = 0 (mod mod)
    """
    assert lower <= upper
    diff = (upper - lower) % mod
    if diff > mod//2:
        diff = diff - mod # Negative
    lower_correction = diff//2
    upper_correction = diff - diff//2
    return lower + lower_correction, upper - upper_correction

def get_hist_med(histogram):
    """Returns the mean bin value"""
    pdf = histogram/np.sum(histogram)
    cdf = np.cumsum(pdf)
    med = (cdf > 0.5).argmax()
    return med


def savefig_no_whitespace(filepath):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filepath, bbox_inches='tight',
                pad_inches=0)


def save_depth_image(img, vmin, vmax, filepath):
    plt.figure()
    plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.axis('off')
    savefig_no_whitespace(filepath)


def depth_imwrite(img, filepath):
    """
    Save the image, scaling it to be in [0,255] first.
    :param img:
    :param filepath:
    :return:
    """
    print("depth_imwrite to {}".format(filepath))
    img_scaled = ((img - np.min(img)) * 65535./(np.max(img) - np.min(img))).astype(np.uint16)
    cv2.imwrite(filepath + ".png", img_scaled)


def load_spad(spad_file):
    print("Loading SPAD data...")
    spad_data = loadmat_h5py(spad_file)
    return spad_data["mat"]


def preprocess_spad(spad_single, ambient_estimate, min_depth, max_depth, sid_obj):
    print("Processing SPAD data...")
    # Remove DC
    spad_denoised = remove_dc_from_spad_edge(spad_single,
                                             ambient=ambient_estimate,
                                             grad_th=5*np.sqrt(2*ambient_estimate))

    # Correct Falloff
    bin_edges = np.linspace(min_depth, max_depth, len(spad_denoised) + 1)
    bin_values = (bin_edges[1:] + bin_edges[:-1]) / 2
    spad_corrected = spad_denoised * bin_values ** 2

    # Scale SID object to maximize bin utilization
    min_depth_bin = np.min(np.nonzero(spad_corrected))
    max_depth_bin = np.max(np.nonzero(spad_corrected))
    min_depth_pred = bin_values[min_depth_bin]
    max_depth_pred = bin_values[max_depth_bin]
    sid_obj_pred = SID(sid_bins=sid_obj.sid_bins,
                       alpha=min_depth_pred,
                       beta=max_depth_pred,
                       offset=0.)

    # Convert to SID
    spad_sid = rescale_bins(spad_corrected[min_depth_bin:max_depth_bin+1],
                            min_depth_pred, max_depth_pred, sid_obj_pred)
    return spad_sid, sid_obj_pred, spad_denoised, spad_corrected


def load_and_crop_kinect(rootdir, calibration_file="calibration.mat", kinect_file="kinect.mat"):
    # Calibration data
    print("Loading calibration data...")
    calib = loadmat_h5py(os.path.join(rootdir, calibration_file))

    print("Loading kinect data...")
    kinect = loadmat_h5py(os.path.join(rootdir, kinect_file))
    # Transpose
    kinect_rgb = np.fliplr(kinect["rgb_im"].transpose(2, 1, 0))

    # Extract crop
    top = (calib["pos_01"][0] + calib["pos_11"][0]) // 2
    bot = (calib["pos_10"][0] + calib["pos_00"][0]) // 2
    left = (calib["pos_11"][1] + calib["pos_10"][1]) // 2
    right = (calib["pos_01"][1] + calib["pos_00"][1]) // 2

    # Scene-specific crop
    crop = (int(top[0]), int(bot[0]), int(left[0]), int(right[0]))
    top_mod, bot_mod = get_closer_to_mod(crop[0], crop[1], 32)
    left_mod, right_mod = get_closer_to_mod(crop[2], crop[3], 32)
    crop = (top_mod, bot_mod, left_mod, right_mod)

    # Crop
    rgb_cropped = kinect_rgb[crop[0]:crop[1], crop[2]:crop[3], :]

    # Intensity
    intensity = rgb_cropped[:, :, 0] / 255.
    return kinect_rgb, rgb_cropped, intensity, crop
