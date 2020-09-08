#!/usr/bin/env python3
import h5py
import numpy as np
import scipy.io as sio
from pathlib import Path

def load_and_crop_kinect(rootdir, calibration_file="calibration.mat", kinect_file="kinect.mat"):
    # Calibration data
    calib = loadmat_h5py(Path(rootdir)/calibration_file)
    kinect = loadmat_h5py(Path(rootdir)/kinect_file)
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

def extract_camera_params_from_file(filepath):
    arr = sio.loadmat(filepath)
    param_struct = arr["param_struct"]
    RotationOfCamera2 = param_struct["RotationOfCamera2"][0][0]
    TranslationOfCamera2 = param_struct["TranslationOfCamera2"][0][0][0]
    cp1 = param_struct["CameraParameters1"][0][0][0][0]
    cp2 = param_struct["CameraParameters2"][0][0][0][0]

    camera_params1 = extract_camera_intrinsics(cp1)
    camera_params2 = extract_camera_intrinsics(cp2)

    return camera_params1, camera_params2, RotationOfCamera2, TranslationOfCamera2

def extract_camera_intrinsics(CameraParameters):
    """
    Get intrinsics from param struct
    """
    out = {}
    for k in ["FocalLength", "PrincipalPoint", "RadialDistortion", "TangentialDistortion"]:
        out[k] = CameraParameters[k][0]
    return out

def loadmat_h5py(file):
    output = {}
    with h5py.File(file, 'r') as f:
        for k, v in f.items():
            output[k] = np.array(v)
    return output

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
