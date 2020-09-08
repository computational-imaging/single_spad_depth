#!/usr/bin/env python3

import scipy.io as sio
import numpy as np
import cv2
import h5py
from pathlib import Path

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

def undistort_img(img, fc, pc, rdc, tdc):
    """Apply standard distortion correction to an image given the camera distortion
    parameters. Wrapper for cv2.undistort.

    img: The image to undistort.
    fc: focal length parameter
    pc: Principal point
    rdc: Radial distortion coefficients
    tdc: Tangential distortion coefficients
    """
    if len(rdc) == 2:
        rdc = np.append(rdc, 0.)
    distortionCoefficients = np.concatenate((rdc[:2], tdc, rdc[2:]))
    cameraMatrix = np.array([[fc[0], 0.,     pc[0]],
                             [0.,     fc[1], pc[1]],
                             [0.,     0.,    1.]])
    img_undist = cv2.undistort(img, cameraMatrix, distortionCoefficients)
    return img_undist

def get_hist_med(histogram):
    """Returns the mean bin value"""
    pdf = histogram/np.sum(histogram)
    cdf = np.cumsum(pdf)
    med = (cdf > 0.5).argmax()
    return med


def savefig_no_whitespace(filepath):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0,)
    param_struct = arr["param_struct"]
    RotationOfCamera2 = param_struct["RotationOfCamera2"][0][0]
    TranslationOfCamera2 = param_struct["TranslationOfCamera2"][0][0][0]
    cp1 = param_struct["CameraParameters1"][0][0][0][0]
    cp2 = param_struct["CameraParameters2"][0][0][0][0]

    camera_params1 = extract_camera_intrinsics(cp1)
    camera_params2 = extract_camera_intrinsics(cp2)

    return camera_params1, camera_params2, RotationOfCamera2, TranslationOfCamera2


def project_depth(z, z_mask, imagesize2, fc1, fc2, pc1, pc2, RotationOfCamera2, TranslationOfCamera2):
    """
    Project a depth map z from camera 1 to camera 2, given the rotation matrix and translation vectors.

    Does not account for any distortion.

    Pay attention that the units of z and the units of the translation vector match.

    :param z: Input depth map
    :param imagesize2: Size of output image
    :param fc1: Focal Length in pixels ([x, y] order) of camera 1.
    :param fc2: Focal Length in pixels ([x, y] order) of camera 2.
    :param pc1: pair (y, x) of the center pixel in input depth map
    :param pc2: pair (y, x) of the center pixel in the output depth map
    :param RotationOfCamera2: Rotation of camera 2 relative to camera 1.
    :param TranslationOfCamera2: Translation of camera 2 relative to camera 1.
        # NOTE: To transform, use x^T*R + t^T = x_new^T
    """
    new_image = np.zeros(imagesize2)
    new_mask = np.zeros(imagesize2)
    yy, xx = np.meshgrid(range(z.shape[0]), range(z.shape[1]), indexing="ij")

    # Get coordinates in world units
    x = ((xx - pc1[0]) * z) / fc1[0]
    y = ((yy - pc1[1]) * z) / fc1[1]

    # Use rotation and translation to get (x', y', z')
    xyz = np.array([x, y, z]).transpose(1, 2, 0)
    xyz_new = np.matmul(xyz, RotationOfCamera2) + TranslationOfCamera2
    x_new = xyz_new[..., 0]
    y_new = xyz_new[..., 1]
    z_new = xyz_new[..., 2]

    # Get pixel coords from world coords
    xx_new = np.floor(x_new * fc2[0] / z_new + pc2[0]).astype('int')
    yy_new = np.floor(y_new * fc2[1] / z_new + pc2[1]).astype('int')
    mask = (xx_new >= 0) & (xx_new < imagesize2[1]) & (yy_new >= 0) & (yy_new < imagesize2[0])
    new_image[yy_new[mask], xx_new[mask]] = z_new[mask]
    new_mask[yy_new[mask], xx_new[mask]] = z_mask[mask]
    return new_image, new_mask
