#!/usr/bin/env python3

import numpy as np
from data.captured.camera_utils import (extract_camera_params_from_file,
                                        load_and_crop_kinect,
                                        project_depth,
                                        r_to_z,
                                        z_to_r,
                                        loadmat_h5py)



from pdb import set_trace
from pathlib import Path
from models.discretize import SI

# Scenes and metadata
# Offset is a pixel-wise offset to adjust the crop and compensate for calibration drift.
# bin_width_ps is the bin width of the SPAD in picosecnds for that particular scene.
# min_r and max_r are the min and max distances and are set to reject direct reflections off the beam splitter.

SCENE_DIR = Path(__file__).parent/'data'/'captured'/'raw'/'scanned'
CALIBRATION_FILE = SCENE_DIR/'calibration'/'camera_params.mat'

SCANNED_SCENES = {
    '8_29_kitchen_scene': {
        'offset': (-10, -8),
        'bin_width_ps': 16,
        'min_r': 0.4,
        'max_r': 9.
    },
    '8_29_conference_room_scene': {
        'offset': (-16, -12),
        'bin_width_ps': 16,
        'min_r': 0.4,
        'max_r': 9.
    },
    '8_30_conference_room2_scene': {
        'offset': (-16, -12),
        'bin_width_ps': 16,
        'min_r': 0.4,
        'max_r': 9.
    },
    '8_30_Hallway': {
        'offset': (0, 0),
        'bin_width_ps': 16,
        'min_r': 0.4,
        'max_r': 9.
    },
    '8_30_poster_scene': {
        'offset': (0, 0),
        'bin_width_ps': 16,
        'min_r': 0.4,
        'max_r': 9.
    },
    '8_30_small_lab_scene': {
        'offset': (0, 0),
        'bin_width_ps': 16,
        'min_r': 0.4,
        'max_r': 9.
    },
    '8_31_outdoor3': {
        'offset': (0, 0),
        'bin_width_ps': 32,
        'min_r': 0.4,
        'max_r': 11.
    },
}

if __name__ == '__main__':
    # Load the calibration file
    kinect_intrinsics, spad_intrinsics, RotationOfSpad, TranslationOfSpad = \
        extract_camera_params_from_file(str(CALIBRATION_FILE))
    RotationOfKinect = RotationOfSpad.T
    TranslationOfKinect = -TranslationOfSpad.dot(RotationOfSpad.T)
    AMBIENT_MAX_DEPTH_BIN = 100

    input_root = Path(__file__).parent/'data'/'captured'/'raw'/'scanned'
    output_dir = Path(__file__).parent/'data'/'captured'/'processed'

    for scene, meta in SCANNED_SCENES.items():
        # Check for the directory first:
        scene_dir = input_root/f'{scene}'
        if not scene_dir.exists():
            print(f"Skipping {scene}...")
            continue
        print("Preprocessing {}...".format(scene))
        offset = meta["offset"]
        bin_width_ps = meta["bin_width_ps"]
        min_r = meta["min_r"]
        max_r = meta["max_r"]
        # rootdir = os.path.join(data_dir, scene)
        # scenedir = os.path.join(output_dir, scene)
        # safe_makedir(os.path.join(scenedir))

        bin_width_m = bin_width_ps * 3e8 / (2 * 1e12)
        min_depth_bin = np.floor(min_r / bin_width_m).astype('int')
        max_depth_bin = np.floor(max_r / bin_width_m).astype('int')
        # Compensate for z translation between SPAD and RGB camera
        min_depth = min_depth_bin * bin_width_m - TranslationOfSpad[2]/1e3
        # print(TranslationOfSpad)
        max_depth = (max_depth_bin + 1) * bin_width_m - TranslationOfSpad[2]/1e3
        # sid_obj_init = SI(n_bins=600, alpha=min_depth, beta=max_depth, offset=0)
        # ambient_max_depth_bin = 100

        # RGB from Kinect
        rgb, rgb_cropped, intensity, crop = load_and_crop_kinect(scene_dir, kinect_file='kinect.mat')
        if not use_intensity:
            intensity = np.ones_like(intensity)

        # Load all the SPAD and kinect data
        spad = load_spad(scene_dir/'spad'/'data_accum.mat')
        spad_relevant = spad[..., min_depth_bin:max_depth_bin]
        spad_single_relevant = np.sum(spad_relevant, axis=(0,1))
        ambient_estimate = np.mean(spad_single_relevant[:ambient_max_depth_bin])
        # np.save(os.path.join(scenedir, "spad_single_relevant.npy"), spad_single_relevant)


        # Get ground truth depth
        idx_gt = np.argmax(spad[..., :max_depth_bin], axis=2)
        r_gt = signal.medfilt(np.fliplr(np.flipud((idx_gt * bin_width_m).T)), kernel_size=5)
        mask = (r_gt >= min_depth).astype('float').squeeze()
        z_gt = r_to_z(r_gt, spad_intrinsics["FocalLength"])
        z_gt = undistort_img(z_gt, **spad_intrinsics)
        mask = np.round(undistort_img(mask, **spad_intrinsics))
        # Nearest neighbor upsampling to reduce holes in output
        scale_factor = 2
        z_gt_up = cv2.resize(z_gt, dsize=(scale_factor*z_gt.shape[0], scale_factor*z_gt.shape[1]),
                                interpolation=cv2.INTER_NEAREST)
        mask_up = cv2.resize(mask, dsize=(scale_factor*mask.shape[0], scale_factor*mask.shape[1]),
                                interpolation=cv2.INTER_NEAREST)

        # Project GT depth and mask to RGB image coordinates and crop it.
        z_gt_proj, mask_proj = project_depth(z_gt_up, mask_up, (rgb.shape[0], rgb.shape[1]),
                                                spad_intrinsics["FocalLength"]*scale_factor,
                                                kinect_intrinsics["FocalLength"],
                                                spad_intrinsics["PrincipalPoint"]*scale_factor,
                                                kinect_intrinsics["PrincipalPoint"],
                                                RotationOfKinect, TranslationOfKinect/1e3)
        z_gt_proj_crop = z_gt_proj[crop[0]+offset[0]:crop[1]+offset[0],
                                    crop[2]+offset[1]:crop[3]+offset[1]]
        z_gt_proj_crop = signal.medfilt(z_gt_proj_crop, kernel_size=5)
        mask_proj_crop = (z_gt_proj_crop >= min_depth).astype('float').squeeze()

        # print("z_gt_proj_crop range:")
        # print(np.min(z_gt_proj_crop))
        # print(np.max(z_gt_proj_crop))

        out = {
            'depth': z_gt_proj_crop,
            'mask': mask_proj_crop,
            'image': rgb_cropped,
            'r_transient': spad_relevant,
            'min_depth': min_depth,
            'max_depth': max_depth,
        }

        np.save(output_dir/f'{scene}', out)

        # DEBUG
        break
