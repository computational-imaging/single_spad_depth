

# Disambiguating Monocular Depth Estimation with a Single Transient

<http://www.computationalimaging.org/publications/single_spad/>

1.  [Setup and Installation](#orgf06a742)
2.  [Getting and Preprocessing the Data](#org6252436)
3.  [Running on NYU Depth v2](#orgda7b46c)
4.  [Running on Scanned Data](#org34e6a00)
5.  [Diffuse SPAD Example](#org7c2da03)
6.  [Two Planes Example](#orgbfffee3)
7.  [Citation and Contact Info](#org1a38d63)


<a id="orgf06a742"></a>

## Setup and Installation


### Anaconda

We recommend using anaconda and creating an environment via the following
command.

    conda env create -f environment.yml

This will create an environment called `single-spad-depth` which can be activated via
the command

    conda activate single-spad-depth


<a id="org6252436"></a>

## Getting and Preprocessing the Data


### NYU Depth v2

Download the labeled dataset and the official train/test splits by running the
following commands from the root directory

    curl http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat  -o ./data/nyu_depth_v2/raw/nyu_depth_v2_labeled.mat
    curl http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat -o ./data/nyu_depth_v2/raw/splits.mat

Then, run the split script to generate `.npz` files containing the dataset
parts:

    python split_nyuv2.py

To simulate the SPAD on the test set run the command

    python simulate_single_spad.py test


### Captured Data

1.  Scanned

    Data for the scanned scenes [can be downloaded here.](https://drive.google.com/uc?export=download&id=1uckREyTwRShJBOVr0HWgbmu4oqPpNmxH)
    After downloading and extracting the file to the `data/captured/raw` directory, run
    
        python preprocess_scans.py
    
    to generate `.npy` and `.yml` files with the data and configs necessary to run
    the model, respectively.

2.  Scanned + Diffuse

    [Data available here,](https://drive.google.com/uc?export=download&id=1brsjTX_kFIEn2Pmj8CrEmc4OU2GewrIA) download and extract to the `data/captured/raw` directory.

3.  Two Planes Image

    [Data available here,](https://drive.google.com/uc?export=download&id=1oAl2q_SuzwaG2aMj9OaUcW8ECC09Kww6) download and extract to the `data/captured/raw` directory.


### Model Weights

DORN, DenseDepth, and MiDaS weights can be downloaded at the following links:

-   [DORN](https://drive.google.com/uc?export=download&id=1WPD2mf2wSvPwisaeeEDvzyxkAekj_rxR)
-   [DenseDepth](https://drive.google.com/uc?export=download&id=1Ua73crX4X8ma4h-MEIF9C1gXLmWOt8Yn)
-   [MiDaS](https://drive.google.com/uc?export=download&id=1ug1z2zmZA-ZTtOz8m7d_cDIbgu8FuRhi)

Each should be placed in the relevant `*_backend` folder in the `models` directory.


<a id="orgda7b46c"></a>

## Running on NYU Depth v2

The basic pattern is to run the command

    python eval_nyuv2.py \
      -c configs/<MDE>/<method.yml>
      [--sbr SBR]
      [--gpu GPU]

MDE can be `dorn`, `densedepth`, or `midas`.
method.yml can take on the following values:

-   `mde.yml` to run the MDE alone (default) - DORN and DenseDepth only.
-   `median.yml` for median matching - DORN and DenseDepth only.
-   `gt_hist.yml` for ground truth histogram matching
-   `transient.yml` for transient matching, note that the `--sbr` option will need
    to be set if this is used.

For running on GPU, use the `--gpu` argument with number indicating which one to
use.


### Shell scripts

Also provided are three shell scripts, `run_all_<method>.sh` which will run all
the MDEs on the given method.


### Results

Results are automatically saved to
`results/<method>/<mde>/[<sbr>]/[summary.npy|preds_cropped.npy]` files.
`summary.npy` is a dictionary of aggregate metrics and can be loaded using

    import numpy as np
    d = np.load('summary.npy', allow_pickle=True)[()]

`preds_cropped.npy` contains an N by 440 by 592 numpy array containing the final depth
estimates on the official NYUv2 center crop of each image.


<a id="org34e6a00"></a>

## Running on Scanned Data

The basic pattern is to run the command:

    python eval_captured.py \
        --mde-config configs/captured/<mde>.yml \
        --scene-config data/captured/processed/<scene>.yml\
        --method METHOD \
        [--gpu GPU]

`mde` can be `dorn`, `densedepth`, or `midas`.
`scene` is one of the above scenes.
`METHOD` is either `mde` or `transient`.
`GPU` is the number of the gpu to run on.


### Shell scripts

Also provided are shell scripts of the form `run_<scene>.sh` which can be run to
run all of the MDEs on that scene. `--method` still must be specified.


### Results

Results are saved in the `results_captured` folder. A jupyter notebook is
provided for inspecting the results.


<a id="org7c2da03"></a>

## Diffuse SPAD Example

A jupyter notebook is provided for running the method on the diffuse spad scene.
It also provides a good reference for how to use the API if one wishes to
isolate particular parts, such as the MDEs, the transient preprocessing, or the
histogram matching.


<a id="orgbfffee3"></a>

## Two Planes Example

A jupyter notebook is provided for comparing the transients produced by the
scanned and diffuse methods on the two planes image.


<a id="org1a38d63"></a>

## Citation and Contact Info

M. Nishimura, D. B. Lindell, C. Metzler, G. Wetzstein, “Disambiguating Monocular Depth Estimation with a Single Transient”, European Conference on Computer Vision (ECCV), 2020.


### BibTeX

    @article{Nishimura:2020,
    author={M. Nishimura and D. B. Lindell and C. Metzler and G. Wetzstein},
    journal={European Conference on Computer Vision (ECCV)},
    title={{Disambiguating Monocular Depth Estimation
    with a Single Transient}},
    year={2020},
    }


### Contact info

For more questions please email Mark Nishimura: markn1 at stanford dot edu

