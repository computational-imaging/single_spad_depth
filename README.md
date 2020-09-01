

# Disambiguating Monocular Depth Estimation with a Single Transient

1.  [Setup and Installation](#orgb153ad8)
2.  [Getting and Preprocessing the Data](#orga3ff254)
3.  [Running on NYU Depth v2](#org9a31e9c)
4.  [Running on Captured Data](#orgcab9406)


<a id="orgb153ad8"></a>

## Setup and Installation


### Anaconda

We recommend using anaconda and creating an environment via the following
command.

    conda create -f environment.yml

This will create an environment called `spad-single` which can be activated via
the command

    conda activate spad-single


<a id="orga3ff254"></a>

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

TODO


### Model Weights

DORN, DenseDepth, and MiDaS weights can be downloaded at the following links:

-   [DORN](https://drive.google.com/uc?export=download&id=1WPD2mf2wSvPwisaeeEDvzyxkAekj_rxR)
-   [DenseDepth](https://drive.google.com/uc?export=download&id=1Ua73crX4X8ma4h-MEIF9C1gXLmWOt8Yn)
-   [MiDaS](https://drive.google.com/uc?export=download&id=1ug1z2zmZA-ZTtOz8m7d_cDIbgu8FuRhi)

Each should be placed in the relevant `*_backend` folder in the `models` directory.


<a id="org9a31e9c"></a>

## Running on NYU Depth v2

The basic pattern is to run the command

    python eval_nyuv2.py \
      -c configs/<MDE>/<method.yml>
      [--sbr SBR]
      [--gpu GPU]

MDE can be DORN, DenseDepth, or MiDaS.
method.yml can take on the following values:

-   `mde.yml` to run the MDE alone (default) - DORN and DenseDepth only.
-   `median.yml` for median matching - DORN and DenseDepth only.
-   `gt_hist.yml` for ground truth histogram matching
-   `transient.yml` for transient matching, note that the `--sbr` option will need
    to be set if this is used.

For running on GPU, use the `--gpu` argument with number indicating which one to
use.
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


<a id="orgcab9406"></a>

## Running on Captured Data

TODO

