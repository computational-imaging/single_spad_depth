

# Disambiguating Monocular Depth Estimation with a Single Transient

1.  [Setup and Installation](#orgce76c9f)
2.  [Getting and Preprocessing the Data](#org1a56cc0)
3.  [Running on NYU Depth v2](#org9c6f553)
4.  [Running on Captured Data](#org94f8924)


<a id="orgce76c9f"></a>

## Setup and Installation


### Anaconda

We recommend using anaconda and creating an environment via the following
command.

    conda create -f environment.yml

This will create an environment called `spad-single` which can be activated via
the command

    conda activate spad-single


<a id="org1a56cc0"></a>

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


<a id="org9c6f553"></a>

## Running on NYU Depth v2

The basic pattern is to run the command

    python eval_nyuv2.py \
    [--eval-config method.yml] \
    [--mde-config single_spad_depth/models/<mde.cfg>] \
    [--mde-transient-config single_spad_depth/models/<mde_transient.cfg>]
    [--sbr SBR]
    [--gpu GPU]

method.yml can take on the following values:

-   `mde.yml` to run the MDE alone (default)
-   `median.yml` for median matching
-   `gt_hist.yml` for ground truth histogram matching
-   `transient.yml` for transient matching, note that the `--sbr` option will need
    to be set if this is used.

mde.cfg selects the monocular depth estimator and can be one of

-   `dorn_mde.cfg` (default)
-   `densedepth_mde.cfg`
-   `midas_mde.cfg`

For running transient matching, additionally specify mde<sub>transient.cfg</sub> to be one
of

-   `dorn_transient.cfg`
-   `densedepth_transient.cfg`
-   `midas_transient.cfg`

according to which model is being used. This sets further model-specific
parameters necessary for our method.
For running on GPU, use the `--gpu` argument with number indicating which one to
use.

An example command is given in `example.sh`


<a id="org94f8924"></a>

## Running on Captured Data

TODO

