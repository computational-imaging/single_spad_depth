

# Disambiguating Monocular Depth Estimation with a Single Transient

1.  [Setup and Installation](#org068fb5c)
2.  [Getting and Preprocessing the Data](#orgefeb5b6)
3.  [Running on NYU Depth v2](#orgf05b39c)
4.  [Running on Captured Data](#orgd3edfea)


<a id="org068fb5c"></a>

## Setup and Installation


### Anaconda

We recommend using anaconda and creating an environment via the following
command.

    conda create -f environment.yml

This will create an environment called `spad-single` which can be activated via
the command

    conda activate spad-single


<a id="orgefeb5b6"></a>

## Getting and Preprocessing the Data


### NYU Depth v2

Download the labeled dataset and the official train/test splits by running the
following commands from the root directory

    curl http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat  -o ./data/nyu_depth_v2/raw/nyu_depth_v2_labeled.mat
    curl http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat -o ./data/nyu_depth_v2/raw/splits.mat

Then, run the split script to generate `.npz` files containing the dataset
parts:

    python data/nyu_depth_v2/split_nyuv2.py

To simulate the SPAD on the test set, go to the parent folder of this repo and
run the script

    python -m single_spad_depth.data.nyu_depth_v2.simulate_single_spad test


### Captured Data

TODO


### Model Weights

DORN, DenseDepth, and MiDaS weights can be downloaded at the following links:

-   =

They should be placed in the relevant \`\*<sub>backend</sub>\` folder in the \`models\` directory.


<a id="orgf05b39c"></a>

## Running on NYU Depth v2

The basic pattern is to go to the parent folder of this repo and run the
following command

    python -m single_spad_depth.eval_nyuv2 \
    [--eval-config single_spad_depth/method.yml] \
    [--mde-config single_spad_depth/models/<mde.cfg>] \
    [--sbr SBR]

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


<a id="orgd3edfea"></a>

## Running on Captured Data

TODO

