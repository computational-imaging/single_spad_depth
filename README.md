

# Disambiguating Monocular Depth Estimation with a Single Transient

1.  [Setup and Installation](#orgaad60bb)
2.  [Data](#org767979a)
3.  [Running on NYU Depth v2](#orgdaf3ca1)
4.  [Running on Captured Data](#orge67e9c4)


<a id="orgaad60bb"></a>

## Setup and Installation


### Anaconda

We recommend using anaconda and creating an environment via the following
command.

    $ conda create -f environment.yml

This will create an environment called `spad-single` which can be activated via
the command

    $ conda activate spad-single


### Pip

Run the following command from the root directory. This will install all the
required packages into the current environment.

    $ python setup.py install


<a id="org767979a"></a>

## Data


### NYU Depth v2

Download the labeled dataset and the official train/test splits by running the
following commands from the root directory

    $ curl http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat -o ./data/nyu_depth_v2/raw/nyu_depth_v2_labeled.mat
    $ curl http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat -o ./data/nyu_depth_v2/raw/splits.mat


<a id="orgdaf3ca1"></a>

## Running on NYU Depth v2

    python -m eval_nyuv2.py


<a id="orge67e9c4"></a>

## Running on Captured Data

