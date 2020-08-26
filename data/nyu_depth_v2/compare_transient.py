#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

processed = Path(__file__).parent/'processed'
orig_file = processed/'test_int_True_fall_True_lamb_False_dc_10000.0_jit_True_poiss_True_spad.npy'
new_file = processed/'counts_test.npz'

if __name__ == '__main__':
    # Check spad simulation
    orig = np.load(orig_file, allow_pickle=True)[()]
    orig_int = orig['intensity']
    orig_spad = orig['spad']

    new_spad = np.load(new_file, allow_pickle=True)['100.0']

    plt.plot(orig_spad[0, :], linewidth=0.5, label='orig')
    plt.plot(new_spad[0, :], linewidth=0.5, label='new')
    # plt.savefig('out.pdf')

    #
