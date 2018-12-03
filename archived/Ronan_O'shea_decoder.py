#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 18:22:15 2018
@author: roshea101
"""
import sys

sys.path.append('/Users/roshea101/NEMS')
import nems.decoder as dc
import nems
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nems.recording import load_recording
import nems.recording as recording
import nems.plots.api as nplt
import nems.epoch as ep
import nems.preprocessing as preproc
from nems.signal import RasterizedSignal
import copy
from scipy.signal import convolve2d
from random import *
from nems.recording import Recording
from nems.signal import RasterizedSignal, PointProcess
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

uri = '/Users/roshea101/NEMS/recordings/pupil/TAR010c.NAT.fs50.tgz'
rec = load_recording(uri)


def PCA(r, center=True):
    """
    computes pca on the input matrix r.
    r can also be a nems signal, either will work.

    output is a dictionary containing the PCs, variance explained per net pcs, stepwise
    variance explained per individual pc, and the loading (rotation) matrix.
    """
    if type(r) is RasterizedSignal:
        r_pca = r.as_continuous()
    elif type(r) is PointProcess:
        r_pca = r.rasterize().as_continuous()
    else:
        r_pca = r

    if center is True:
        m = np.mean(r_pca, axis=0)
        r_pca = r_pca - m;

    if r_pca.shape[0] < r_pca.shape[1]:
        r_pca = r_pca.T

    U, S, V = np.linalg.svd(r_pca, full_matrices=False)
    v = S ** 2
    step = v;
    var_explained = []
    for i in range(0, r_pca.shape[1]):
        var_explained.append(100 * (sum(v[0:(i + 1)]) / sum(v)));
    loading = V;
    pcs = U * S;

    out = {'pcs': pcs,
           'variance': var_explained,
           'step': step,
           'loading': loading
           }
    return out


def stim_estimator_PCA(rec, npcs):
    x = len(rec['resp'].chans)
    ##########
    pca_out = PCA(rec['resp'], center=False)
    rec['resp'] = rec['resp'].rasterize()._modified_copy(
        np.matmul(pca_out['pcs'][:, 0:npcs], pca_out['loading'][0:npcs, :]).T)
    for i in range(10):
        rec['stim'] = rec['stim'].rasterize()
        i += 1
    nfolds = 10
    ests, vals, m = preproc.mask_est_val_for_jackknife(rec, modelspecs=None,
                                                       njacks=nfolds)
    for i in range(10):
        vals[i]['stim'] = vals[i]['stim'].rasterize()
        i += 1
    new_val = recording.jackknife_inverse_merge(vals)
    S_est_list = []
    for i in range(10):
        est = ests[i].apply_mask()
        stim = est['stim']
        resp_est = est['resp'].rasterize()
        X_est = stim.rasterize().as_continuous()
        Y_est = resp_est.as_continuous()
        mean_array = dc.mean_array_resp(x, Y_est)
        Y_est_subtracted = dc.Y_est_subtractedavg(mean_array, x, Y_est)
        S = X_est  # stim est set
        R = Y_est_subtracted  # response est set
        R1 = np.concatenate((R[:x, 2:], np.zeros((x, 2))), axis=1)
        R2 = np.concatenate((R, R1), axis=0)
        R3 = np.concatenate((R[:x, 4:], np.zeros((x, 4))), axis=1)
        R4 = np.concatenate((R2, R3), axis=0)
        R5 = np.concatenate((R[:x, 6:], np.zeros((x, 6))), axis=1)
        R6 = np.concatenate((R4, R5), axis=0)
        R7 = np.concatenate((R[:x, 8:], np.zeros((x, 8))), axis=1)
        R8 = np.concatenate((R6, R7), axis=0)
        R9 = np.concatenate((R[:x, 10:], np.zeros((x, 10))), axis=1)
        R10 = np.concatenate((R8, R9), axis=0)
        R11 = np.concatenate((R[:x, 12:], np.zeros((x, 12))), axis=1)
        R12 = np.concatenate((R10, R11), axis=0)
        R13 = np.concatenate((R[:x, 14:], np.zeros((x, 14))), axis=1)
        R14 = np.concatenate((R12, R13), axis=0)
        R15 = np.concatenate((R[:x, 16:], np.zeros((x, 16))), axis=1)
        R16 = np.concatenate((R14, R15), axis=0)
        R17 = np.concatenate((R[:x, 18:], np.zeros((x, 18))), axis=1)
        R18 = np.concatenate((R16, R17), axis=0)
        R19 = np.concatenate((R[:x, 20:], np.zeros((x, 20))), axis=1)
        R20 = np.concatenate((R18, R19), axis=0)
        R_trans = np.transpose(R20)
        Crr = np.matmul(R20, R_trans)
        Crs = np.matmul(S, R_trans)
        Crr_inv = np.linalg.inv(Crr)
        G = np.matmul(Crs, Crr_inv)

        resp_val = new_val['resp']
        Y_val = resp_val.as_continuous()
        Y_val = np.nan_to_num(Y_val)
        shape = Y_val.shape
        y = shape[1]
        Y_val_segment = Y_val[:, int((y / 10) * i):int((y / 10) * (i + 1))]
        shape = Y_val_segment.shape
        y = shape[1]
        Y_val_subtractedavg = np.zeros(shape=(x, y))
        for u in range(x):
            Y_est_cell_mean = mean_array[u]
            difference = Y_val_segment[u, :] - Y_est_cell_mean
            Y_val_subtractedavg[u] = difference
            u += 1
        R_val = Y_val_subtractedavg
        R1_val = np.concatenate((R_val[:x, 2:], np.zeros((x, 2))), axis=1)
        R2_val = np.concatenate((R_val, R1_val), axis=0)
        R3_val = np.concatenate((R_val[:x, 4:], np.zeros((x, 4))), axis=1)
        R4_val = np.concatenate((R2_val, R3_val), axis=0)
        R5_val = np.concatenate((R_val[:x, 6:], np.zeros((x, 6))), axis=1)
        R6_val = np.concatenate((R4_val, R5_val), axis=0)
        R7_val = np.concatenate((R_val[:x, 8:], np.zeros((x, 8))), axis=1)
        R8_val = np.concatenate((R6_val, R7_val), axis=0)
        R9_val = np.concatenate((R_val[:x, 10:], np.zeros((x, 10))), axis=1)
        R10_val = np.concatenate((R8_val, R9_val), axis=0)
        R11_val = np.concatenate((R_val[:x, 12:], np.zeros((x, 12))), axis=1)
        R12_val = np.concatenate((R10_val, R11_val), axis=0)
        R13_val = np.concatenate((R_val[:x, 14:], np.zeros((x, 14))), axis=1)
        R14_val = np.concatenate((R12_val, R13_val), axis=0)
        R15_val = np.concatenate((R_val[:x, 16:], np.zeros((x, 16))), axis=1)
        R16_val = np.concatenate((R14_val, R15_val), axis=0)
        R17_val = np.concatenate((R_val[:x, 18:], np.zeros((x, 18))), axis=1)
        R18_val = np.concatenate((R16_val, R17_val), axis=0)
        R19_val = np.concatenate((R_val[:x, 20:], np.zeros((x, 20))), axis=1)
        R20_val = np.concatenate((R18_val, R19_val), axis=0)
        S_est = np.matmul(G, R20_val)
        S_est[S_est < 0] = 0
        S_est_list.append(S_est)
        i += 1
    S_est = np.concatenate((S_est_list[0], S_est_list[1], S_est_list[2], S_est_list[3], S_est_list[4], S_est_list[5],
                            S_est_list[6], S_est_list[7], S_est_list[8], S_est_list[9]), axis=1)
    return S_est


def mse_in_time_jack_pca(rec, bin_size, npcs):
    S_est = stim_estimator_PCA(rec, npcs)
    stim = rec['stim']
    X = stim.rasterize().as_continuous()
    a = S_est.shape
    time = a[1]
    mse_time_array = np.zeros(shape=(18, (int((time / 50) / bin_size))))
    for i in range(18):
        mse_time_list = []
        S_est_fq_channel = S_est[i, :]
        X_fq_channel = X[i, :]
        var = np.var(X_fq_channel)
        for u in range(int((time / 50) / bin_size)):
            S_est_timebin = S_est_fq_channel[((50 * bin_size) * u):((50 * bin_size) * (u + 1))]
            X_timebin = X_fq_channel[((50 * bin_size) * u):((50 * bin_size) * (u + 1))]
            squared_errors = (S_est_timebin - X_timebin) ** 2
            mse_timebin = np.nanmean(squared_errors)
            mse_timebin_norm = mse_timebin / var
            mse_time_list.append(mse_timebin_norm)
            u += 1
        mse_time_array[i] = mse_time_list
        i += 1
    return mse_time_array, S_est


mse_pca5, S_est_pca5 = mse_in_time_jack_pca(deepcopy(rec), 30, 5)
mse_pca10, S_est_pca10 = mse_in_time_jack_pca(deepcopy(rec), 30, 10)
# mse_pca15, S_est_pca15 = mse_in_time_jack_pca(deepcopy(rec), 30, 15)
# mse_pca20, S_est_pca20 = mse_in_time_jack_pca(deepcopy(rec), 30, 20)
# mse_pca25, S_est_pca25 = mse_in_time_jack_pca(deepcopy(rec), 30, 25)
# mse_pca30, S_est_pca30 = mse_in_time_jack_pca(deepcopy(rec), 30, 30)
# mse_pca35, S_est_pca35 = mse_in_time_jack_pca(deepcopy(rec), 30, 35)
# mse_pca40, S_est_pca40 = mse_in_time_jack_pca(deepcopy(rec), 30, 40)
# mse_pca45, S_est_pca45 = mse_in_time_jack_pca(deepcopy(rec), 30, 45)
# mse_pca50, S_est_pca50 = mse_in_time_jack_pca(deepcopy(rec), 30, 50)
# mse_pca55, S_est_pca55 = mse_in_time_jack_pca(deepcopy(rec), 30, 55)