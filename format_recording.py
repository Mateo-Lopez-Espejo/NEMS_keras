import nems_db as nd
import nems_db.baphy as nb
import nems.recording as recording
import nems.epoch as ne
import numpy as np
from nems.recording import load_recording


'''
deal with initial data wrangling to place in different formats compatible with the Keras workframe.
Namely downsample, and clean data
Transform in a format with delayed lines (memory)
splitn into training, (validation) and testing sets 
'''

def as_delayed_lines(X, time_window=40):
    '''
    for a time series X with dimentions C x T where C is channel and T is time bin,
    returns a 4d matrix T x W x C x B where T is time bins, W is time window, and B is batch. to complete early time
    windows, padds with zeros
    todo figure out why this empty dimentions in necesary
    :param X: 2d matrix  Channel x Time
    :param time_window: lenght of the delayed line, it will be equivalent to the length of feature vectors intput to the
                        Keras model
    :return: 4d matrix Time x Window x Channel x Batch
    '''
    W = np.zeros((X.shape[1], time_window, X.shape[0], 1))
    X = np.pad(X, ((0,0),(time_window - 1, 0)), 'constant')
    for i in range(W.shape[0]):
        W[i,:,:,0] = X[:,i:i + time_window].T
    return W


def sig_to_keras(signal, delayed_line, val_frac, test_frac=0):
    '''
    splits a signal into training, v alidation and test sets, and reshapes as delayed lines.
    :param signal: nems Signal object
    :param delayed_line: lengh (in time bins) of the delayed line window
    :param val_frac: factions used for validation
    :param test_frac: fraction used for testing
    :return:
    '''
    # gets the array
    matr = signal.rasterize().as_continuous() # shape Channels x Time


    if val_frac + test_frac >= 1:
        raise ValueError("val_frac plus test_frac cannot be the entire signal")

    # splits in traning, validation and test fractions
    subsets = dict()
    train_val_split = int(matr.shape[1] * (1 - val_frac - test_frac))
    val_test_split = int(matr.shape[1] * (1 - test_frac))
    subsets['training'] = matr[:, :train_val_split]
    subsets['validation'] = matr[:, train_val_split:val_test_split]
    if test_frac != 0:
        subsets['test'] = matr[:, val_test_split:]
    else:
        pass

    # reshapes into delayed lines, vals in dict have shape Window x Time x Channel x Batch
    subsets = {key: as_delayed_lines(val, delayed_line) for key, val in subsets.items()}

    return subsets


