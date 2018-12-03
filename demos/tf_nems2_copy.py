import os
import numpy as np
# from scipy.signal import decimate
import pandas as pd
import gc

from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
import keras

#run load_array_demo_v3.py # IPython
# exec(open('load_array_demo_v3.py').read()) # Python

from nems.recording import load_recording, get_demo_recordings
import nems.epoch as ep
import nems.preprocessing as preproc

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import io
from time import sleep

#import nems.recording
import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
import nems.utils

#import nems_db.baphy as nb
import nems_db.db as nd
import nems_db.xform_wrappers as nw
from nems_lbhb.old_xforms.xform_wrappers import generate_recording_uri as ogru
import nems_lbhb.old_xforms.xform_helper as oxfh
import logging

log = logging.getLogger(__name__)

import nems.xforms as xforms
from nems import get_setting
from nems.registry import KeywordRegistry
from nems.plugins import default_keywords
from nems.plugins import default_loaders
from nems.plugins import default_initializers
from nems.plugins import default_fitters

# Window stimulus



def windowed(X, context=12):
    W = np.zeros((X.shape[1],context,X.shape[0],1))
    X = np.pad(X,((0,0),(context-1,0)),'constant')
    for i in range(W.shape[0]):
        W[i,:,:,0] = X[:,i:i+context].T
    return W


def reinitLayers(model):
    """ from:
        https://www.codementor.io/nitinsurya/how-to-re-initialize-keras-model-weights-et41zre2g
    """
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            print('reinitializing layer {}'.format(layer.name))
            layer.kernel.initializer.run(session=session)


def fit_model(cellid, batch=289, spec=[4,8]):

    loadkey="ozgf.fs100.ch18"
    recording_uri = nw.generate_recording_uri(cellid, batch, loadkey)

    rec = load_recording(recording_uri)
    rec['resp']=rec['resp'].rasterize()

    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')

    # Optional: Take nanmean of ALL occurrences of all signals
    est = preproc.average_away_epoch_occurrences(est, epoch_regex='^STIM_')
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')

    X_tr = est['stim'].as_continuous()
    Y_tr = est['resp'].as_continuous()
    X_te = val['stim'].as_continuous()
    Y_te = val['resp'].as_continuous()

    goodbins = np.isfinite(Y_tr[0,:])
    Y_tr = Y_tr[:,goodbins]
    X_tr = X_tr[:,goodbins]
    goodbins = np.isfinite(Y_te[0,:])
    Y_te = Y_te[:,goodbins]
    X_te = X_te[:,goodbins]

#    maxx=np.nanmax(X_tr)
#    maxy=np.nanmax(Y_tr)
#    X_tr /= maxx
#    X_te /= maxx
#    Y_tr /= maxy
#    Y_te /= maxy

    X_te = windowed(X_te)
    X_tr = windowed(X_tr)
    Y_te = Y_te.T
    Y_tr = Y_tr.T

    print('X_te :=',X_te.shape)
    print('Y_te :=',Y_te.shape)
    print('X_tr :=',X_tr.shape)
    print('Y_tr :=',Y_tr.shape)

    num_channels = Y_tr.shape[1]

    model = Sequential()
    model.add(Conv2D(spec[0],(3,4),activation='relu',padding='valid',kernel_initializer='he_normal',kernel_regularizer='l2',input_shape=X_tr.shape[1:]))
    model.add(Dropout(.3))
    # model.add(Conv2D(4,3,activation='relu',padding='valid',kernel_initializer='he_normal',kernel_regularizer='l2'))
    # model.add(Dropout(.3))
    #model.add(Conv2D(4,1,activation='relu',padding='valid',kernel_initializer='he_normal',kernel_regularizer='l2'))
    #model.add(Dropout(.3))
    model.add(Flatten())
    #model.add(Dense(spec[0],activation='relu',kernel_initializer='he_normal',kernel_regularizer='l2'))
    model.add(Dense(spec[1],activation='relu',kernel_initializer='he_normal',kernel_regularizer='l2'))
    model.add(Dense(1,activation='relu',kernel_initializer='he_normal',kernel_regularizer='l2',use_bias=False))
    #model.add(Dense(1,activation='linear',kernel_initializer='he_normal',kernel_regularizer='l2',use_bias=False))

    # Save model configuration
    model.summary()
    model_json = model.to_json()

    print('Training model for cell %s) ----------------'%(cellid))

    # Training options
    modelstring="{}_{}".format(spec[0],spec[1])

    filepath = 'models2/model_%s_%s.hdf5'%(modelstring,cellid)
    checkpoint = ModelCheckpoint(filepath,mode='min',monitor='val_loss',save_best_only=True,save_weights_only=True)
    early_stop = EarlyStopping(mode='min',monitor='val_loss',patience=5)
    callbacks = [checkpoint, early_stop]

    # Train model
    model = model_from_json(model_json)
    model.compile(loss='mse', optimizer='rmsprop')

    good_start=False
    c=0
    while (c<10) and not good_start:
        print('Initialization {}'.format(c))
        Z_tr = model.predict(X_tr)
        tr_corr=np.corrcoef(Z_tr.T, Y_tr.T)[0,1]
        print("Initial cc: {}".format(tr_corr))
        model.fit(X_tr,Y_tr,batch_size=64,epochs=5,validation_split=.2,
                  verbose=2,callbacks=callbacks)
        model.load_weights(filepath)
        Z_tr = model.predict(X_tr)
        tr_corr=np.corrcoef(Z_tr.T, Y_tr.T)[0,1]
        print("Training cc after 3 epochs: {}".format(tr_corr))
        if np.isfinite(tr_corr) and (tr_corr>0):
            good_start=True
        else:
            reinitLayers(model)
        c += 1

    model.fit(X_tr,Y_tr,batch_size=64,initial_epoch=5,
              epochs=20,validation_split=.2,
              verbose=2,callbacks=callbacks)
    model.load_weights(filepath)

    # Check test set prediction
    Z_te = model.predict(X_te)
    corr = np.corrcoef(Z_te.T, Y_te.T)[0,1]

    return corr

n0="ozgf.fs50.ch18-ld-sev_dlog-wc.18x2.g-fir.2x15_init-basic"
n1="ozgf.fs50.ch18-ld-sev_dlog-wc.18x2.g-stp.2-fir.2x15_init-basic"
cellid = 'TAR010c-18-2'
batch = 289
try:
    d=pd.read_csv('modelcomp3.csv', index_col='cellid')
except FileNotFoundError:
    d=nd.batch_comp(batch=batch, modelnames=[n0,n1], cellids=[cellid])

speclist=[[3,8],[3,10],[3,12],[3,16],[3,24],
          [4,8],[4,10],[4,12],[4,16],[4,24],
          [5,8],[5,10],[5,12],[5,16],[5,24]]

celllist=list(d.index)
errorlist=[]
for spec in speclist:
    modelstring="{}_{}".format(spec[0],spec[1])
    if modelstring not in d:
        d[modelstring] = pd.Series(np.zeros(len(d))*np.nan, index=d.index)

    f=plt.figure()
    for i,cellid in enumerate(celllist[50:150]):
        print('{}: cell={}, model={}'.format(i,cellid,modelstring))
        # if np.isnan(d.loc[cellid,modelstring]) or (d.loc[cellid,modelstring]==0):
        if np.isnan(d.loc[cellid,modelstring]):
            try:
                #K.clear_session()
                d.loc[cellid,modelstring] = fit_model(cellid, batch=289, spec=spec)
                if np.isnan(d.loc[cellid,modelstring]):
                    d.loc[cellid,modelstring] = 0
                print("cellid = {} keras {}={:.3f} vs nems={:.3f}".format(
                        cellid,modelstring,d.loc[cellid,modelstring],d.loc[cellid,n1]))
                f.clf()
                x=np.array(d[n1])
                y=np.array(d[modelstring])
                k=np.isfinite(x) & np.isfinite(y)
                plt.plot(np.array([0,1]),np.array([0,1]),'k--')
                plt.plot(x[k],y[k],'.')
                plt.xlabel('nems (mean {:.3f})'.format(np.mean(x[k])))
                plt.ylabel('keras {} (mean {:.3f})'.format(
                        modelstring,np.mean(y[k])))
                plt.draw()
                sleep(0.01)
                f.savefig('models2/model_{}.pdf'.format(modelstring))
                d.to_csv('modelcomp3.csv')
            except:
                errorlist.append([cellid,modelstring])
            sleep(0.1)
        else:
            print('{} {}, r={:.3f}'.format(
                    cellid,modelstring,d.loc[cellid,modelstring]))
