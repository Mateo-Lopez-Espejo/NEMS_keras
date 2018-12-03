import os
import numpy as np
from scipy.signal import decimate

from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint


# %run load_array_demo_v3.py # IPython
filename = '/home/mateo/NEMS_keras/load_array_demo_v3.py'
exec(open(filename).read()) # Python
print('X  :=',X.shape)
print('Y  :=',Y.shape)
print('Xa :=',Xa.shape)
print('Ya :=',Ya.shape)

# Split to test and train sets
data_sr = 200
test_len = 50*data_sr
X_te, X_tr = Xa[:,:test_len], Xa[:,test_len:]
Y_te, Y_tr = Ya[:,:test_len], Ya[:,test_len:]

# Downsample to 100Hz
# X_te = X_te[:,::2]
# Y_te = Y_te[:,::2]
# X_tr = X_tr[:,::2]
# Y_tr = Y_tr[:,::2]

# Downsample to 100Hz # todo ask stephen why start from 1 and not 0
X_te = (X_te[:,::2] + X_te[:,1::2])
Y_te = (Y_te[:,::2] + Y_te[:,1::2])
X_tr = (X_tr[:,::2] + X_tr[:,1::2])
Y_tr = (Y_tr[:,::2] + Y_tr[:,1::2])

# Downsample to 100Hz
# X_te = decimate(X_te,2,axis=1)
# Y_te = decimate(Y_te,2,axis=1)
# X_tr = decimate(X_tr,2,axis=1)
# Y_tr = decimate(Y_tr,2,axis=1)

# Window stimulus is this the delayed lines???
def windowed(X, context=40):
    W = np.zeros((X.shape[1],context,X.shape[0],1))
    X = np.pad(X,((0,0),(context-1,0)),'constant')
    for i in range(W.shape[0]):
        W[i,:,:,0] = X[:,i:i+context].T
    return W

X_te = windowed(X_te)
X_tr = windowed(X_tr)
Y_te = Y_te.T
Y_tr = Y_tr.T

print('X_te :=',X_te.shape)
print('Y_te :=',Y_te.shape)
print('X_tr :=',X_tr.shape)
print('Y_tr :=',Y_tr.shape)

num_channels = Y_tr.shape[1]

# Define model
model = Sequential()
model.add(Conv2D(4,3,activation='relu',padding='valid',kernel_initializer='he_normal',kernel_regularizer='l2',input_shape=X_tr.shape[1:]))
model.add(Dropout(.3))
# model.add(Conv2D(4,3,activation='relu',padding='valid',kernel_initializer='he_normal',kernel_regularizer='l2'))
# model.add(Dropout(.3))
# model.add(Conv2D(4,1,activation='relu',padding='valid',kernel_initializer='he_normal',kernel_regularizer='l2'))
# model.add(Dropout(.3))
model.add(Flatten())
model.add(Dense(64,activation='relu',kernel_initializer='he_normal',kernel_regularizer='l2'))
model.add(Dense(1,activation='relu',kernel_initializer='he_normal',kernel_regularizer='l2',use_bias=False))

# Save model configuration
model.summary()
model_json = model.to_json()

models = []
corrs = []
for ch in range(num_channels):
    print('Training model for channel %d -----------------'%(ch+1))
    model = model_from_json(model_json)
    model.compile(loss='mse', optimizer='rmsprop')
    
    # Training options
    filepath = 'model_ch%02d.hdf5'%(ch+1)
    checkpoint = ModelCheckpoint(filepath,mode='min',monitor='val_loss',save_best_only=True,save_weights_only=True)
    early_stop = EarlyStopping(mode='min',monitor='val_loss',patience=5)
    callbacks = [checkpoint, early_stop]
    
    # Train model
    model.fit(X_tr,Y_tr[:,ch],batch_size=64,epochs=20,validation_split=.2,verbose=2,callbacks=callbacks)
    model.load_weights(filepath)
    models.append(model)
    
    # Check test set prediction
    Z_te = model.predict(X_te)
    corr = np.corrcoef(Z_te.T, Y_te[:,ch].T)[0,1]
    corrs.append(corr)
print(corrs)