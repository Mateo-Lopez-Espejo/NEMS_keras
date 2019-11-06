import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt



def make_delay_lines(X, delay):
    '''
    turns a 2d array into a version with delayed lines looking back in time
    :param X: 2d ndarray. shape Channels x Time
    :param delay: the amount of delay to stack in the delayed array
    :return: 2d ndarray. shape (Channels * delays) x Time
    '''
    W = np.zeros((int(X.shape[0] * delay), X.shape[1])) # (Chan * Delay) * Time
    X = np.pad(X, ((0,0), (delay - 1, 0)), mode='constant', constant_values=0)
    for tt in range(W.shape[1]):
        W[:,tt] = np.reshape(X[:, tt:tt + delay].T, W.shape[0])
    return W

def strf_to_vector(strf):
    vector = np.reshape(strf.T, int(strf.shape[0]*strf.shape[1]))
    return vector

def vector_to_strf(vector, delays):
    channels = vector.size/delays
    if not channels.is_integer(): raise ValueError('cannot divide vector by delays')
    channels = int(channels)
    strf = np.reshape(vector, (delays, channels)).T
    return strf


delays = 21
channels = 18
timebins = 40000

# generates some random data and transfomrs into delayed lines
X =  np.random.rand(channels, timebins) # Channels x Time
delayed = make_delay_lines(X, delays)

# defines a ground truth STRF and transforms into a vecotr
true_strf =  np.random.rand(channels, delays)
true_vect = strf_to_vector(true_strf)

# # plots the stim, delayed lines version, STRF and vectorized STRF
# fig, axes = plt.subplots(2,2)
# axes = np.ravel(axes)
# axes[0].imshow(X, aspect='auto')
# axes[0].set_title('stimulus')
# axes[1].imshow(delayed, aspect='auto')
# axes[1].set_title('stim delayed lines')
# axes[2].imshow(true_strf, aspect='auto')
# axes[2].set_title('true STRF')
# axes[3].imshow(true_vect[:, None],)
# axes[3].set_title('true vector')

# uses matrix multiplication to get the STRF response(Y) to the dummy sound (X)
Y = np.matmul(true_vect, delayed)

# defienes linear model
model = Sequential()
model.add(Dense(1, input_dim=int(channels*delays), kernel_initializer='normal', activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# fits the model and transforms the vector into the equivalent STRF
model.fit(delayed.T , Y, batch_size=100, epochs=10000, verbose=1)

fit_vect = model.layers[0].get_weights()[0].squeeze()
fit_strf = vector_to_strf(fit_vect, delays)

# plots the true and fitted STRFS
fig, axes = plt.subplots(1,2)
axes = np.ravel(axes)
axes[0].imshow(true_strf, aspect='auto')
axes[0].set_title('ground truth')

axes[1].imshow(fit_strf, aspect='auto')
axes[1].set_title('fited values')


# predicts a new dataset with the model

Z = make_delay_lines(np.random.rand(channels, 100), delays)

real_Zresp = np.matmul(true_vect, Z)
pred_Zresp = model.predict(Z.T)

fig, ax = plt.subplots()
ax.plot(real_Zresp, color='black', label='real')
ax.plot(pred_Zresp, color='green', label='predicted')
ax.legend()
ax.set_title('real vs predicted response to a novel stim')


