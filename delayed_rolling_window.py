'''
how toe implement an encoding/decoding linear model that considers a window in the past, of a certain size (duration)
and at a certain offset (latency of the window)
'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# simulate some ground truth data with a linear transformation and a delay


data =  np.random.rand(100, 5)
true_trans =  np.random.rand(5, 1)
true_offset = 3 # offset in time bins
# true_trans = np.array([[1], [0], [0], [0], [0]])
labels = np.matmul(data, true_trans)
