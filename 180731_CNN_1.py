import format_recording
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
import keras.backend as K


# todo define a 2d convolutional model with delayed lines and a 1D convolutional model, which sould take care of the time

def cnn_1(formated_rec, **kwargs):

    # gets the shape of the

    model = Sequential()
    model.add(Conv2D(filters=4,kernel_size=()))


def cnn_2(formated_rec, **kwargs):

    return None