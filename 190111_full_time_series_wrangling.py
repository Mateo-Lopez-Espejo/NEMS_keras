import numpy as np
import collections as col

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K



import cpp_reconstitute_rec as crec
import cpp_epochs as cep
import nker_epochs as kep
import nems.epoch as ne

'''
The purpose of this script is to format a NEMS Context Probe Pair recording, and eventually any recording into
a format designed for use in a machine learning paradig.
The main task of the script is to transform the epoch data frame into label to be fed into a machine learning 
algorithm. 
In the particular case of Context Probe Pairs, these labels are the identiti of the context of probe asociated
with a neural response. However this code should be extended to be capable of transforming any arbitrary epoch or
group of epochs int tags
'''



# get the data, formats as appropiate

site_IDs = crec.get_site_ids(310)

modelname = 'wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.S-dexp.1'
best_site = 'BRT056b'

reconstituted_rec = crec.reconsitute_rec(310, site_IDs[best_site], modelname) # loads any model
frec = cep.set_recording_subepochs(reconstituted_rec, set_pairs=False)

# keeps only relevant signal
rec = frec.copy()
rec['resp'] = rec['resp'].rasterize()
rec['stim'] = rec['stim'].rasterize()

# keeps only the context identity
CP_dict = col.defaultdict(lambda: col.defaultdict(dict))
for c_p in ['context', 'probe']:
    new_eps = cep.rename_into_part(rec.epochs, context_or_probe=c_p)
    rec['resp'].epochs = new_eps

    # transform epochs into vector over time
    context_epochs = ne.epoch_names_matching(rec.epochs, r'\AC\d$')
    # masked = kep.make_multiepoch_mask(rec, context_epochs)

    # split into train and test sets 80 / 20
    train, test = rec.split_at_time(0.2)

    # extrac each of the epochs, concatenates, makes a vector with the value
    # train

    for subset, this_rec  in zip(['train', 'test'], [train, test]):

        arrays = this_rec['resp'].extract_epochs(context_epochs)

        X = list()
        Y = list()
        for name, array in arrays.items():
            tags = [int(name[-1])] * array.shape[0]
            X.append(array)
            Y.extend(tags)

        X = np.expand_dims(np.concatenate(X), axis=3)
        Y = to_categorical(Y)
        CP_dict[c_p][subset]['x'] = X
        CP_dict[c_p][subset]['y'] = Y

# dimensions of our images.
response_shape = CP_dict['context']['test']['x'].shape

# todo, explore different combinations of metaparameters.
epochs = 20
batch_size = 30
input_shape = (response_shape[1], response_shape[2], 1)

model = Sequential()
model.add(Conv2D(20, (13, 10), input_shape=input_shape, data_format='channels_last'))
model.add(Activation('linear'))

# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 7)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 4)))

model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# fits the model
accuracy_dict = col.defaultdict(lambda: col.defaultdict(dict))
for cont_probe, arr in CP_dict.items():
    # print('######\nfittign for {}\nx shape {}\n'
    #       'y shape {}'.format(cont_probe, arr['train']['x'].shape, arr['train']['y'].shape))

    model.fit(arr['train']['x'], arr['train']['y'], batch_size=batch_size, epochs=epochs, verbose=1)
    # calculates and holds accuracy
    for tt in ['train', 'test']:
        # print('######\nevalutating for {}, {} set\nx shape {}\n'
        #       'y shape {}'.format(cont_probe, tt, arr[tt]['x'].shape, arr[tt]['y'].shape))
        scores = model.evaluate(arr[tt]['x'],arr[tt]['y'], verbose=0)
        accuracy_dict[cont_probe][tt]['real'] = scores
        y_shuffle = arr[tt]['y'].copy()
        np.random.shuffle(y_shuffle)

        shuffle_scores = model.evaluate(arr[tt]['x'],y_shuffle, verbose=0)

        accuracy_dict[cont_probe][tt]['shuffle'] = shuffle_scores

# prints accuracy
for cont_probe, first_inner in accuracy_dict.items():
    for train_test, second_inner in first_inner.items():
        for real_shuffle, accuracy  in second_inner.items():
            print('{}, {} {} accuracy: {:.2f}%'.format(cont_probe, train_test, real_shuffle, accuracy[1]*100))
