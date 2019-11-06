import nems.signal as Signal
import numpy as np

def make_multiepoch_mask(recording, epochs_list):
    '''
    takes a list of epochs and transforms them into a boolean signal indicating what time bins correspond to the
    specified epochs.
    :param recording:
    :param epochs_list:
    :return:
    '''

    raster = recording['resp'].rasterize()
    time_len = raster._data.shape[1]
    all_masks = np.empty([len(epochs_list), time_len])

    for ee, epoch in enumerate(epochs_list):
        all_masks[ee, :] = recording.create_mask(epoch)['mask']._data

    # all_masks = recording['resp']._modified_copy(data=all_masks, name='all_masks')
    all_masks = raster._modified_copy(data=all_masks, name='all_masks', chans=epochs_list)
    recording.signals['all_masks'] = all_masks

    return recording