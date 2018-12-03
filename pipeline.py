import joblib as jl
import matplotlib.pyplot as plt
import numpy as np

import cpp_dispersion as cdisp
import cpp_epochs as cpe
import cpp_plots as cplt
import nems.recording as recording
import itertools as itt

Test = True

if Test is True:
    # # sets automatic path finding
    # this_script_dir = os.path.dirname(os.path.realpath(__file__))
    # pickle_path = '{}/pickles'.format(this_script_dir)
    # test_rec_path = os.path.normcase('{}/BRT037b'.format(pickle_path))
    test_rec_path = '/home/mateo/context_probe_analysis/pickles/BRT037b'
    loaded_rec = jl.load(test_rec_path)

else:
    # gives the uri to a cached recording. if the cached data does not exists, creates it and saves it.
    import nems_db.baphy as nb
    options = {'batch': 310,
               'site': 'BRT037b',
               'rasterfs': 100}
    load_URI = nb.baphy_load_multichannel_recording(**options)
    loaded_rec = recording.load_recording(load_URI)

# sets eps correpsondign to individual sounds as well as context probe pairs
rec = cpe.set_recording_subepochs(loaded_rec, set_pairs=True)
sig = rec['resp']
eps = sig.epochs

# transforms the recording into its PCA equivalent


