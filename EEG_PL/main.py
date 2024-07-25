from preprocessing import Preprocessing
import numpy as np
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta
from scipy.signal import welch
from mne.time_frequency import psd_array_multitaper
from scipy.integrate import simps
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet
from get_events import Events

mpl.use("MacOSX")

if __name__ == "__main__":
    events = Events('small',
                    '../Data/EEG_PL/11.07/EEG/see all_EPOCFLEX_229567_2024.07.11T14.40.32+08.00.edf',
                    '../Data/EEG_PL/11.07/EEG/see all_EPOCFLEX_229567_2024.07.11T14.40.32+08.00.md.csv')
    # events.create_trial_files()
    # events.create_eeg_events_files()
    preprocessing = Preprocessing()
    preprocessing.divide_to_events()





    names = ['see_tree_without_blank']
    # events_duration = events.create_start_stop_df_for_events()
    for name in names:

        raw = mne.io.read_raw_fif(f'../Data/EEG_PL/11.07/EEG/EEG_Event_data/{name}_raw.fif', preload=True)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')

        raw.filter(0.1, 40, fir_design='firwin')

        raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
        raw.plot(duration=5, n_channels=30)
        #
        ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
        ica.fit(raw)
        ica.plot_components()
        ica.exclude = [1]
        raw_ica = ica.apply(raw)
        raw_ica.plot()
        ...
        # ica.plot_properties(raw, picks=ica.exclude)
