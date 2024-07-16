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
    events = Events()
    # events.create_trial_files('small')

    raw_csv = pd.read_csv('../Data/EEG_PL/11.07/EEG/see all_EPOCFLEX_229567_2024.07.11T14.40.32+08.00.md.csv',
                          nrows=1)
    start_time = datetime.fromtimestamp(float(raw_csv.columns[1].split(":")[1])).strftime('%H:%M:%S.%f')
    events.create_start_stop_df_for_events(start_time)
    raw = mne.io.read_raw_edf('../Data/EEG_PL/11.07/EEG/see all_EPOCFLEX_229567_2024.07.11T14.40.32+08.00.edf'
                              , preload=True)
    # get_events('small')

    eg_channels = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9',
                   'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4',
                   'F8', 'Fp2']
    # raw = mne.io.read_raw_edf('../Data/EEG_PL/07.02_EPOCFLEX_228045_2024.07.02T13.54.25+08.00.edf')
    raw.pick(eg_channels)

    raw.plot()
    print(raw)
    print(raw.info)
    #
    # print(raw.info['ch_names'])  # 打印所有通道名称
    # print(raw.get_channel_types())  # 获取并打印所有通道的类型

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    raw.filter(0.1, 30, fir_design='firwin')

    # raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
    # raw.plot(duration=5, n_channels=30)
    #
    # ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    # ica.fit(raw)
    # ica.exclude = [1, 2]  # details on how we picked these are omitted here
    # ica.plot_properties(raw, picks=ica.exclude)
    ...
