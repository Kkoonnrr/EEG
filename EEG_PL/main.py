from preprocessing import Preprocessing
import numpy as np
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('MacOSX')
import pandas as pd
from datetime import datetime, timedelta
from scipy.signal import welch
from mne.time_frequency import psd_array_multitaper
from scipy.integrate import simps
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet

if __name__ == "__main__":

    raw = mne.io.read_raw_edf('../Data/EEG_PL/11.07/EEG/see all_EPOCFLEX_229567_2024.07.11T14.40.32+08.00.edf')

    gaze = pd.read_csv('../Data/EEG_PL/11.07/small/see_tree/User 13_all_gaze.csv')
    click = gaze[gaze['KBS'] == 1]
    col = ["Number", "Duration", "TimeStampStart", "TimeStampEnd"]
    events = pd.DataFrame(columns = col)
    for index, data in enumerate(click):
        events.add([f'{index}', f'{click.iloc[:,1].diff()[index+1]}',
                    f'{datetime.strptime(click.columns[1][16:-1], "%H:%M:%S.%f")+click.iloc[:,1][index]}', 1])
    ...
    # eg_channels = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9',
    #                'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4',
    #                'F8', 'Fp2']
    # # eeg_data = pd.read_csv("../Data/EEG_PL/07.02.csv", skiprows=1)
    # # preprocessing = Preprocessing(eeg_data)
    # # preprocessing.csv_to_grouped_dic()
    # raw = mne.io.read_raw_edf('../Data/EEG_PL/07.02_EPOCFLEX_228045_2024.07.02T13.54.25+08.00.edf')
    # raw.pick(eg_channels)
    # raw.plot()
    # print(raw)
    # print(raw.info)
    # #
    # print(raw.info['ch_names'])  # 打印所有通道名称
    # print(raw.get_channel_types())  # 获取并打印所有通道的类型
    #
    # raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
    # raw.plot(duration=5, n_channels=30)
    #
    # ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    # ica.fit(raw)
    # ica.exclude = [1, 2]  # details on how we picked these are omitted here
    # ica.plot_properties(raw, picks=ica.exclude)
