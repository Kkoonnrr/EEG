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

def get_events(path : str):
    gazes = []
    directories = os.listdir('../Data/EEG_PL/11.07/'+path)
    for directory in directories:
        files = os.listdir('/'.join(['../Data/EEG_PL/11.07',path,directory]))
        gaze = {directory:i for i in files if 'gaze' in i}
        gazes.append(gaze)
        gaze_data = pd.read_csv('/'.join(['../Data/EEG_PL/11.07',path,directory,gaze.get(f'{directory}')]))
        click = gaze_data[gaze_data['KBS'] == 1]
        col = ["Time Of Experiment", "Duration", "TimeStampStart", "TimeStampEnd"]
        events = pd.DataFrame(columns=col)
        for index, data in enumerate(click):
            try:
                duration = click.iloc[:, 1].diff().iloc[index + 1]
            except IndexError:
                continue
            if 4.1 < duration or duration < 3.9:
                continue
            else:
                start_time = datetime.strptime(click.columns[1][16:-1], "%H:%M:%S.%f")
                delta_time = start_time + timedelta(0, click.iloc[:, 1].iloc[index])
                end_time = delta_time + timedelta(0, click.iloc[:, 1].diff().iloc[index + 1])
                events.loc[index] = ([
                    click.iloc[index, 1],
                    f'{click.iloc[:, 1].diff().iloc[index + 1]:.2f}',
                    f'{delta_time.hour}:{delta_time.minute}:{delta_time.second}:{int(delta_time.microsecond)}'[:-3],
                    f'{end_time.hour}:{end_time.minute}:{end_time.second}:{end_time.microsecond}'[:-3]
                ])
        events.to_csv('/'.join(['../Data/EEG_PL/11.07',path,directory, "Good_Experiments.csv"]))


if __name__ == "__main__":

    raw = mne.io.read_raw_edf('../Data/EEG_PL/11.07/EEG/see all_EPOCFLEX_229567_2024.07.11T14.40.32+08.00.edf')
    get_events('small')



    eg_channels = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9',
                   'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4',
                   'F8', 'Fp2']
    # raw = mne.io.read_raw_edf('../Data/EEG_PL/07.02_EPOCFLEX_228045_2024.07.02T13.54.25+08.00.edf')
    raw.pick(eg_channels)
    raw.plot()
    print(raw)
    print(raw.info)
    #
    print(raw.info['ch_names'])  # 打印所有通道名称
    print(raw.get_channel_types())  # 获取并打印所有通道的类型

    # raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
    # raw.plot(duration=5, n_channels=30)
    #
    # ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    # ica.fit(raw)
    # ica.exclude = [1, 2]  # details on how we picked these are omitted here
    # ica.plot_properties(raw, picks=ica.exclude)
    ...
