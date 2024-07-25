import pandas as pd
import numpy as np
import mne
from pathlib import Path
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet


class Preprocessing:
    def __init__(self, eeg_data = None, eye_det_data = None):
        self.eeg_data = eeg_data
        self.eye_det_data = eye_det_data
        self.sfreq = 128

    eventIDs = {"Tree": 1, "Sun": 2, "River": 3}

    bandwidth = {"delta": (0.5, 4),
                 "theta": (4, 8),
                 "alpha": (8, 12),
                 "sigma": (12, 16),
                 "beta": (16, 30),
                 "gamma": (30, 100)}

    EEG_data_labels = {}

    def load_annotations(self):
        global labels
        base_path = Path('../Data/EEG_PL/11.07/EEG/EEG_Event_data')
        files = [d for d in base_path.iterdir() if 'without' in d.name]
        event_labels = pd.DataFrame(columns=[['Index', 'Time']])
        glob_labels = []
        glob_time = []
        glopbal_raw = []
        time = 0
        for file in files:
            raw = mne.io.read_raw_fif(file, preload=True)
            if 'tree' in file.name:
                labels = [1 for i in range(int(len(raw)/raw.info['sfreq']/2))]
            elif 'sun' in file.name:
                labels = [2 for i in range(int(len(raw) / raw.info['sfreq'] / 2))]
            elif 'river' in file.name:
                labels = [3 for i in range(int(len(raw) / raw.info['sfreq'] / 2))]
            glob_labels.extend(labels)
            time_temp = [i*2+time for i in range(len(labels))]
            time = max(time_temp)+2
            glob_time.extend(time_temp)
            glopbal_raw.append(raw)
        event_labels = pd.DataFrame({'Time': glob_time, 'Index': glob_labels})
        combined_raw = mne.concatenate_raws(glopbal_raw)
        return event_labels, combined_raw

    def divide_to_events(self):
        labels, raw_data = self.load_annotations()
        montage = mne.channels.make_standard_montage('standard_1020')
        raw_data.set_montage(montage, on_missing='ignore')
        raw_data.filter(0.1, 40, fir_design='firwin')
        event_times_s = labels.iloc[:, 0]
        event_time_freq = (event_times_s * self.sfreq).astype(int)
        event_type = (labels.iloc[:, 1]).astype(int)

        EEGevents = np.column_stack((event_time_freq, np.zeros_like(event_time_freq), event_type))
        #raw_data.plot(duration = 10.0, events = EEGevents, title = 'EEG with events', event_id = self.eventIDs)

        epochs = mne.Epochs(raw_data, EEGevents, self.eventIDs, -.1, 2, (-.1,0),reject = None)
        # epochs.plot(events = EEGevents, title = 'EEG with events', event_id = self.eventIDs)
        # con1 = epochs['River']
        # con2 = epochs['Sun']
        # con3 = epochs['Tree']
        #
        # avg_con1 = con1.average()
        # avg_con2 = con2.average()
        # avg_con3 = con3.average()
        #
        # avg_con1.plot(time_unit='ms')
        # avg_con2.plot(time_unit='ms')
        # avg_con3.plot(time_unit='ms')

        ica = ICA(n_components=20, max_iter='auto', random_state=5)
        ica.fit(epochs)
        # ica.plot_components()
        ica.apply(epochs.load_data())

        epochs.set_eeg_reference('average', projection=True)
        epochs.apply_proj()

        epochs.plot(title='EEG with events', event_id=self.eventIDs)
        # con1 = epochs['River']
        # con2 = epochs['Sun']
        # con3 = epochs['Tree']
        #
        # avg_con1 = con1.average()
        # avg_con2 = con2.average()
        # avg_con3 = con3.average()
        #
        # avg_con1.plot(time_unit='ms')
        # avg_con2.plot(time_unit='ms')
        # avg_con3.plot(time_unit='ms')

        ...










