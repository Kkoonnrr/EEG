import pandas as pd
import numpy as np

import mne

class Preprocessing:
    def __init__(self, eeg_data, eye_det_data = None):
        self.eeg_data = eeg_data
        self.eye_det_data = eye_det_data

    eeg_channels = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9',
                    'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4',
                    'F8', 'Fp2']

    bandwidth = {"delta": (0.5, 4),
                 "theta": (4, 8),
                 "alpha": (8, 12),
                 "sigma": (12, 16),
                 "beta": (16, 30),
                 "gamma": (30, 100)}

    EEG_data_labels = {}
    def csv_to_grouped_dic(self):
        ungrouped_dict = self.eeg_data.iloc[:, :36].to_dict()
        grouped_dict = 1

    def mne_processing(self):
        raw = mne.io.read_raw_edf('07.02_EPOCFLEX_228045_2024.07.02T13.54.25+08.00.edf')
        ...

