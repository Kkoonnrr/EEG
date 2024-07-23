import pandas as pd
import numpy as np
import mne


class Preprocessing:
    def __init__(self, eeg_data, eye_det_data = None):
        self.eeg_data = eeg_data
        self.eye_det_data = eye_det_data

    eventIDs = {"Tree": 1, "Sun": 2, "River": 3}

    bandwidth = {"delta": (0.5, 4),
                 "theta": (4, 8),
                 "alpha": (8, 12),
                 "sigma": (12, 16),
                 "beta": (16, 30),
                 "gamma": (30, 100)}

    EEG_data_labels = {}

    def load_annotations(self):
        pass



