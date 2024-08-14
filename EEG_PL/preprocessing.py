import pandas as pd
import numpy as np
import mne
from pathlib import Path
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


mpl.use("MacOSX")

eeg_channels = {'Frontal': ['Fz', 'F7', 'F3', 'Fp1', 'F8', 'F4', 'Fp2', 'FC1', 'FC2'],
                'Temporal': ['FT10', 'FT9', 'T7', 'T8', 'FC6', 'FC5'],
                'Parietal': ['Cz', 'C3', 'CP5', 'CP1', 'P3', 'P7', 'Pz', 'P8', 'P4', 'CP2', 'CP6', 'C4', ],
                'Occipital': ['O1', 'Oz', 'O2', 'PO10', 'PO9']}


class Preprocessing:
    def __init__(self, eeg_data=None, eye_det_data=None):
        self.eeg_data = eeg_data
        self.eye_det_data = eye_det_data
        self.sfreq = 128
        self.pca_features = None
        self.best_features = None
        self.classes = None

    eventIDs = {"Tree": 1, "Sun": 2, "River": 3}

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
                labels = [1 for i in range(int(len(raw) / raw.info['sfreq'] / 2))]
            elif 'sun' in file.name:
                labels = [2 for i in range(int(len(raw) / raw.info['sfreq'] / 2))]
            elif 'river' in file.name:
                labels = [3 for i in range(int(len(raw) / raw.info['sfreq'] / 2))]
            glob_labels.extend(labels)
            time_temp = [i * 2 + time for i in range(len(labels))]
            time = max(time_temp) + 2
            glob_time.extend(time_temp)
            glopbal_raw.append(raw)
        event_labels = pd.DataFrame({'Time': glob_time, 'Index': glob_labels})
        combined_raw = mne.concatenate_raws(glopbal_raw)
        return event_labels, combined_raw

    def divide_to_events(self):
        labels, raw_data = self.load_annotations()
        montage = mne.channels.make_standard_montage('standard_1020')
        raw_data.set_montage(montage, on_missing='ignore')
        raw_data.filter(l_freq=0.2, h_freq=None, fir_design='firwin')
        raw_data.filter(l_freq=None, h_freq=40, fir_design='firwin')
        event_times_s = labels.iloc[:, 0]
        event_time_freq = (event_times_s * self.sfreq).astype(int)
        event_type = (labels.iloc[:, 1]).astype(int)

        EEGevents = np.column_stack((event_time_freq, np.zeros_like(event_time_freq), event_type))
        # raw_data.plot(duration = 10.0, events = EEGevents, title = 'EEG with events', event_id = self.eventIDs)

        epochs = mne.Epochs(raw=raw_data, events=EEGevents, event_id=self.eventIDs, tmin=0, tmax=2, baseline=(0, 0))
        # epochs.plot(events=EEGevents, title='EEG with events', event_id=self.eventIDs)

        ica = ICA(n_components=0.95, max_iter='auto', random_state=4)
        ica.fit(epochs)
        # ica.plot_components()
        eog_indices, eog_scores = ica.find_bads_eog(epochs, ch_name='Fz', threshold=1.7)
        ica.exclude = eog_indices
        ica.apply(epochs.load_data())
        # ica.plot_scores(eog_scores)

        epochs.set_eeg_reference('average', projection=True)
        epochs.apply_proj()

        # epochs.plot_image(combine="mean")
        # epochs.average().plot_joint()

        # epochs.plot(events=EEGevents, title='EEG after', event_id=self.eventIDs)
        # epochs['Tree'].plot(events=EEGevents, title='EEG after for trees', event_id=self.eventIDs)

        self.pca_features, self.best_features, self.classes = self.features_extraction('PSD', epochs)
        # self.autocorr()
        return self.pca_features, self.best_features, self.classes

    def features_extraction(self, method, epoch_data):
        scaler_minmax = MinMaxScaler()
        scaler_standard = StandardScaler()
        match method:
            case "PSD":
                features, classes = self.get_psd_features(epoch_data)
            case "Statistical":
                features, classes = self.get_statistical_features(epoch_data)

        features_reduced = SelectKBest(f_classif, k=10)
        features_reduced.fit(features, classes)
        cols = features_reduced.get_support(indices=True)
        features_reduced_best = features.iloc[:, cols]
        features_reduced_best = scaler_minmax.fit_transform(features_reduced_best)

        pca = PCA(n_components=3)
        pca.fit(features, classes)
        features_reduced_pca = pca.transform(features)
        features_reduced_pca = scaler_minmax.fit_transform(features_reduced_pca)

        return features_reduced_pca, features_reduced_best, classes

    def get_psd_features(self, data):

        classes_list = []

        bandwidth = {"delta": (0.5, 4),
                     "theta": (4, 8),
                     "alpha": (8, 12),
                     "sigma": (12, 16),
                     "beta": (16, 30),
                     "gamma": (30, 60)}

        features = []
        for item_class in self.eventIDs.keys():
            item_epochs = data[item_class].get_data(copy = True)
            channel_names = data.ch_names
            for epoch in item_epochs:
                psd_features = []
                for channel_name, channel_data in zip(channel_names, epoch):
                    nperseg = min(len(channel_data), 128)
                    freqs, psd = welch(channel_data, fs=128, nperseg=nperseg, nfft=nperseg)
                    band_powers = []
                    for fmin, fmax in bandwidth.values():
                        idx_min = np.argmax(freqs >= fmin)
                        idx_max = np.argmax(freqs >= fmax)
                        band_powers.append(np.sum(psd[idx_min:idx_max]))
                    psd_features.append(band_powers)
                features.append(np.array(psd_features).flatten())
                classes_list.append(self.eventIDs[item_class])
        column_names = [f'{ch}_{band}' for ch in channel_names for band in bandwidth.keys()]
        features_df = pd.DataFrame(features, columns=column_names)
        return features_df, classes_list

    def get_statistical_features(self, data):
        classes_list = []
        features = []
        stat_list = ['mean', 'median', 'variance', 'std', 'rms', 'skew', 'kurt']
        for item_class in self.eventIDs.keys():
            item_epochs = data[item_class].get_data(copy = True)
            channel_names = data.ch_names
            for epoch in item_epochs:
                channel_stat = []
                for channel_name, channel_data in zip(channel_names, epoch):
                    mean = np.mean(channel_data)
                    median = np.median(channel_data)
                    variance = np.var(channel_data)
                    std = np.std(channel_data)
                    rms = np.sqrt(np.mean(channel_data ** 2))
                    skewness = skew(channel_data)
                    kurtosis_stat = kurtosis(channel_data)
                    channel_stat.append([mean, median, variance, std, rms, skewness, kurtosis_stat])
                features.append(np.array(channel_stat).flatten())
                classes_list.append(self.eventIDs[item_class])
        column_names = [f'{ch}_{stat}' for ch in channel_names for stat in stat_list]
        features_df = pd.DataFrame(features, columns=column_names)
        return features_df, classes_list

    def autocorr(self):
        LE = LabelEncoder()
        df_encoded = pd.DataFrame(self.best_features)
        df_encoded['class'] = LE.fit_transform(self.classes)
        hm = sns.heatmap(df_encoded.corr(numeric_only=True))
        plt.show()
