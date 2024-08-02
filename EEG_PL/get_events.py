import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import mne
import numpy as np
import shutil

eeg_channels = ['Fz', 'F7', 'F3', 'Fp1', 'F8', 'F4', 'Fp2', 'FC1', 'FC2',
                'FT10', 'FT9', 'T7', 'T8', 'FC6', 'FC5', 'Cz', 'C3', 'CP5',
                'CP1', 'P3', 'P7', 'Pz', 'P8', 'P4', 'CP2', 'CP6', 'C4',
                'O1', 'Oz', 'O2', 'PO10', 'PO9']


def to_date(date: str):
    return datetime.strptime(date, '%H:%M:%S.%f')


def clear_folder(folder_path):
    folder = Path(folder_path)
    for item in folder.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()  # Remove the file or link
            elif item.is_dir():
                shutil.rmtree(item)  # Remove the directory and its contents
        except Exception as e:
            print(f'Failed to delete {item}. Reason: {e}')


class Events:
    def __init__(self, dir_name: str, eeg_edf: str, eeg_csv):
        self.dir_name = dir_name
        self.eeg_csv = eeg_csv
        self.eeg_edf = eeg_edf
        self.directories = None
        self.experiment_data = {}

    def create_trial_files(self):
        base_path = Path('../Data/EEG_PL/11.07') / self.dir_name
        directories = [d for d in base_path.iterdir() if d.is_dir()]
        self.directories = directories
        for directory in directories:
            gaze_files = [f for f in directory.iterdir() if 'gaze' in f.name]
            if not gaze_files:
                continue
            gaze_file = gaze_files[0]
            self.process_gaze_file(gaze_file, directory)

    def process_gaze_file(self, gaze_file: Path, directory: Path):
        try:
            gaze_data = pd.read_csv(gaze_file)
        except Exception as e:
            print(f"Error reading {gaze_file}: {e}")
            return
        click_data = gaze_data[gaze_data['KBS'] == 1]
        events = self.extract_events(click_data)
        if not events.empty:
            output_file = directory / "Good_Experiments.csv"
            events.to_csv(output_file, index=False)
            self.experiment_data[directory] = events
            print(f"Events saved to {output_file}")
        else:
            print(f"No valid events found in {gaze_file}")

    def extract_events(self, click_data: pd.DataFrame) -> pd.DataFrame:
        columns = ["Relative Time", "Time Of Experiment", "Duration", "TimeStampStart", "TimeStampEnd"]
        events = pd.DataFrame(columns=columns)
        for index in range(len(click_data) - 1):
            start_time = click_data.iloc[index, 1]
            duration = click_data.iloc[index + 1, 1] - start_time
            if not (3.9 <= duration <= 4.1):
                continue
            try:
                start_timestamp = datetime.strptime(click_data.columns[1][16:-1], "%H:%M:%S.%f")
            except ValueError as e:
                print(f"Error parsing timestamp: {e}")
                continue

            delta_start = start_timestamp + timedelta(seconds=start_time)
            delta_end = delta_start + timedelta(seconds=duration)
            events.loc[index] = ({
                "Relative Time": 0,
                "Time Of Experiment": start_time,
                "Duration": f'{duration:.2f}',
                "TimeStampStart": delta_start.strftime("%H:%M:%S.%f")[:-3],
                "TimeStampEnd": delta_end.strftime("%H:%M:%S.%f")[:-3]
            })
        events["Relative Time"] = events["Time Of Experiment"] - events["Time Of Experiment"].iloc[0]
        return events

    # TODO
    def load_experiment_data(self):
        experiment_data = {}
        for directory in self.directories:
            input_file = directory / "Good_Experiments.csv"
            experiment_data[directory] = pd.read_csv(input_file)
        return experiment_data

    def create_start_stop_df_for_events(self, start_time: str):
        if not self.experiment_data:
            self.experiment_data = self.load_experiment_data()  # not working
        event_col = ['directory',
                     'start_of_the_event(EEG data sec)',
                     'end_of_the_event(EEG data sec)',
                     'duration_of_the_event']
        event_duration = pd.DataFrame(columns=event_col)
        for index, data in enumerate(self.experiment_data.items()):
            name, ex_data = data
            event_duration.loc[index] = [name.name,
                                         (to_date(ex_data.iloc[0]["TimeStampStart"]) -
                                          to_date(start_time)).seconds,
                                         (to_date(ex_data.iloc[-1]["TimeStampEnd"]) -
                                          to_date(start_time)).seconds,
                                         (to_date(ex_data.iloc[-1]["TimeStampEnd"]) -
                                          to_date(ex_data.iloc[0]["TimeStampStart"])).seconds]
        event_duration.to_csv(f"../Data/EEG_PL/11.07/{self.dir_name}/Event_time_count_by_EGG_data.csv")
        return event_duration

    def create_eeg_events_files(self):
        clear_folder('../Data/EEG_PL/11.07/EEG/EEG_Event_data')
        raw_csv = pd.read_csv(self.eeg_csv, nrows=1)
        start_time = datetime.fromtimestamp(float(raw_csv.columns[1].split(":")[1])).strftime('%H:%M:%S.%f')[:-3]
        raw = mne.io.read_raw_edf(self.eeg_edf, preload=True)
        raw.pick(eeg_channels)
        events_duration = self.create_start_stop_df_for_events(start_time)
        all_data = []
        for event_duration in events_duration.iterrows():
            directory_name = event_duration[1]['directory']
            start_time = event_duration[1]['start_of_the_event(EEG data sec)']
            end_time = event_duration[1]['end_of_the_event(EEG data sec)']
            raw_segment = raw.copy().crop(tmin=start_time, tmax=end_time)
            raw_segment.save(f'../Data/EEG_PL/11.07/EEG/EEG_Event_data/{directory_name}_with_blank_raw.fif',
                             overwrite=True)

            for index, data in enumerate(self.experiment_data.items()):
                name, ex_data = data
                sfreq = raw_segment.info['sfreq']
                segment_duration = 2
                segment_samples = int(segment_duration * sfreq)
                relative_times = ex_data['Relative Time']
                segments = []
                for times in relative_times:
                    start_sample = int((times + 2) * sfreq)
                    end_sample = start_sample + segment_samples
                    if end_sample <= raw_segment.n_times:
                        segment_data, _ = raw_segment[:, start_sample:end_sample]
                        segments.append(segment_data)

                # trigger_codes = relative_times[:,0]
                sfreq = raw_segment.info['sfreq']
                condition_labels = [1 for i in range(len(relative_times))]
                duration = [2 for i in range(len(relative_times))]
                trigger_samples = (relative_times * sfreq).astype(int)
                EEG_events = np.column_stack((trigger_samples, np.zeros_like(trigger_samples), condition_labels))
                eventIDs = {"Tree": 1, "Sun": 2, "River": 3}
                raw_segment.plot(duration=10.0, events=EEG_events, title=f"{directory_name}", event_id=eventIDs)
                # mne.Epochs(raw_segment, EEG_events, eventIDs, 0.0, 2.0, 0.0)

                # add all data together, events on time + time of ending last event

                combined_segments = np.concatenate(segments, axis=1)
                info = raw_segment.info.copy()
                new_raw = mne.io.RawArray(combined_segments, info)
                new_raw.save(f'../Data/EEG_PL/11.07/EEG/EEG_Event_data/{directory_name}_without_blank_raw.fif',
                             overwrite=True)
