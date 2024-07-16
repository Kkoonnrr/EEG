import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


class Events:
    def __init__(self):
        self.directories = None
        self.experiment_data = {}

    def create_trial_files(self, path: str):
        base_path = Path('../Data/EEG_PL/11.07') / path
        directories = [d for d in base_path.iterdir() if d.is_dir()]
        self.directories = directories
        for directory in directories:
            gaze_files = [f for f in directory.iterdir() if 'gaze' in f.name]
            if not gaze_files:
                continue  # Skip if no gaze files found

            gaze_file = gaze_files[0]  # Assuming only one gaze file per directory
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
        columns = ["Time Of Experiment", "Duration", "TimeStampStart", "TimeStampEnd"]
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
                "Time Of Experiment": start_time,
                "Duration": f'{duration:.2f}',
                "TimeStampStart": delta_start.strftime("%H:%M:%S.%f")[:-3],
                "TimeStampEnd": delta_end.strftime("%H:%M:%S.%f")[:-3]
            })

        return events

    def load_experiment_data(self):
        experiment_data = {}
        for directory in self.directories:
            input_file = directory / "Good_Experiments.csv"
            experiment_data[directory] = pd.read_csv(input_file)
        return experiment_data

    def create_start_stop_df_for_events(self, start_time: str):
        if not self.experiment_data:
            self.experiment_data = self.load_experiment_data()


