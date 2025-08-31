from preprocessing import Preprocessing
import matplotlib as mpl
from get_events import Events
from ml_models import MlModels

mpl.use("MacOSX")
if __name__ == "__main__":
    events = Events('small',
                    '../Data/EEG_PL/11.07/EEG/see all_EPOCFLEX_229567_2024.07.11T14.40.32+08.00.edf',
                    '../Data/EEG_PL/11.07/EEG/see all_EPOCFLEX_229567_2024.07.11T14.40.32+08.00.md.csv')
    events.create_trial_files()
    events.create_eeg_events_files()
    preprocessing = Preprocessing()
    pca, best, labels = preprocessing.divide_to_events()
    mlmodels = MlModels(best, labels)
    mlmodels.hipertuning()
    mlmodels.plot_data()