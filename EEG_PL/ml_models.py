import pandas as pd
import numpy as np
import mne
from pathlib import Path
import matplotlib as mpl
from scipy.stats import kurtosis, skew
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier
import optuna
from optuna_dashboard import run_server
from sklearn.svm import SVC
import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution1D, Dropout
from keras.optimizers import SGD
from keras.initializers import random_uniform

mpl.use("MacOSX")
eventIDs = {"Tree": 1, "Sun": 2, "River": 3}


def get_key_by_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None


class MlModels:
    def __init__(self, X, Y):
        self.X = pd.DataFrame(X)
        self.Y = Y

    def mlp(self, trial):
        n_layers = trial.suggest_int('n_layers', 1, 4)
        activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
        solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])
        alpha = trial.suggest_float('alpha', 1e-4, 1e-2)
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-1)
        warm_start = trial.suggest_categorical('warm_start', [False, True])

        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_{i}', 1, 100))

        mlp = MLPClassifier(hidden_layer_sizes=tuple(layers),
                            activation=activation,
                            solver=solver,
                            alpha=alpha,
                            learning_rate=learning_rate,
                            learning_rate_init=learning_rate_init,
                            warm_start=warm_start,
                            max_iter=2000,
                            random_state=42)
        # k_folds = KFold(n_splits=5, shuffle=True)
        # scores = cross_val_score(mlp, self.X, self.Y, cv=k_folds)
        return mlp

    def cnn(self):
        # model = Sequential()
        # model.add(Convolution1D(nb_filter=32, filter_length=3, input_shape=X_train.shape[1:3], activation='relu'))
        # model.add(Convolution1D(nb_filter=16, filter_length=1, activation='relu'))
        # model.add(Flatten())
        # model.add(Dropout(dropout_rate))
        # model.add(Dense(128, input_dim=input_dimension, kernel_initializer=hidden_initializer, activation='relu'))
        # model.add(Dropout(dropout_rate))
        # model.add(Dense(64, kernel_initializer=hidden_initializer, activation='relu'))
        # model.add(Dense(2, kernel_initializer=hidden_initializer, activation='softmax'))
        pass

    def svc(self, trial):
        c=trial.suggest_float('C', 1e-1, 2)
        kernel=trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        gamma=trial.suggest_categorical('gamma', ['scale', 'auto'])
        svc = SVC(C=c,
                  gamma=gamma,
                  kernel=kernel)
        return svc


    def xgboost(self, trial):
        # model = Sequential()
        # model.add(LSTM(128, input_shape=(18,)))
        # model.add(Dense(len(self.X), activation='softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer='adam')
        # return model
        pass

    def cross_validation(self, trial):
        k_folds = KFold(n_splits=5, shuffle=True)
        scores = []
        mlp = self.mlp(trial)
        # svc = self.svc(trial)
        for train_index, val_index in k_folds.split(self.X):
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            Y_train, Y_val = self.Y.iloc[train_index], self.Y.iloc[val_index]

            mlp = mlp.fit(X_train, np.ravel(Y_train))
            scores.append(mlp.score(X_val, np.ravel(Y_val)))

            # svc = svc.fit(X_train, np.ravel(Y_train))
            # scores.append(svc.score(X_val, np.ravel(Y_val)))


        # print(np.mean(scores))
        return np.mean(scores)

    def hipertuning(self):
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(storage=storage, study_name="dashboard-example", direction="maximize")
        study.optimize(self.cross_validation, n_trials=100)
        run_server(storage, host="localhost", port=8080)
        print(study.best_value)

    def plot_data(self):
        fig = go.Figure()
        labels = set(self.Y)
        colors = ['green', 'yellow', 'blue']
        self.X['label'] = self.Y
        for label, color in zip(labels, colors):
            filtered_df = self.X[self.X['label'] == label]
            fig.add_trace(go.Scatter3d(
                x=filtered_df[self.X.keys()[0]],
                y=filtered_df[self.X.keys()[1]],
                z=filtered_df[self.X.keys()[2]],
                mode='markers',
                marker=dict(size=5, color=color),
                name=get_key_by_value(eventIDs, label)
            ))
        fig.update_layout(
            title='3D Scatter Plot with Three Labels for Seeing',
            scene=dict(
                xaxis_title=self.X.columns[0],
                yaxis_title=self.X.columns[1],
                zaxis_title=self.X.columns[2]
            )
        )
        fig.show()
