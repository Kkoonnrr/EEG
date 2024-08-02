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

    def kfold_test(self):

        clf = DecisionTreeClassifier(random_state=42)

        k_folds = KFold(n_splits=5)

        scores = cross_val_score(clf, self.X, self.Y, cv=k_folds)

        print("Cross Validation Scores: ", scores)
        print("Average CV Score: ", scores.mean())
        print("Number of CV Scores used in Average: ", len(scores))

    def plot_data(self):
        fig = go.Figure()

        # Add traces for each label
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
                name=get_key_by_value(eventIDs,label)
            ))
        fig.update_layout(
            title='3D Scatter Plot with Three Labels for Seeing',
            scene=dict(
                xaxis_title=self.X.columns[0],
                yaxis_title=self.X.columns[1],
                zaxis_title=self.X.columns[2]
            )
        )

        # Show the plot
        fig.show()

