#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train standard classification models on the botnet detection task with the TI-2016 dataset.
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


class _DebugProperties:
    DAYS_SUBSET = ["Day0"]


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_path", help="Path to the TI-2016 dataset.")
    argparser.add_argument("-d", "--debug", action="store_true", help="Debug mode.")
    args = argparser.parse_args()

    # Find DayX folders in the data path
    day_folders = [
        folder for folder in os.listdir(args.data_path) if re.match("Day\d+", folder)
    ]
    day_folders.sort()
    if args.debug:
        day_folders = [
            day for day in day_folders if day in _DebugProperties.DAYS_SUBSET
        ]

    data = None

    print("Loading data...")
    for day_folder in tqdm(day_folders):
        # Find labeled csvs in the folder
        filenames = os.listdir(os.path.join(args.data_path, day_folder))
        labeled_csvs = [
            filename
            for filename in filenames
            if re.match("^\d+_\d+.pcap_Flow_labeled.csv$", filename)
        ]
        labeled_csvs.sort()

        for csv_file in tqdm(labeled_csvs):
            # Load the csv
            csv = pd.read_csv(os.path.join(args.data_path, day_folder, csv_file))

            # Remove rows with N/A label (mostly non-users IP sources)
            csv = csv[~pd.isna(csv["bot_family"])]

            # Iteratively concatenate all csvs together
            if data is None:
                data = csv
            data = pd.concat((data, csv))

    print(data)

    CLASSES = [
        "Clean",
        "Unknown",
        "modpack",
        "virut",
        "necurs",
        "conficker",
        "ud3",
        "suppobox",
        "nymaim",
        "tofsee",
        "pitou",
    ]
    for c in CLASSES:
        data[c] = (data["bot_family"] == c).astype(int)

    # Remove non-numeric features
    data = data.select_dtypes(include=["number", "datetime"])

    # Normalize data
    minmax_scaler = MinMaxScaler((-1, 1))
    data_scaled = minmax_scaler.fit_transform(data)

    # Split into train and test set
    train_test_split = 0.8
    X_train = data_scaled[: int(train_test_split * len(data_scaled)), : -len(CLASSES)]
    y_train = data_scaled[: int(train_test_split * len(data_scaled)), -len(CLASSES) :]
    _, y_train = np.where(y_train > 0)  # convert one-hot encoding with class index
    X_test = data_scaled[int(train_test_split * len(data_scaled)) :, : -len(CLASSES)]
    y_test = data_scaled[int(train_test_split * len(data_scaled)) :, -len(CLASSES) :]
    _, y_test = np.where(y_test > 0)  # convert one-hot encoding with class index

    print("Starting training with shapes:")
    print(f"> X_train: {X_train.shape}")
    print(f"> y_train: {y_train.shape}")
    print(f"> X_test: {X_test.shape}")
    print(f"> y_test: {y_test.shape}")

    # Define models to use
    Models = {
        RandomForestClassifier: {"conf": {"oneclass": False}},
        # SVC: {"conf": {"oneclass": False}, "kwargs": {"probability": True}}, # takes too long
        AdaBoostClassifier: {"conf": {"oneclass": False}},
        GaussianNB: {"conf": {"oneclass": False}},
        DecisionTreeClassifier: {"conf": {"oneclass": False}},
    }

    # Train, predict and get metrics for each model
    for Model in Models:
        print(f"\nEvaluating {Model.__name__}...")
        model = ModelWrapper(
            Model,
            Models[Model].get("conf"),
            *Models[Model].get("args", []),
            **Models[Model].get("kwargs", {}),
        )

        model.fit(X=X_train, y=y_train)
        y_pred, probs = model.predict(X_test)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))


class ModelWrapper:
    def __init__(self, Model, conf, *args, **kwargs):
        super(ModelWrapper, self).__init__()
        self.Model = Model
        self.conf = conf
        self.args = args
        self.kwargs = kwargs
        self.model = Model(*args, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        pred = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return pred, probs


if __name__ == "__main__":
    main()
