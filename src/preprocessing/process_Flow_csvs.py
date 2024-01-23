#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocessing and labeling of CSV files after conversion with CICFlowMeter.

The data_path argument must point to a folder with the following structure:

data_path/
├── ...
├── labeled_features.csv
├── Day0/
    ├── <something>.pcap_Flow.csv
    ├── ...
    ├── <something>.pcap_Flow.csv
    └── ...
├── Day1/
    ├── <something>.pcap_Flow.csv
    ├── ...
    ├── <something>.pcap_Flow.csv
    └── ...
├── ...
├── Day9/
    ├── <something>.pcap_Flow.csv
    ├── ...
    ├── <something>.pcap_Flow.csv
    └── ...
└── ...

The output will be saved as:

data_path/
├── Day0/
    ├── <something>.pcap_Flow_labeled.csv
    ├── ...
    └── <something>.pcap_Flow_labeled.csv
├── Day1/
    ├── <something>.pcap_Flow_labeled.csv
    ├── ...
    └── <something>.pcap_Flow_labeled.csv
├── ...
└── Day9/
    ├── <something>.pcap_Flow_labeled.csv
    ├── ...
    └── <something>.pcap_Flow_labeled.csv

Author: Massimiliano Altieri <massimiliano.altieri@ec.europa.eu>
Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# -- Argument Parsing
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "data_path",
    help="Path to the TI-2016 dataset folder. Only _Flow.csv files will be considered.",
)
args = argparser.parse_args()

# -- Get host labels (for botnet belonging)
labels = pd.read_csv(os.path.join(args.data_path, "labeled_features.csv"))
labels = labels[["Hostname_MMDDHH", "bot_family"]]

# -- Split hostname and datetime into two columns. Datetime will be used to match with the processed csv files
labels[["Hostname", "MMDDHH"]] = labels["Hostname_MMDDHH"].str.split("_", expand=True)

# -- Get (sorted) folder names matching "Day<number>"
day_folders = [
    folder for folder in os.listdir(args.data_path) if re.match("^Day\d+$", folder)
]
day_folders.sort()

# -- For each Day<number> folder
for day_folder in tqdm(day_folders):
    # -- Get the (sorted) _Flow.csv files
    filenames = os.listdir(os.path.join(args.data_path, day_folder))
    flow_csvs = [
        filename
        for filename in filenames
        if re.match("^\d+_\d+.pcap_Flow.csv$", filename)
    ]
    flow_csvs.sort()

    # -- For each Flow csv
    for csv in tqdm(flow_csvs):
        # -- Extract the date and time as a 6-number string (e.g. 042409)
        csv_date = re.search("(?<=^\d{4})(\d{4})_(\d{2})", csv)
        csv_date = csv_date.group(1) + csv_date.group(2)

        # -- Load it as a Pandas DataFrame
        df = pd.read_csv(os.path.join(args.data_path, day_folder, csv))

        # -- Replace infs and nans with 0 TODO should the rows be removed instead?
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # -- Join flow csv with the correct labels
        df_src = df[df["Dst Port"] == 53].merge(
            labels[labels["MMDDHH"] == csv_date],
            how="left",
            left_on="Src IP",
            right_on="Hostname",
            validate="m:1",
        )
        df_dst = df[df["Src Port"] == 53].merge(
            labels[labels["MMDDHH"] == csv_date],
            how="left",
            left_on="Dst IP",
            right_on="Hostname",
            validate="m:1",
        )
        df = pd.concat([df_src, df_dst])

        # -- Remove columns from labeled_features.csv that are not needed
        df = df.drop(["Label", "Hostname_MMDDHH", "Hostname", "MMDDHH"], axis="columns")

        # -- Replace NaN labels (hosts not present in labeled_features.csv) with "N/A"
        df["bot_family"] = df["bot_family"].fillna("N/A")

        # -- Sort rows chronologically. WARNING: rows in the input flow csvs
        # are not sorted exacty. This is probably becase timestamp refers to
        # the start of the flow but they are sorted by the end time or vice versa.
        # The final ordering after preprocessing will not reflect the ordering
        # of the input flow csvs.
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"], format="%d/%m/%Y %I:%M:%S %p"
        )  # fix 12-hour format bug
        df = df.sort_values("Timestamp", axis="index")

        df.to_csv(
            os.path.join(
                args.data_path, day_folder, f"{os.path.splitext(csv)[0]}_labeled.csv"
            )
        )
