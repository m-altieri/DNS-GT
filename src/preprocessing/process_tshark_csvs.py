# -*- coding: utf-8 -*-
"""Preprocessing PCAP files.

Author: Massimiliano Altieri <massimiliano.altieri@ec.europa.eu>
Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import os
from pathlib import Path
import pandas as pd

PATH = Path("/mnt/storage15/TI-2016/csv")
OUTPUT_FOLDER = "clean_csvs"

for csv in os.listdir(PATH):
    print("Analyzing", csv)
    df = pd.read_csv(os.path.join(PATH, csv), delimiter=";")

    ips = df["ip.src"].unique()
    requests, responses, is_resolver = [], [], []

    # calculate, for each unique ip within the current hour, its number of requests,
    # responses, and whether it is a resolver
    for ip in ips:
        requests.append(len(df[(df["ip.src"] == ip) & (df["udp.dstport"] == 53)]))
        responses.append(len(df[(df["ip.dst"] == ip) & (df["udp.srcport"] == 53)]))
        is_resolver.append(
            not df[(df["ip.src"] == ip) & (df["udp.srcport"] == 53)].empty
            or not df[(df["ip.dst"] == ip) & (df["udp.dstport"] == 53)].empty
        )
    ip_info = pd.DataFrame(
        {
            "ip": ips,
            "requests": requests,
            "responses": responses,
            "is_resolver": is_resolver,
        }
    )
    ip_info.to_csv("ip_info.csv")

    # good_ips are ips having at least 100 requests for the current hour,
    # having a ratio of requests to responses between 0.985 and 1.015,
    # and that are not dns resolvers (never sent a response and never received a query)
    good_ips = ip_info[
        (ip_info["requests"] > 100)
        & (ip_info["requests"] / ip_info["responses"] > 0.985)
        & (ip_info["requests"] / ip_info["responses"] < 1.015)
        & (ip_info["is_resolver"] == False)
    ]
    good_ips.to_csv("good_ips.csv")

    # select from the csv file only the requests, sent by an ip in good_ips, of type A,
    # not retransmission, [TODO having a response with the same id within 60 seconds]
    print(len(df))
    df = df[
        (df["ip.src"].isin(good_ips["ip"]))
        & (df["udp.dstport"] == 53)
        & (pd.isna(df["dns.retransmission"]))
        & (df["dns.qry.type"] == 1)
    ]
    print(len(df))
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    df.to_csv(os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(csv)[0]}-clean.csv"))
