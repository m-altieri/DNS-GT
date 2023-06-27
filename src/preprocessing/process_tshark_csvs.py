import os
import sys
import numpy as np
import pandas as pd

PATH = "/mnt/storage15/TI-2016/csv"

for csv in os.listdir(PATH):
    print("Analyzing", csv)
    df = pd.read_csv("/mnt/storage15/TI-2016/csv/20160501_051543.csv", delimiter=";")

    ips = df["ip.src"].unique()
    requests, responses, is_resolver = [], [], []

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

    good_ips = ip_info[
        (ip_info["requests"] > 100)
        & (ip_info["requests"] / ip_info["responses"] > 0.985)
        & (ip_info["requests"] / ip_info["responses"] < 1.015)
        & (ip_info["is_resolver"] == False)
    ]
    good_ips.to_csv("good_ips.csv")

    # seleziona dal csv solo le requests, inviate da ip in good_ips, di tipo A,
    # non retransmission, aventi una risposta con stesso id entro 60 secondi
    print(len(df))
    df = df[
        (df["ip.src"].isin(good_ips["ip"]))
        & (df["udp.dstport"] == 53)
        & (pd.isna(df["dns.retransmission"]))
        & (df["dns.qry.type"] == 1)
    ]
    print(len(df))
    df.to_csv(f"{os.path.splitext(csv)[0]}-clean.csv")
