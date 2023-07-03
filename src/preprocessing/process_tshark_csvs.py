# -*- coding: utf-8 -*-
"""Preprocessing of CSV files after conversion with `tshark`.

The preprocessing applies the fo

Definitions
-----------

request:
    DNS packet with destination port equal to 53
response:
    DNS packet with source port equal to 53
resolver
    - IP with packets sent 

Valid IPs are sorted as follows:
    - at least 100 requests for the current hour,
    - a ratio of requests to responses between 0.985 and 1.015,
    - not dns resolvers 


Author: Massimiliano Altieri <massimiliano.altieri@ec.europa.eu>
Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import sys
import pandas as pd
from datetime import datetime

from tacks.utils.base import Workspace, get_argparser, get_config

# -- Argument Parser and Workspace

argparser = get_argparser(
    sys.modules[__name__].__doc__,
    flags=["debug", "silent"],
)

args = argparser.parse_args()
args.force = True

workspace = Workspace(name="PCAP_Dataset", instance_name="Preprocessing", args=args)
config = get_config()


# -- Data
data_path = config.get_path("paths", "data") / "TI-2016"
tcsv_path = data_path / "tcsv"

csv_path = data_path / "csv"
csv_path.mkdir(exist_ok=True)

# -- Preprocessing
for tcsvfile_path in tcsv_path.glob("*.csv"):
    workspace.logger.info(f"Analysing {tcsvfile_path}...")

    # get file name
    filename = tcsvfile_path.name

    # check whether the file was already pre-processed
    if (csv_path / filename).exists():
        workspace.logger.info("Already preprocessed tshark CSV. Skipped.")
        continue

    # load packets with pandas
    packets = pd.read_csv(str(tcsvfile_path), delimiter=";", on_bad_lines="skip")

    # renaming column names
    packets = packets.rename(
        columns={
            "frame.number": "frame_id",
            "frame.time_epoch": "timestamp",
            "ip.src": "ip_src",
            "ip.dst": "ip_dst",
            "udp.srcport": "port_src",
            "udp.dstport": "port_dst",
            "dns.id": "id",
            "dns.retransmission": "is_retransmission",
            "dns.qry.name": "qry_name",
            "dns.qry.type": "qry_type",
            "dns.flags.response": "is_response",
            "dns.resp.name": "resp_names",
            "dns.resp.type": "resp_types",
            "dns.flags.rcode": "rcode",
        }
    )

    # fill missing values
    packets = packets.fillna(
        {"port_src": -1, "port_dst": -1, "qry_type": -1, "is_retransmission": 0}
    )

    # define type of known fields
    packets = packets.astype(
        {
            "frame_id": int,
            "timestamp": float,
            "port_src": int,
            "port_dst": int,
            "qry_type": int,
            "is_response": bool,
            "is_retransmission": bool,
        }
    )

    # remove queries with missing name
    packets.dropna(subset=["qry_name"], inplace=True)

    # set the time origin to Day 0 00:00
    packets["timestamp"] -= datetime(2016, 4, 24, 0, 0, 0).timestamp()

    # convert query types
    packets["qry_type"] = packets["qry_type"].astype(int)
    packets["qry_type"].replace(
        to_replace={
            -1: "NA",
            0: "NA",
            1: "A",
            2: "NS",
            5: "CNAME",
            6: "SOA",
            12: "PTR",
            15: "MX",
            16: "TXT",
            28: "AAAA",
            33: "SRV",
            43: "DS",
            48: "DNSKEY",
            255: "*",
        },
        inplace=True,
    )

    # get unique IPS
    ips = packets["ip_src"].unique().tolist()

    # get number of requests and responses per IP
    packets_by_ipsrc = packets.groupby("ip_src")
    packets_by_ipdst = packets.groupby("ip_dst")

    n_requests_sent = packets_by_ipsrc.apply(lambda g: (g["port_dst"] == 53).sum())
    n_responses_sent = packets_by_ipsrc.apply(lambda g: (g["port_src"] == 53).sum())

    n_requests_received = packets_by_ipdst.apply(lambda g: (g["port_dst"] == 53).sum())
    n_responses_received = packets_by_ipdst.apply(lambda g: (g["port_src"] == 53).sum())

    ip_info = (
        pd.concat(
            [
                n_requests_sent,
                n_responses_sent,
                n_requests_received,
                n_responses_received,
            ],
            axis=1,
        )
        .reset_index()
        .fillna(0)
        .rename(
            columns={
                "index": "ip",
                0: "n_requests_sent",
                1: "n_responses_sent",
                2: "n_requests_received",
                3: "n_responses_received",
            }
        )
        .astype(
            {
                "n_requests_sent": int,
                "n_responses_sent": int,
                "n_requests_received": int,
                "n_responses_received": int,
            }
        )
    )

    ip_info["is_resolver"] = (ip_info["n_requests_received"] > 0) | (
        ip_info["n_responses_sent"] > 0
    )

    # requests, responses, is_resolver = [], [], []

    # for each unique ip within the current hour, its number of requests,
    # responses, and whether it is a resolver

    # for ip in ips:
    #     # get number of requests
    #     requests.append(
    #         len(packets[(packets["ip_src"] == ip) & (packets["port_dst"] == 53)])
    #     )

    #     # get number of responses
    #     responses.append(
    #         len(packets[(packets["ip_dst"] == ip) & (packets["port_src"] == 53)])
    #     )

    #     # whether it is a resolver
    #     is_resolver.append(
    #         not packets[(packets["ip_src"] == ip) & (packets["port_src"] == 53)].empty
    #         or not packets[
    #             (packets["ip_dst"] == ip) & (packets["port_dst"] == 53)
    #         ].empty
    #     )

    # ip_info2 = pd.DataFrame(
    #     {
    #         "ip": ips,
    #         "requests": requests,
    #         "responses": responses,
    #         "is_resolver": is_resolver,
    #     }
    # )

    # # Test
    # for ip in ips:
    #     assert (
    #         ip_info[ip_info["ip"] == ip]["requests"].iloc[0]
    #         == ip_info2[ip_info2["ip"] == ip]["n_requests_sent"].iloc[0]
    #     )

    #     assert (
    #         ip_info[ip_info["ip"] == ip]["responses"].iloc[0]
    #         == ip_info2[ip_info2["ip"] == ip]["n_responses_received"].iloc[0]
    #     )
    #     assert (
    #         ip_info[ip_info["ip"] == ip]["is_resolver"].iloc[0]
    #         == ip_info2[ip_info2["ip"] == ip]["is_resolver"].iloc[0]
    #     )

    # get valid IPs
    ip_info["ratio_req_res"] = (
        ip_info["n_requests_sent"] / ip_info["n_responses_received"]
    )
    ip_info["is_valid"] = (
        (ip_info["n_requests_sent"] > 100)
        & (ip_info["ratio_req_res"] > 0.985)
        & (ip_info["ratio_req_res"] < 1.015)
        & (ip_info["is_resolver"] == False)
    )

    # save info on IPS as CSV
    ip_info.to_csv(data_path / "artifacts" / f"{filename}_ip_info.csv")

    valid_ips = ip_info[ip_info["is_valid"]]["ip"]

    # select from the csv file only the requests, sent by an ip in good_ips, of type A,
    # not retransmission, [TODO having a response with the same id within 60 seconds]

    valid_packets = packets[
        packets["ip_src"].isin(valid_ips)
        & (packets["port_dst"] == 53)
        & ~packets["is_retransmission"]
        & (packets["qry_type"] == "A")
    ]

    valid_packets.to_csv(csv_path / filename, sep=";")

    workspace.logger.info(
        "Found %d valid packets for %d IPs", len(valid_packets), len(valid_ips)
    )
