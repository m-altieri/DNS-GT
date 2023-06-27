import numpy as np
import pandas as pd
import os
import argparse
import re

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "input_folder", help="Folder containing processed PCAPs in .csv format"
)
argparser.add_argument("output_folder", help="Folder to save output arrays to")
argparser.add_argument(
    "--from-responses",
    action="store_const",
    const=True,
    default=False,
    help="Whether to extract query information from requests or responses. If True and some queries have no response, it will throw an error.",
)
argparser.add_argument("--from-original", action="store_true")
args = argparser.parse_args()


def clean_urls(url):
    url = re.sub("[^!-~]+", "", url).lower()
    # match = re.match('^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$', url)
    #    if match:
    #        url = match[0]
    return url


hosts_col = 1 if args.from_original else None
domains_col = 2 if args.from_original else None

# check input folder is valid
if not os.path.exists(args.input_folder) or len(os.listdir(args.input_folder)) == 0:
    raise FileNotFoundError(f"{args.input_folder} is not a valid input folder.")

# create output folder if not exists
if not os.path.exists(os.path.join(args.output_folder, "hosts")):
    os.makedirs(os.path.join(args.output_folder, "hosts"))
if not os.path.exists(os.path.join(args.output_folder, "domains")):
    os.makedirs(os.path.join(args.output_folder, "domains"))
if not os.path.exists(os.path.join(args.output_folder, "queries")):
    os.makedirs(os.path.join(args.output_folder, "queries"))

for filename in os.listdir(args.input_folder):
    # read processed pcap
    print(f"Reading {os.path.join(args.input_folder, filename)}")
    df = pd.read_csv(os.path.join(args.input_folder, filename), encoding="latin3")

    # get host and domain columns
    # @TODO this is now wrong. i've moved from extracting hosts and domains and processing them
    # to processing the df directly. this way, if i remove a host i remove the entire line automatically
    # there was a probable bug where i zip hosts and domains arrays of different lengths, so they don't actually match
    if not args.from_original:
        hosts = df["req_src_ip"] if not args.from_responses else df["res_dst_ip"]
        domains = (
            df["req_qry_domain"] if not args.from_responses else df["res_qry_domain"]
        )
    else:
        hosts = df.iloc[:, hosts_col]
        domains = df.iloc[:, domains_col]

    # remove NaNs (responses with no query)
    if args.from_original:
        df = df[df.iloc[:, hosts_col].notna() & df.iloc[:, domains_col].notna()]

    # clean domains (now they are in the form b'something')
    # print(domains[domains.str.startswith("b'") != False])
    # assert all(domains.str.startswith("b'")) or not any(domains.str.startswith("b'"))
    # if all(domains.str.startswith("b'")):
    #     domains = domains.map(lambda x: str(x).split("'")[1])
    assert all(df.iloc[:, domains_col].str.startswith("b'")) or not any(
        df.iloc[:, domains_col].str.startswith("b'")
    )
    if all(df.iloc[:, domains_col].str.startswith("b'")):
        df.iloc[:, domains_col] = df.iloc[:, domains_col].map(
            lambda x: str(x).split("'")[1]
        )

    if not args.from_original:
        hosts = hosts.map(clean_urls)
        domains = domains.map(clean_urls)
    else:
        df.iloc[:, hosts_col].map(clean_urls)
        df.iloc[:, domains_col].map(clean_urls)

    # save array of communications
    # np.save(
    #     os.path.join(args.output_folder, "queries", f"queries-{filename}.npy"),
    #     list(zip(hosts, domains)),
    # )
    np.save(
        os.path.join(args.output_folder, "queries", f"queries-{filename}.npy"),
        df.iloc[:, [hosts_col, domains_col]].to_numpy(),
    )

    # save arrays of unique hosts and domains
    # np.save(
    #     os.path.join(args.output_folder, "hosts", f"hosts-nonunique-{filename}.npy"),
    #     hosts,
    # )
    # np.save(
    #     os.path.join(
    #         args.output_folder, "domains", f"domains-nonunique-{filename}.npy"
    #     ),
    #     domains,
    # )
    # np.save(
    #     os.path.join(args.output_folder, "hosts", f"hosts-{filename}.npy"),
    #     hosts.unique(),
    # )
    # np.save(
    #     os.path.join(args.output_folder, "domains", f"domains-{filename}.npy"),
    #     domains.unique(),
    # )
    np.save(
        os.path.join(args.output_folder, "hosts", f"hosts-nonunique-{filename}.npy"),
        df.iloc[:, hosts_col],
    )
    np.save(
        os.path.join(
            args.output_folder, "domains", f"domains-nonunique-{filename}.npy"
        ),
        df.iloc[:, domains_col],
    )
    np.save(
        os.path.join(args.output_folder, "hosts", f"hosts-{filename}.npy"),
        df.iloc[:, hosts_col].unique(),
    )
    np.save(
        os.path.join(args.output_folder, "domains", f"domains-{filename}.npy"),
        df.iloc[:, domains_col].unique(),
    )
