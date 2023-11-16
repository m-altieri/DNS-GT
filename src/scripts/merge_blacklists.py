#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Merge blacklists in a single CSV file, for each category and quality."""

import os
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "path", help="Path to the data folder containing the blacklists folder."
)
args = argparser.parse_args()


# path = "."
bl_path = os.path.join(args.path, "blacklists")
bl_domains = {}
good_bl_domains = {}

for category in os.listdir(bl_path):
    # check whether it's a folder; if it's a file, skip
    if os.path.isfile(os.path.join(bl_path, category)):
        print(f"Skipping {category}: not a folder.")
        continue

    bl_domains[category] = []

    good_bls = os.path.join(bl_path, category, "good")
    ok_bls = os.path.join(bl_path, category, "ok")

    # traverse each category and read its 'good' blacklists
    for bl in os.listdir(good_bls):
        # these blacklists were already merged in blacklist.txt, don't use it again
        if bl == "blacklist.txt":
            print(f"Skipping {good_bls}/{bl}.")
            continue

        # append the domains in this blacklist to the blacklisted domains for this category
        with open(os.path.join(good_bls, bl), "r") as f:
            for line in f.readlines():
                if not line.startswith("#"):
                    domain = line.split(" ")[-1]
                    bl_domains[category].append(domain)

    # Remove duplicates
    bl_domains[category] = list(set(bl_domains[category]))

    # Write 'good' domains to file
    with open(os.path.join(bl_path, category, "good", "blacklist.txt"), "w") as f:
        f.write("".join(bl_domains[category]))

    # Save copy before adding 'ok' ones
    good_bl_domains[category] = bl_domains[category].copy()

    # traverse each category and read its 'ok' blacklists
    for bl in os.listdir(ok_bls):
        # these blacklists were already merged in blacklist.txt, don't use it again
        if bl == "blacklist.txt":
            print(f"Skipping {ok_bls}/{bl}.")
            continue

        # append the domains in this blacklist to the blacklisted domains for this category
        with open(os.path.join(ok_bls, bl), "r") as f:
            for line in f.readlines():
                if not line.startswith("#"):
                    domain = line.split(" ")[-1]
                    bl_domains[category].append(domain)

    # Remove duplicates again
    bl_domains[category] = list(set(bl_domains[category]))

    # Write also 'ok' ones to file (also includes good ones)
    with open(os.path.join(bl_path, category, "ok", "blacklist.txt"), "w") as f:
        f.write("".join(bl_domains[category]))

with open(os.path.join(bl_path, "good_blacklist.txt"), "w") as f:
    f.write("".join([d for category in good_bl_domains.values() for d in category]))

with open(os.path.join(bl_path, "ok_blacklist.txt"), "w") as f:
    f.write("".join([d for category in bl_domains.values() for d in category]))
