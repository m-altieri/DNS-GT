from colorama import Fore, Style
from tqdm import tqdm
import pandas as pd
import sys
import os


small_domains_path = "../../data/vocabs/small/domains_vocab.txt"
all_domains_path = "../../data/vocabs/all/domains_vocab.txt"
blacklist_paths = "../../blacklists"

with open(small_domains_path, "r") as f:
    small_domains = f.read().split("\n")

with open(all_domains_path, "r") as f:
    all_domains = f.read().split("\n")

print("Loading blacklist data...")
blacklists = {}
with open(
    os.path.join(blacklist_paths, "advertising", "good", "blacklist.txt"), "r"
) as f:
    blacklists["advertising_good"] = f.read().split("\n")
with open(
    os.path.join(blacklist_paths, "advertising", "ok", "blacklist.txt"), "r"
) as f:
    blacklists["advertising_ok"] = f.read().split("\n")
with open(
    os.path.join(blacklist_paths, "malicious", "good", "blacklist.txt"), "r"
) as f:
    blacklists["malicious_good"] = f.read().split("\n")
with open(os.path.join(blacklist_paths, "malicious", "ok", "blacklist.txt"), "r") as f:
    blacklists["malicious_ok"] = f.read().split("\n")
with open(
    os.path.join(blacklist_paths, "suspicious", "good", "blacklist.txt"), "r"
) as f:
    blacklists["suspicious_good"] = f.read().split("\n")
with open(os.path.join(blacklist_paths, "suspicious", "ok", "blacklist.txt"), "r") as f:
    blacklists["suspicious_ok"] = f.read().split("\n")
with open(os.path.join(blacklist_paths, "tracking", "good", "blacklist.txt"), "r") as f:
    blacklists["tracking_good"] = f.read().split("\n")
with open(os.path.join(blacklist_paths, "tracking", "ok", "blacklist.txt"), "r") as f:
    blacklists["tracking_ok"] = f.read().split("\n")
with open(os.path.join(blacklist_paths, "other", "good", "blacklist.txt"), "r") as f:
    blacklists["other_good"] = f.read().split("\n")
with open(os.path.join(blacklist_paths, "other", "ok", "blacklist.txt"), "r") as f:
    blacklists["other_ok"] = f.read().split("\n")

# Remove empty strings
blacklists = {k: [x for x in blacklists[k] if x != ""] for k in blacklists}

df = pd.DataFrame(
    columns=pd.MultiIndex.from_arrays(
        [
            [
                "advertising",
                "advertising",
                "malicious",
                "malicious",
                "suspicious",
                "suspicious",
                "tracking",
                "tracking",
                "other",
                "other",
                "any",
                "any",
            ],
            [
                "good",
                "ok",
                "good",
                "ok",
                "good",
                "ok",
                "good",
                "ok",
                "good",
                "ok",
                "good",
                "ok",
            ],
        ]
    )
)
df.insert(0, "domain", all_domains)
df = df.fillna(0)
print(df.head(5))

print("Creating domain labels...")

# Also check subdomains
subdomains = df["domain"].copy()
for i in tqdm(range(df["domain"].str.split(".").str.len().max())):
    # 1 = IN blacklist, 0 = NOT IN blacklist
    # df.where setta il nuovo valore dove la condizione è FALSA
    df["advertising", "good"] = df["advertising", "good"].where(
        ~subdomains.isin(blacklists["advertising_good"]), 1
    )
    df["advertising", "ok"] = df["advertising", "ok"].where(
        ~subdomains.isin(blacklists["advertising_ok"]), 1
    )
    df["malicious", "good"] = df["malicious", "good"].where(
        ~subdomains.isin(blacklists["malicious_good"]), 1
    )
    df["malicious", "ok"] = df["malicious", "ok"].where(
        ~subdomains.isin(blacklists["malicious_ok"]), 1
    )
    df["suspicious", "good"] = df["suspicious", "good"].where(
        ~subdomains.isin(blacklists["suspicious_good"]), 1
    )
    df["suspicious", "ok"] = df["suspicious", "ok"].where(
        ~subdomains.isin(blacklists["suspicious_ok"]), 1
    )
    df["tracking", "good"] = df["tracking", "good"].where(
        ~subdomains.isin(blacklists["tracking_good"]), 1
    )
    df["tracking", "ok"] = df["tracking", "ok"].where(
        ~subdomains.isin(blacklists["tracking_ok"]), 1
    )
    df["other", "good"] = df["other", "good"].where(
        ~subdomains.isin(blacklists["other_good"]), 1
    )
    df["other", "ok"] = df["other", "ok"].where(
        ~subdomains.isin(blacklists["other_ok"]), 1
    )
    subdomains = subdomains.str.partition(".")[
        2
    ]  # strip the bottom level from domain name

df["any", "good"] = df["any", "good"].where(
    (df["advertising", "good"] == 0)
    & (df["malicious", "good"] == 0)
    & (df["suspicious", "good"] == 0)
    & (df["tracking", "good"] == 0)
    & (df["other", "good"] == 0),
    1,
)
df["any", "ok"] = df["any", "ok"].where(
    (df["advertising", "ok"] == 0)
    & (df["malicious", "ok"] == 0)
    & (df["suspicious", "ok"] == 0)
    & (df["tracking", "ok"] == 0)
    & (df["other", "ok"] == 0),
    1,
)

print(df)
df.to_csv("labels.csv")

# malicious = list(set(blacklist) & set(domains))
# print("Exact matches:", len(malicious))

# print("Matching subdomains...")
# # Etichetto come malicious tutti i sottodomini di quelli malicious
# nonmalicious = list(set(domains) - set(malicious))
# with tqdm(
#     total=len(nonmalicious),
#     bar_format="{desc:<25.25}{percentage:3.2f}%|{bar:10}{r_bar}",
# ) as pbar:
#     for d in nonmalicious:
#         splitted = d.split(".")
#         for l in range(1, len(splitted)):  # for each level
#             subdomain = ".".join(splitted[-l:])
#             if subdomain in malicious:
#                 malicious.append(d)
#                 pbar.set_description(
#                     f"{d[:-len(subdomain)]}{Style.BRIGHT}{subdomain}{Style.RESET_ALL}"
#                 )
#         pbar.update()

# print("Malicious:", len(malicious))
