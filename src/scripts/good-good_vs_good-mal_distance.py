import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.runs_management import RunManager


def eucl_dist(a, b):
    return np.linalg.norm(a - b)


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


argparser = argparse.ArgumentParser()
argparser.add_argument("model")
argparser.add_argument("run")
args = argparser.parse_args()

root_conf = RunManager.load_root_conf()
data_path = root_conf.get("data_path")

# Load
embeddings_path = f"../../runs/{args.model}/{args.run}/embeddings.npy"
embeddings = np.load(embeddings_path)
domains_path = os.path.join(data_path, "vocab", "domains_vocab.txt")
domains = None
with open(domains_path, "r") as f:
    domains = np.array(f.read().split("\n"))

labels = pd.read_csv("labels.csv")

N = 100

malicious_domains = []
common_domains = []

all_malicious = labels[labels["any"] == 1]["domain"].to_list()
print(all_malicious)

# Take the N most frequent malicious domain (according to (any, good))
for d in domains:
    if d in all_malicious:
        malicious_domains.append(d)
    if len(malicious_domains) == N:
        break
print(malicious_domains)

# Take the N most common domains
for d in domains:
    if d not in all_malicious:
        common_domains.append(d)
    if len(common_domains) == N:
        break
print(common_domains)

# malicious_domains = [
#     "googleads.g.doubleclick.net",
#     "cm.g.doubleclick.net",
#     "ad.doubleclick.net",
#     "ads.mopub.com",
#     "track.appsflyer.com",
#     "v10.vortex-win.data.microsoft.com",
#     "t.appsflyer.com",
#     "ssdk.adkmob.com",
#     "rtd.tubemogul.com",
#     "m.addthis.com",
#     "idsync.rlcdn.com",
#     "bid.g.doubleclick.net",
#     "tags.bluekai.com",
#     "ads.flurry.com",
#     "log.dmtry.com",
#     "dpm.demdex.net",
#     "p.rfihub.com",
#     "sync.adaptv.advertising.com",
#     "pasta.esfile.duapps.com",
#     "dts.ushareit.com",
# ]

# common_domains = domains[:N]

# common_domains = [
#     "edge-mqtt.facebook.com",
#     "graph.facebook.com",
#     "www.google.com",
#     "clients3.google.com",
#     "android.clients.google.com",
#     "www.google.co.in",
#     "www.facebook.com",
#     "mtalk.google.com",
#     "www.msftncsi.com",
#     "asia.api.targetingmantra.com",
#     "international-gfe.download.nvidia.com",
#     "clients4.google.com",
#     "mail.google.com",
#     "googleads.g.doubleclick.net",
#     "ssl.gstatic.com",
#     "www.googleapis.com",
#     "plus.google.com",
#     "graph.instagram.com",
#     "accounts.google.com",
#     "www.gstatic.com",
# ]

common_emb = embeddings[[np.where(domains == d)[0][0] for d in common_domains]]
malicious_emb = embeddings[[np.where(domains == d)[0][0] for d in malicious_domains]]

avg_cc_eucl_dist = np.zeros((len(common_emb)))
avg_cm_eucl_dist = np.zeros((len(common_emb)))
avg_cc_cos_sim = np.zeros((len(common_emb)))
avg_cm_cos_sim = np.zeros((len(common_emb)))
avg_mc_eucl_dist = np.zeros((len(malicious_emb)))
avg_mm_eucl_dist = np.zeros((len(malicious_emb)))
avg_mc_cos_sim = np.zeros((len(malicious_emb)))
avg_mm_cos_sim = np.zeros((len(malicious_emb)))

# Line plots
for i, emb1 in enumerate(common_emb):
    dist = 0
    sim = 0
    for _, emb2 in enumerate(common_emb):
        dist += eucl_dist(emb1, emb2)
        sim += cos_sim(emb1, emb2)
    avg_cc_eucl_dist[i] = dist / len(common_emb)
    avg_cc_cos_sim[i] = sim / len(common_emb)

for i, emb1 in enumerate(common_emb):
    dist = 0
    sim = 0
    for _, emb2 in enumerate(malicious_emb):
        dist += eucl_dist(emb1, emb2)
        sim += cos_sim(emb1, emb2)
    avg_cm_eucl_dist[i] = dist / len(common_emb)
    avg_cm_cos_sim[i] = sim / len(common_emb)

for i, emb1 in enumerate(malicious_emb):
    dist = 0
    sim = 0
    for _, emb2 in enumerate(common_emb):
        dist += eucl_dist(emb1, emb2)
        sim += cos_sim(emb1, emb2)
    avg_mc_eucl_dist[i] = dist / len(malicious_emb)
    avg_mc_cos_sim[i] = sim / len(malicious_emb)

for i, emb1 in enumerate(malicious_emb):
    dist = 0
    sim = 0
    for _, emb2 in enumerate(malicious_emb):
        dist += eucl_dist(emb1, emb2)
        sim += cos_sim(emb1, emb2)
    avg_mm_eucl_dist[i] = dist / len(malicious_emb)
    avg_mm_cos_sim[i] = sim / len(malicious_emb)

# Sort distances / similarities in descending order
if True:
    avg_cc_eucl_dist = -np.sort(-avg_cc_eucl_dist)
    avg_cm_eucl_dist = -np.sort(-avg_cm_eucl_dist)
    avg_cc_cos_sim = -np.sort(-avg_cc_cos_sim)
    avg_cm_cos_sim = -np.sort(-avg_cm_cos_sim)
    avg_mc_eucl_dist = -np.sort(-avg_mc_eucl_dist)
    avg_mm_eucl_dist = -np.sort(-avg_mm_eucl_dist)
    avg_mc_cos_sim = -np.sort(-avg_mc_cos_sim)
    avg_mm_cos_sim = -np.sort(-avg_mm_cos_sim)


plt.figure(figsize=(5, 2.5))
plt.plot(avg_cc_eucl_dist, color="blue", label="Common domains")
plt.plot(avg_cm_eucl_dist, color="red", label="Malicious domains")
plt.xlabel("Common domains")
plt.ylabel("Euclidean distance")
plt.subplots_adjust(bottom=0.2)
plt.legend()
plt.tight_layout()
plt.savefig("../../runs/_extras/DELM/c_eucl.png", bbox_inches="tight")

plt.figure(figsize=(5, 2.5))
plt.plot(avg_cc_cos_sim, color="blue", label="Common domains")
plt.plot(avg_cm_cos_sim, color="red", label="Malicious domains")
plt.legend()
plt.xlabel("Common domains")
plt.ylabel("Cosine similarity")
plt.subplots_adjust(bottom=0.2)
plt.savefig("../../runs/_extras/DELM/c_cos.png", bbox_inches="tight")

plt.figure(figsize=(5, 2.5))
plt.plot(avg_mc_eucl_dist, color="blue", label="Common domains")
plt.plot(avg_mm_eucl_dist, color="red", label="Malicious domains")
plt.legend()
plt.savefig("../../runs/_extras/DELM/m_eucl.png")

plt.figure(figsize=(5, 2.5))
plt.plot(avg_mc_cos_sim, color="blue", label="Common domains")
plt.plot(avg_mm_cos_sim, color="red", label="Malicious domains")
plt.legend()
plt.savefig("../../runs/_extras/DELM/m_cos.png")

# Heatmaps
cc_eucl_dists = np.zeros((len(common_emb), len(common_emb)))
cc_cos_sims = np.zeros((len(common_emb), len(common_emb)))
for i, emb1 in enumerate(common_emb):
    for j, emb2 in enumerate(common_emb):
        cc_eucl_dists[i, j] = eucl_dist(emb1, emb2)
        cc_cos_sims[i, j] = cos_sim(emb1, emb2)

cm_eucl_dists = np.zeros((len(common_emb), len(common_emb)))
cm_cos_sims = np.zeros((len(common_emb), len(common_emb)))
for i, emb1 in enumerate(common_emb):
    for j, emb2 in enumerate(malicious_emb):
        cm_eucl_dists[i, j] = eucl_dist(emb1, emb2)
        cm_cos_sims[i, j] = cos_sim(emb1, emb2)

mc_eucl_dists = np.zeros((len(common_emb), len(common_emb)))
mc_cos_sims = np.zeros((len(common_emb), len(common_emb)))
for i, emb1 in enumerate(malicious_emb):
    for j, emb2 in enumerate(common_emb):
        mc_eucl_dists[i, j] = eucl_dist(emb1, emb2)
        mc_cos_sims[i, j] = cos_sim(emb1, emb2)

mm_eucl_dists = np.zeros((len(common_emb), len(common_emb)))
mm_cos_sims = np.zeros((len(common_emb), len(common_emb)))
for i, emb1 in enumerate(malicious_emb):
    for j, emb2 in enumerate(malicious_emb):
        mm_eucl_dists[i, j] = eucl_dist(emb1, emb2)
        mm_cos_sims[i, j] = cos_sim(emb1, emb2)

print(np.mean(cc_eucl_dists))
print(np.mean(cm_eucl_dists))
print(np.mean(mc_eucl_dists))
print(np.mean(mm_eucl_dists))
print(np.mean(cc_cos_sims))
print(np.mean(cm_cos_sims))
print(np.mean(mc_cos_sims))
print(np.mean(mm_cos_sims))


# Sort
if False:
    cc_eucl_dists = cc_eucl_dists[np.sum(cc_eucl_dists, axis=1).argsort()]
    cm_eucl_dists = cm_eucl_dists[np.sum(cm_eucl_dists, axis=1).argsort()]
    mc_eucl_dists = mc_eucl_dists[np.sum(mc_eucl_dists, axis=1).argsort()]
    mm_eucl_dists = mm_eucl_dists[np.sum(mm_eucl_dists, axis=1).argsort()]
    cc_cos_sims = cc_cos_sims[np.sum(cc_cos_sims, axis=1).argsort()]
    cm_cos_sims = cm_cos_sims[np.sum(cm_cos_sims, axis=1).argsort()]
    mc_cos_sims = mc_cos_sims[np.sum(mc_cos_sims, axis=1).argsort()]
    mm_cos_sims = mm_cos_sims[np.sum(mm_cos_sims, axis=1).argsort()]

plt.figure(figsize=(5, 5))
sns.heatmap(
    np.concatenate(
        (
            np.concatenate((cc_eucl_dists, cm_eucl_dists), axis=1),
            np.concatenate((mc_eucl_dists, mm_eucl_dists), axis=1),
        ),
        axis=0,
    )
)
plt.xlabel("Domains")
plt.ylabel("Domains")
plt.savefig("../../runs/_extras/DELM/eucl_heatmap.png")

plt.figure(figsize=(5, 5))
sns.heatmap(
    np.concatenate(
        (
            np.concatenate((cc_cos_sims, cm_cos_sims), axis=1),
            np.concatenate((mc_cos_sims, mm_cos_sims), axis=1),
        ),
        axis=0,
    )
)
plt.savefig("../../runs/_extras/DELM/cos_heatmap.png")
