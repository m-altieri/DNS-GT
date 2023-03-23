import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def eucl_dist(a, b):
    return np.linalg.norm(a - b)


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Load
embeddings_path = "../embeddings/embeddings-increasingmasking.npy"
embeddings = np.load(embeddings_path)
domains_path = "../preprocessing/vocabs/small/domains_vocab.txt"
domains = None
with open(domains_path, "r") as f:
    domains = np.array(f.read().split("\n"))

malicious_domains = [
    "bootstraplugin.com",
    "buzzonclick.com",
    "jump.ogtrk.net",
    "mgid.com",
    "reimageplus.com",
    "www.buzzonclick.com",
    "www.mgid.com",
    "www.reimageplus.com",
    "www.videodownloadconverter.com",
    "bestadbid.com",
    "parkingcrew.net",
    "download.televisionfanatic.com",
    "mackeeperapp.mackeeper.com",
    "yotefiles.com",
    "bigbangads.go2cloud.org",
    "hp.myway.com",
    "play.leadzu.com",
]

common_domains = [
    "international-gfe.download.nvidia.com",
    "edge-mqtt.facebook.com",
    "graph.facebook.com",
    "www.google.com",
    "clients3.google.com",
    "android.clients.google.com",
    "mtalk.google.com",
    "officecdn.microsoft.com",
    "ads.adaptv.advertising.com",
    "www.msftncsi.com",
    "www.facebook.com",
    "www.googleapis.com",
    "international-gfe.download.nvidia.com.global.ogslb.com",
    "clients4.google.com",
    "googleads.g.doubleclick.net",
    "www.google.co.in",
    "graph.instagram.com",
]

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


plt.figure(figsize=(5, 2.5))
plt.plot(avg_cc_eucl_dist, color="blue", label="Common domains")
plt.plot(avg_cm_eucl_dist, color="red", label="Malicious domains")
plt.xlabel("Common domains")
plt.ylabel("Cosine similarity")
plt.subplots_adjust(bottom=0.2)
plt.legend()

plt.savefig("c_eucl.png")

plt.figure(figsize=(5, 2.5))
plt.plot(avg_cc_cos_sim, color="blue", label="Common domains")
plt.plot(avg_cm_cos_sim, color="red", label="Malicious domains")
plt.legend()
plt.xlabel("Common domains")
plt.ylabel("Euclidean distance")
plt.subplots_adjust(bottom=0.2)
plt.savefig("c_cos.png")

plt.figure(figsize=(5, 2.5))
plt.plot(avg_mc_eucl_dist, color="blue", label="Common domains")
plt.plot(avg_mm_eucl_dist, color="red", label="Malicious domains")
plt.legend()
plt.savefig("m_eucl.png")

plt.figure(figsize=(5, 2.5))
plt.plot(avg_mc_cos_sim, color="blue", label="Common domains")
plt.plot(avg_mm_cos_sim, color="red", label="Malicious domains")
plt.legend()
plt.savefig("m_cos.png")

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
plt.savefig("eucl_heatmap.png")

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
plt.savefig("cos_heatmap.png")
