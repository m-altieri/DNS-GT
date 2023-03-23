import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
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

# Boolean mask of domains to mark
# marked_domains = np.isin(domains, [d for d in domains if d.endswith('google.com') or d.endswith('youtube.com') or d.endswith('facebook.com')])
# Continuous value (# occurrences)
# with open('domains_occurrences.txt', 'r') as f:
#     occurrences = f.read()
# marked_domains =
# Save blobs of the final plot
# np.savetxt('left_blob.txt', domains[np.unique(np.where(np.isin(embeddings, [e for e in embeddings if e[0] < 0.5 and e[1] > -1 and e[1] < 0.8]))[0])], fmt='%s')


# Clustering
clustering = False
if clustering:
    k = 5
    samples_per_cluster = 10
    cluster_mapping = KMeans(n_clusters=k, max_iter=3000, tol=1e-6).fit_predict(
        embeddings
    )
    centroids = np.zeros((k, embeddings.shape[-1]))
    medoids_idx = np.zeros((k), dtype=int)

# PCA
if False:
    embeddings = PCA(n_components=10).fit_transform(embeddings)
    print(embeddings)
    print(embeddings.shape)

# Heatmaps
if False:
    stratified_samples_idx = np.zeros((k, samples_per_cluster), dtype=int)
    for c in range(k):
        centroids[c] = np.mean(embeddings[np.where(cluster_mapping == c)[0]], axis=0)
        medoids_idx[c] = np.array(
            [eucl_dist(e, centroids[c]) for e in embeddings]
        ).argmin()  # Most similar embedding to centroid c
        if not cluster_mapping[medoids_idx[c]] == c:
            print(
                f"Weird! domain {domains[medoids_idx[c]]} is the best representative for cluster {c} but it is not in cluster {c}!"
            )
        try:
            stratified_samples_idx[c] = np.random.choice(
                np.where(cluster_mapping == c)[0], samples_per_cluster, replace=False
            )
        except ValueError:
            print(
                f"Trying to sample {samples_per_cluster} elements from cluster {c}, but it "
                + f"has less than {samples_per_cluster} elements! Some elements for cluster "
                + f"{c} will be sampled multiple times."
            )
            stratified_samples_idx[c] = np.random.choice(
                np.where(cluster_mapping == c)[0], samples_per_cluster, replace=True
            )
    stratified_samples_idx = stratified_samples_idx.flatten()

    distances = np.zeros((k * samples_per_cluster, k * samples_per_cluster))
    for i, _ in enumerate(distances):
        for j, _ in enumerate(distances):
            # distances[i, j] = cos_sim(
            #     embeddings[medoids_idx[i]], embeddings[medoids_idx[j]]
            # )  # Centroids cosine similarity

            distances[i, j] = eucl_dist(
                embeddings[stratified_samples_idx[i]],
                embeddings[stratified_samples_idx[j]],
            )
    sns.heatmap(distances)
    plt.savefig("heatmap.png")

    # sys.exit(0)

    with open("blacklist.txt", "r") as f:
        blacklist = f.read()
    blacklist = blacklist.split("\n")
    blacklist = [b.replace("0.0.0.0 ", "") for b in blacklist]
    malicious_idx = np.where(np.isin(domains, blacklist))[0]
    # google_nn_idx = np.array(
    #     [eucl_dist(embeddings[87966], e) for e in embeddings]
    # ).argsort()[:17]
    mal_and_gen_idx = np.concatenate((malicious_idx, medoids_idx))

    # for loop annidato, per ogni malevolo e ogni medoide, prendo 10-NN per ciascuno di loro
    # e metto nella heatmap nella parte alta i 10 malevoli e nella bassa i 10 buoni
    # alla fine avrò 17*16 heatmaps
    top_nn = 10
    distances = np.zeros((20, 20))
    for i, mal_index in enumerate(malicious_idx):
        mal_nn = np.argsort([eucl_dist(embeddings[mal_index], e) for e in embeddings])[
            :top_nn
        ]
        # for j, med_index in enumerate(medoids_idx):
        for j, med_index in enumerate(np.where(domains == "www.google.com")[0]):
            gen_nn = np.argsort(
                [eucl_dist(embeddings[med_index], e) for e in embeddings]
            )[:top_nn]
            mal_and_gen_nn = np.concatenate((mal_nn, gen_nn))
            # compute heatmap
            for k in range(len(mal_and_gen_nn)):
                for l in range(len(mal_and_gen_nn)):
                    distances[k, l] = (
                        eucl_dist(
                            embeddings[mal_and_gen_nn[k]], embeddings[mal_and_gen_nn[l]]
                        )
                        ** 2
                    )
            plt.clf()
            plt.cla()
            sns.heatmap(distances)
            plt.savefig(f"heatmaps/heatmap-{i}-{j}.png")
            print(f"Saved heatmap for {domains[mal_index]} and {domains[med_index]}")

# sys.exit(0)

# TSNE
# Grid
perps = [30.0]
n_iters = 5
increment = 250

for perp in perps:
    tsne = embeddings.copy()
    for n_iter in range(n_iters):
        tsne = TSNE(
            n_components=2, perplexity=perp, n_iter=increment, verbose=True
        ).fit_transform(tsne)

        plt.figure(figsize=(40, 40))
        plt.scatter(
            tsne[:, 0],
            tsne[:, 1],
            c=cluster_mapping if clustering else None,
            marker=",",
        )

        mark_malicious = False
        mark_common = False
        if mark_malicious:
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
            malicious_tsne = tsne[np.where(np.isin(domains, malicious_domains))[0]]
            plt.scatter(
                malicious_tsne[:, 0],
                malicious_tsne[:, 1],
                c="#ff0000",
                s=2500,
            )
            for i, _ in enumerate(malicious_tsne):
                plt.scatter(
                    malicious_tsne[i, 0],
                    malicious_tsne[i, 1],
                    c="#ffffff",
                    s=2000,
                    marker=f"${i}$",
                )
        if mark_common:
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
            ]
            common_tsne = embeddings[np.where(np.isin(domains, common_domains))[0]]
            plt.scatter(common_tsne[:, 0], common_tsne[:, 1], c="#00aa55", s=2500)
            for i, _ in enumerate(common_tsne):
                plt.scatter(
                    common_tsne[i, 0],
                    common_tsne[i, 1],
                    c="#ffffff",
                    s=2000,
                    marker=f"${i}$",
                )

        plt.savefig(f"tsne/tsne-p{perp}-i{(n_iter+1)*increment}.png")
