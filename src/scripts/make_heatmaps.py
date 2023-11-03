import seaborn as sns
import matplotlib.pyplot as plt

data1 = [[91347, 115889], [31625, 577353]]  # DELM
data2 = [[75194, 111365], [90460, 416596]]  # CBOW
data3 = [[68295, 118264], [31950, 475106]]  # SkipGram

fig, axs = plt.subplots(ncols=3, sharey=True)
sns.heatmap(
    data1,
    annot=True,
    square=True,
    fmt="d",
    ax=axs[0],
    cbar=None,
    vmin=31625,
    vmax=577353,
    xticklabels=["Pred 0", "Pred 1"],
    yticklabels=["True 0", "True 1"],
)
sns.heatmap(
    data2,
    annot=True,
    square=True,
    fmt="d",
    ax=axs[1],
    cbar=None,
    vmin=31625,
    vmax=577353,
    xticklabels=["Pred 0", "Pred 1"],
    yticklabels=["True 0", "True 1"],
)
sns.heatmap(
    data3,
    annot=True,
    square=True,
    fmt="d",
    ax=axs[2],
    cbar=None,
    vmin=31625,
    vmax=577353,
    xticklabels=["Pred 0", "Pred 1"],
    yticklabels=["True 0", "True 1"],
)
plt.savefig("heatmap.png")
