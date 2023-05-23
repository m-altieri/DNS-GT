import os
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument("model", help="model name")
argparser.add_argument("-t", "--type", help="model type")
args = argparser.parse_args()

running_metrics = {
    "accuracy": {},
    "precision": {},
    "recall": {},
    "f1-binary": {},
    "f1-micro": {},
    "f1-macro": {},
    "f1-weighted": {},
}

predictions_path = f"../../predictions/{args.model}"
if args.type:
    predictions_path = os.path.join(predictions_path, args.type)

fig = plt.figure(figsize=(15, 15))

rocs = {}
for pred_file in sorted(
    os.listdir(predictions_path)
):  # each pred_file is a different fold
    if not pred_file.endswith(".csv"):
        continue

    model = pred_file.split(".")[0]

    df = pd.read_csv(os.path.join(predictions_path, pred_file))
    if (
        "pred_hard" not in df
    ):  # some models perform better if they are not required to provide probabilities
        df["pred_hard"] = df["pred"].round()
        print("Adding pred_hard")

    if pred_file.lower() == "cbow.csv" or pred_file.lower() == "skipgram.csv":
        df = pd.concat([df] * 32, ignore_index=True)

    print(
        f"Acc: {metrics.accuracy_score(df['true'], df['pred_hard']):<10.3f}\n "
        + f"P: {metrics.precision_score(df['true'], df['pred_hard']):<10.3f}\n "
        + f"R: {metrics.recall_score(df['true'], df['pred_hard']):<10.3f}\n "
        + f"F1-b: {metrics.f1_score(df['true'], df['pred_hard'], average='binary'):<10.3f}\n "
        + f"F1-m: {metrics.f1_score(df['true'], df['pred_hard'], average='micro'):<10.3f}\n "
        + f"F1-M: {metrics.f1_score(df['true'], df['pred_hard'], average='macro'):<10.3f}\n "
        + f"F1-w: {metrics.f1_score(df['true'], df['pred_hard'], average='weighted'):<10.3f}\n"
    )

    # CONFUSION MATRIX
    print(metrics.confusion_matrix(df["true"], df["pred_hard"], labels=[1, 0]))
    metrics.ConfusionMatrixDisplay.from_predictions(
        df["true"],
        df["pred_hard"],
        labels=[1, 0],
        cmap="plasma",
        values_format="",
        text_kw={"fontsize": "x-large"},
    )
    plt.savefig(f"{pred_file.capitalize()}.pdf", format="pdf")

    fig = plt.figure(figsize=(8, 8))
    plt.clf()

    # PLOT ROC CURVE
    fpr, tpr, _ = metrics.roc_curve(df["true"], df["pred"])
    roc_auc = metrics.auc(fpr, tpr)
    print("AUC:", roc_auc)

    plt.plot(
        fpr,
        tpr,
        label=f"ROC fold {model} (AUC = {roc_auc:.2f})",
    )
    rocs[model] = (fpr, tpr)

    # INIT METRICS
    for threshold in np.arange(0, 1, 0.1):
        for metric in running_metrics:
            if threshold not in running_metrics[metric]:
                running_metrics[metric][threshold] = 0

    print(f"\n{pred_file.split('.')[0]}:")
    print(
        f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-binary':<10} {'F1-micro':<10} {'F1-macro':<10} {'F1-weighted':<10}"
    )

    # COMPUTE METRICS
    for threshold in np.arange(0, 1, 0.1):
        # if (
        #     "pred_hard" not in df
        # ):  # some models perform better if they are not required to provide probabilities
        #     df["pred_hard"] = (df["pred"] > threshold).astype(int)
        #     print("Adding pred_hard")
        df["pred_threshold"] = (df["pred"] > threshold).astype(int)
        running_metrics["accuracy"][threshold] += metrics.accuracy_score(
            df["true"], df["pred_threshold"]
        )
        running_metrics["precision"][threshold] += metrics.precision_score(
            df["true"], df["pred_threshold"]
        )
        running_metrics["recall"][threshold] += metrics.recall_score(
            df["true"], df["pred_threshold"]
        )
        running_metrics["f1-binary"][threshold] += metrics.f1_score(
            df["true"], df["pred_threshold"], average="binary"
        )
        running_metrics["f1-micro"][threshold] += metrics.f1_score(
            df["true"], df["pred_threshold"], average="micro"
        )
        running_metrics["f1-macro"][threshold] += metrics.f1_score(
            df["true"], df["pred_threshold"], average="macro"
        )
        running_metrics["f1-weighted"][threshold] += metrics.f1_score(
            df["true"], df["pred_threshold"], average="weighted"
        )

        # PRINT METRICS
        print(
            f"{threshold:<10.1f} {metrics.accuracy_score(df['true'], df['pred_hard']):<10.3f} "
            + f"{metrics.precision_score(df['true'], df['pred_hard']):<10.3f} "
            + f"{metrics.recall_score(df['true'], df['pred_hard']):<10.3f} "
            + f"{metrics.f1_score(df['true'], df['pred_hard'], average='binary'):<10.3f} "
            + f"{metrics.f1_score(df['true'], df['pred_hard'], average='micro'):<10.3f} "
            + f"{metrics.f1_score(df['true'], df['pred_hard'], average='macro'):<10.3f} "
            + f"{metrics.f1_score(df['true'], df['pred_hard'], average='weighted'):<10.3f}"
        )

# COMPUTE AVERAGE METRICS
for metric in running_metrics:
    for threshold in running_metrics[metric]:
        running_metrics[metric][threshold] /= 5  # TODO 5 fold is now hardcoded

# PRINT AVERAGE METRICS
print("\nAverage metrics:")
print(
    f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-binary':<10} {'F1-micro':<10} {'F1-macro':<10} {'F1-weighted':<10}"
)
for threshold in np.arange(0, 1, 0.1):
    print("{:<10.1f} ".format(threshold), end="")
    for metric in running_metrics:
        print(f"{running_metrics[metric][threshold]:<10.3f} ", end="")
    print()

# PLOT ROC CURVES
plt.clf()
for model in rocs:
    fpr, tpr = rocs[model]
    plt.plot(
        fpr,
        tpr,
        ":" if "SkipGram" in model else "-",
        label=f"{model} (AUC = {metrics.auc(fpr, tpr):.2f})",
    )
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc="lower right")
plt.savefig(
    os.path.join(
        predictions_path, f"roc-{args.model}{f'-{args.type}' if args.type else ''}.png"
    )
)


# CONCAT ALL FOLDS
folds = []
for pred_file in os.listdir(predictions_path):
    if not pred_file.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(predictions_path, pred_file))
    folds.append(df)
df = pd.concat(folds)
df["pred_hard"] = df["pred"].round()

# PLOT ROC CURVE
fpr, tpr, _ = metrics.roc_curve(df["true"], df["pred"])
roc_auc = metrics.auc(fpr, tpr)
fig = plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, label=f"ROC fold {pred_file.split('.')[0]} (AUC = {roc_auc:.2f})")
plt.legend(loc="lower right")
plt.savefig(
    os.path.join(
        predictions_path,
        f"roc-{args.model}{f'-{args.type}' if args.type else ''}-concat.png",
    )
)
