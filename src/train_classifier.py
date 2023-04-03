import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
import argparse
import os
import pandas as pd
from colorama import Style


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("emb_file", action="store")
    argparser.add_argument("--balanced", action="store_true")
    argparser.add_argument(
        "--category",
        action="store",
        default="any",
        choices=["advertising", "malicious", "suspicious", "tracking", "other", "any"],
    )
    argparser.add_argument("--q", action="store", default="ok", choices=["good", "ok"])
    return argparser.parse_args()


def main():
    args = parse_args()

    # Get embeddings
    # Embeddings are input samples (they act as feature vectors)
    emb_path = os.path.join("embeddings", args.emb_file)
    embs = np.load(emb_path)
    print("Loaded embs with shape:", embs.shape)

    # Get data
    # Target is the label (1 for blacklisted, 0 for not-blacklisted)
    labels = pd.read_csv(
        os.path.join("scripts", "labels.csv"), index_col=0, header=[0, 1]
    )
    labels.columns = pd.MultiIndex.from_tuples(
        [
            ("domain", ""),
            ("advertising", "good"),
            ("advertising", "ok"),
            ("malicious", "good"),
            ("malicious", "ok"),
            ("suspicious", "good"),
            ("suspicious", "ok"),
            ("tracking", "good"),
            ("tracking", "ok"),
            ("other", "good"),
            ("other", "ok"),
            ("any", "good"),
            ("any", "ok"),
        ]
    )
    print(labels)

    # Get vocabulary
    vocab_path = os.path.join("preprocessing", "vocabs", "small", "domains_vocab.txt")
    with open(vocab_path, "r") as f:
        vocab = [l.strip() for l in f.readlines()]

    # Only use labels for domains in embs (i.e. in the vocabulary)
    labels = labels[labels["domain"].isin(vocab)]
    print(labels)

    labels = labels.reset_index()
    print(labels)

    labels = labels[args.category, args.q]
    print(labels)

    X = np.array(embs)
    y = np.array(labels)
    print(X)
    print(y)

    if args.balanced:
        positives = len(np.argwhere(y == 1)[:, 0])
        print(np.argwhere(y == 0))
        print(np.argwhere(y == 0)[:, 0])
        neg_idx = np.random.choice(np.argwhere(y == 0)[:, 0], positives, replace=False)
        pos_idx = np.argwhere(y == 1)[:, 0]
        idx = np.sort(np.concatenate((neg_idx, pos_idx)))
        print(idx)
        X = X[idx]
        y = y[idx]
        print(
            f"Balanced: {len(pos_idx)} positives, {len(neg_idx)} negatives; X: {X.shape}"
        )

    # Cross validation
    nsamples = len(X)  # len(embs) > len(X) if --balanced
    folds = 10
    RF_avg = 0
    SVC_avg = 0
    for i, fold in enumerate(range(folds)):
        train_X = np.concatenate(
            (X[: nsamples // folds * i], X[nsamples // folds * (i + 1) :])
        )
        train_y = np.concatenate(
            (y[: nsamples // folds * i], y[nsamples // folds * (i + 1) :])
        )
        test_X = X[nsamples // folds * i : nsamples // folds * (i + 1)]
        test_y = y[nsamples // folds * i : nsamples // folds * (i + 1)]

        print(f"\n{Style.BRIGHT}Evaluating on fold {i+1}/{folds}{Style.RESET_ALL}")
        print("Evaluating RF...")
        RF_score, RF_CM = evaluate_RF(train_X, train_y, test_X, test_y)
        RF_avg += RF_score
        print(f"F1: {RF_score:.3f}")
        print(RF_CM)

        print("Evaluating SVC...")
        SVC_score, SVC_CM = evaluate_SVC(train_X, train_y, test_X, test_y)
        SVC_avg += SVC_score
        print(f"F1: {SVC_score:.3f}")
        print(SVC_CM)

    print(f"RF average score: {RF_avg/folds:.3f}")
    print(f"SVC average score: {SVC_avg/folds:.3f}")


def evaluate_RF(train_X, train_y, test_X, test_y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    return f1_score(test_y, preds), confusion_matrix(test_y, preds)


def evaluate_SVC(train_X, train_y, test_X, test_y):
    model = SVC()
    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    return f1_score(test_y, preds), confusion_matrix(test_y, preds)


if __name__ == "__main__":
    main()
