import os
import json
import numpy as np
import pandas as pd
import argparse
from colorama import Style
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# Binary
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Oneclass
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.abod import ABOD
from pyod.models.abod import ABOD
from pyod.models.ecod import ECOD

# from pyod.models.suod import SUOD

# from pyod.models.cblof import CBLOF

# # from pyod.models.feature_bagging import FeatureBagging
# from pyod.models.hbos import HBOS
# from pyod.models.iforest import IForest
# from pyod.models.knn import KNN
# from pyod.models.lof import LOF
# from pyod.models.mcd import MCD
# from pyod.models.ocsvm import OCSVM
# from pyod.models.pca import PCA
# from pyod.models.lscp import LSCP
# from pyod.models.inne import INNE
# from pyod.models.gmm import GMM
# from pyod.models.kde import KDE
# from pyod.models.lmdd import LMDD


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("emb_file", action="store")
    argparser.add_argument("-b", "--balanced", action="store_true")
    argparser.add_argument(
        "--category",
        action="store",
        default="any",
        choices=["advertising", "malicious", "suspicious", "tracking", "other", "any"],
    )
    argparser.add_argument("--q", action="store", default="ok", choices=["good", "ok"])
    argparser.add_argument("--max-tokens", action="store", type=int)
    return argparser.parse_args()


def main():
    args = parse_args()

    # Get embeddings
    # Embeddings are input samples (they act as feature vectors)
    emb_path = os.path.join("embeddings", args.emb_file)
    embs = np.load(emb_path)
    embs = embs[
        1:
    ]  # TODO IMPORTANT! This is a temporary fix for the duplication of the UNK token. Actually take time to fix this issue by removing UNK from the vocab generating script and using num_oov_indices=1
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
    vocab_path = os.path.join(
        "preprocessing", "vocabs", "small", "exp", "domains_vocab.txt"
    )
    with open(vocab_path, "r") as f:
        vocab = [l.strip() for l in f.readlines()]

    if args.max_tokens:
        vocab = vocab[: args.max_tokens]  # if --max-tokens, truncate the vocab
        print(f"Truncated vocab to first {args.max_tokens} tokens.")
    assert len(embs) == len(vocab)

    # Only use labels for domains in embs (i.e. in the vocabulary)
    labels = labels[labels["domain"].isin(vocab)]
    labels = labels.reset_index()
    print(labels)

    # Select target category
    labels = labels[args.category, args.q]
    print(labels)

    X = np.array(embs)
    y = np.array(labels)

    # Models, metrics, results initialization
    Models = {
        RandomForestClassifier: {"oneclass": False},
        SVC: {"oneclass": False},
        AdaBoostClassifier: {"oneclass": False},
        RidgeClassifier: {"oneclass": False},
        GaussianNB: {"oneclass": False},
        DecisionTreeClassifier: {"oneclass": False},
        ABOD: {"oneclass": True},
        OneClassSVM: {"oneclass": True},
        IsolationForest: {"oneclass": True},
        LocalOutlierFactor: {
            "oneclass": True,
            "args": [],
            "kwargs": {"contamination": 0.3, "novelty": True},
        },
    }
    metrics = ["f1", "acc"]  # for oneclass models, ineligible metrics will be ignored
    results = {Model.__name__: {} for Model in Models}

    # Cross validation
    nsamples = len(X)
    folds = 10
    for i, fold in enumerate(range(folds)):
        train_X = np.concatenate(
            (X[: nsamples // folds * i], X[nsamples // folds * (i + 1) :])
        )
        train_y = np.concatenate(
            (y[: nsamples // folds * i], y[nsamples // folds * (i + 1) :])
        )
        test_X = X[nsamples // folds * i : nsamples // folds * (i + 1)]
        test_y = y[nsamples // folds * i : nsamples // folds * (i + 1)]

        # Adjust label ratio of the training samples according to --label-ratio
        if args.balanced:
            idx = get_balanced_indices(train_y)
            train_X = train_X[idx]
            train_y = train_y[idx]

        idx = np.where(train_y == 0)[0]
        oneclass_train_X = train_X[idx]

        # TODO now I'm ALWAYS balancing the test fold, is it correct?
        idx = get_balanced_indices(test_y)
        test_X = test_X[idx]
        test_y = test_y[idx]

        print(
            f"\n{Style.BRIGHT}Evaluating on fold {i+1}/{folds} "
            + f"{Style.DIM}(pos/neg ratio: "
            + f"train {len(np.where(train_y==1)[0])}/{len(np.where(train_y==0)[0])}, "
            + f"test {len(np.where(test_y==1)[0])}/{len(np.where(test_y==0)[0])}){Style.RESET_ALL}"
        )

        for Model in Models:
            print(f"Evaluating {Model.__name__}...")
            # model = build_model(Model)
            model = ModelWrapper(
                Model, *Models[Model].get("args", []), **Models[Model].get("kwargs", {})
            )
            model.fit(
                X=oneclass_train_X if Models[Model]["oneclass"] else train_X,
                y=None if Models[Model]["oneclass"] else train_y,
            )
            preds = model.predict(test_X)
            scores = model.evaluate(test_y, preds, metrics)
            results[Model.__name__] = {
                metric: results[Model.__name__].get(metric, 0.0) + scores[metric]
                for metric in scores
            }
            print(scores)

    for Model in Models:
        results[Model.__name__] = {
            metric: results[Model.__name__][metric] / 10
            for metric in results[Model.__name__]
        }
    print(f"\n{Style.BRIGHT}Results:{Style.RESET_ALL}")
    print(json.dumps(results, indent=4))


def get_balanced_indices(labels):
    if len(np.where(labels == 1)[0]) > len(np.where(labels == 0)[0]):
        raise ValueError(
            "For now, negative samples must be more than positive ones"  # TODO
        )
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.random.choice(np.where(labels == 0)[0], len(pos_idx), replace=False)
    return np.sort(np.concatenate((neg_idx, pos_idx)))


class ModelWrapper:
    def __init__(self, Model, *args, **kwargs):
        super(ModelWrapper, self).__init__()
        self.Model = Model
        self.args = args
        self.kwargs = kwargs
        self.model = Model(*args, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        pred = self.model.predict(X)
        if (
            self.Model == OneClassSVM
            or self.Model == IsolationForest
            or self.Model == LocalOutlierFactor
        ):
            pred = np.where(pred == 1, np.full_like(pred, 0), pred)
            pred = np.where(pred == -1, np.full_like(pred, 1), pred)
        return pred

    def evaluate(self, test_y, preds, metrics):
        print(confusion_matrix(test_y, preds))
        scores = {}
        for m in metrics:
            if m == "f1" and not self.kwargs.get("oneclass"):
                scores[m] = f1_score(test_y, preds)
            if m == "acc":
                scores[m] = accuracy_score(test_y, preds)
        return scores


if __name__ == "__main__":
    main()
