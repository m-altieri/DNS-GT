import os
import json
import numpy as np
import pandas as pd
import argparse
from colorama import Style
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score

# Binary
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Oneclass
from pyod.models.abod import ABOD
from pyod.models.hbos import HBOS
from pyod.models.vae import VAE
from pyod.models.copod import COPOD
from pyod.models.suod import SUOD


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
    args = argparser.parse_args()
    return args


def main():
    args = parse_args()

    # Get embeddings
    # Embeddings are input samples (they act as feature vectors)
    emb_path = os.path.join("../embeddings", args.emb_file)
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
    # print(labels)

    # Get vocabulary
    vocab_path = os.path.join("../data", "vocabs", "all", "exp", "domains_vocab.txt")
    with open(vocab_path, "r") as f:
        vocab = [l.strip() for l in f.readlines()]

    if args.max_tokens:
        vocab = vocab[: args.max_tokens]  # if --max-tokens, truncate the vocab
        print(f"Truncated vocab to first {args.max_tokens} tokens.")
    if len(embs) != len(vocab):
        raise AssertionError(
            f"Embeddings and vocabulary should have the same length, but have {len(embs)} and {len(vocab)}"
        )
    print(len(embs))
    print(len(vocab))

    # Only use labels for domains in embs (i.e. in the vocabulary)
    labels = labels[labels["domain"].isin(vocab)]
    labels = labels.reset_index()
    # print(labels)

    # Sort labels to match vocabs
    labels = labels.set_index("domain").reindex(vocab)
    print(labels)

    # Select target category
    labels = labels[args.category, args.q]
    # print(labels)

    X = np.array(embs)
    y = np.array(labels)

    # Models, metrics, results initialization
    Models = {
        RandomForestClassifier: {"conf": {"oneclass": False}},
        SVC: {"conf": {"oneclass": False}, "kwargs": {"probability": True}},
        AdaBoostClassifier: {"conf": {"oneclass": False}},
        GaussianNB: {"conf": {"oneclass": False}},
        DecisionTreeClassifier: {"conf": {"oneclass": False}},
        # ABOD: {"conf": {"oneclass": True}, "kwargs": {"contamination": 0.2}},
        # HBOS: {"conf": {"oneclass": True}, "kwargs": {"contamination": 0.2}},
        # VAE: {"conf": {"oneclass": True}, "kwargs": {"contamination": 0.2}},
        # COPOD: {"conf": {"oneclass": True}, "kwargs": {"contamination": 0.2}},
        # SUOD: {"conf": {"oneclass": True}, "kwargs": {"contamination": 0.2}},
    }
    metrics = [
        "f1-weighted",
        "f1-macro",
        "f1-micro",
        "acc",
        "auc",
    ]  # for oneclass models, ineligible metrics will be ignored
    results = {Model.__name__: {} for Model in Models}

    # Get indexes of domains in test fold
    test_domains = np.load("../data/vocabs/small/exp/test_folds5/fold-2.npy")
    test_indexes = np.sort(np.where(np.expand_dims(test_domains, axis=-1) == vocab)[1])

    train_X = np.delete(X, test_indexes, axis=0)
    train_y = np.delete(y, test_indexes, axis=0)
    test_X = X[test_indexes]
    test_y = y[test_indexes]
    print(len(train_X))
    print(len(test_X))

    # Adjust label ratio of the training samples according to --label-ratio
    if args.balanced:
        balanced_idx = get_balanced_indices(train_y)
        train_X = train_X[balanced_idx]
        train_y = train_y[balanced_idx]

    # Select only negative examples for oneclass models
    negative_idx = np.where(train_y == 0)[0]
    oneclass_train_X = train_X[negative_idx]

    # # TODO now I'm ALWAYS balancing the test fold, is it correct?
    # balanced_idx = get_balanced_indices(test_y)
    # test_X = test_X[balanced_idx]
    # test_y = test_y[balanced_idx]

    print(
        f"\n{Style.BRIGHT}Evaluating "
        + f"{Style.DIM}(pos/neg ratio: "
        + f"train {len(np.where(train_y==1)[0])}/{len(np.where(train_y==0)[0])}, "
        + f"test {len(np.where(test_y==1)[0])}/{len(np.where(test_y==0)[0])}){Style.RESET_ALL}"
    )

    for Model in Models:
        print(f"\nEvaluating {Model.__name__}...")
        model = ModelWrapper(
            Model,
            Models[Model].get("conf"),
            *Models[Model].get("args", []),
            **Models[Model].get("kwargs", {}),
        )
        model.fit(
            X=oneclass_train_X if Models[Model]["conf"].get("oneclass") else train_X,
            y=None if Models[Model]["conf"].get("oneclass") else train_y,
        )

        # Predict and save preds
        preds, probs = model.predict(test_X)  # it gives me the prob for the two classes
        probs = probs[:, 1]

        df = pd.DataFrame(
            {
                "domains": np.array(vocab)[test_indexes],
                "true": test_y,
                "pred": probs,
                "pred_hard": preds,
            }
        )
        predictions_path = f"../predictions/{Model.__name__}/"
        if not os.path.exists(predictions_path):
            os.makedirs(predictions_path)
        df.to_csv(os.path.join(predictions_path, f"preds-fold2.csv"))

        # Evaluate
        scores = model.evaluate(test_y, preds, probs, metrics)
        results[Model.__name__] = {
            metric: results[Model.__name__].get(metric, 0.0) + scores[metric]
            for metric in scores
        }
        print(scores)

    for Model in Models:
        results[Model.__name__] = {
            metric: results[Model.__name__][metric]
            for metric in results[Model.__name__]
        }
    print(f"\n{Style.BRIGHT}Results:{Style.RESET_ALL}")
    print(json.dumps(results, indent=4))


def get_balanced_indices(labels):
    if len(np.where(labels == 1)[0]) > len(np.where(labels == 0)[0]):
        raise NotImplementedError(
            "For now, negative samples must be more than positive ones"  # TODO
        )
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.random.choice(np.where(labels == 0)[0], len(pos_idx), replace=False)
    return np.sort(np.concatenate((neg_idx, pos_idx)))


class ModelWrapper:
    def __init__(self, Model, conf, *args, **kwargs):
        super(ModelWrapper, self).__init__()
        self.Model = Model
        self.conf = conf
        self.args = args
        self.kwargs = kwargs
        self.model = Model(*args, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        pred = self.model.predict(X)
        probs = self.model.predict_proba(X)
        # if (
        #     self.Model == OneClassSVM
        #     or self.Model == IsolationForest
        #     or self.Model == LocalOutlierFactor
        # ):
        #     pred = np.where(pred == 1, np.full_like(pred, 0), pred)
        #     pred = np.where(pred == -1, np.full_like(pred, 1), pred)
        return pred, probs

    def evaluate(self, test_y, preds, probs, metrics):
        print(confusion_matrix(test_y, preds))
        scores = {}
        for m in metrics:
            if m == "f1-micro":
                scores[m] = f1_score(test_y, preds, average="micro")
            if m == "f1-macro":
                scores[m] = f1_score(test_y, preds, average="macro")
            if m == "f1-weighted":
                scores[m] = f1_score(test_y, preds, average="weighted")
            if m == "acc":
                scores[m] = accuracy_score(test_y, preds)
            if m == "auc":
                scores[m] = roc_auc_score(test_y, probs)
        return scores


if __name__ == "__main__":
    main()
