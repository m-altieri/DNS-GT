#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import pandas as pd
from colorama import Style
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)

# Binary models
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Oneclass models
from pyod.models.vae import VAE
from pyod.models.abod import ABOD
from pyod.models.hbos import HBOS
from pyod.models.suod import SUOD
from pyod.models.copod import COPOD

from utils.evaluation import Evaluation

# os.environ["OPENBLAS_NUM_THREADS"] = "1"


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "model", action="store", help="Model to retrieve the learned embeddings from."
    )
    argparser.add_argument(
        "run",
        action="store",
        help="Name of the run to retrieve the learned embeddings from. "
        + "The file embeddings.npy must be present in the run folder. ",
    )
    argparser.add_argument(
        "-b",
        "--balanced",
        action="store_true",
        help="Whether to balance the positive and negative training classes, "
        + "so that there are the same amount of examples in each class. "
        + "False by default. The test classes are never balanced.",
    )
    argparser.add_argument(
        "--category",
        action="store",
        default="any",
        choices=[
            "advertising",
            "malicious",
            "suspicious",
            "tracking",
            "other",
            "any",
        ],
        help="Types of blacklists to use to provide labels for domain names. "
        + "If --category any (default), all blacklist types are used and merged.",
    )
    argparser.add_argument(
        "--q",
        action="store",
        default="good",
        choices=["good", "ok"],
        help="Choose the quality of the domain labels to use for classification. "
        + "Quality, in this context, refers to what and how many blacklists are used to generate those labels. "
        + "With --q good (default), only a subset of highly rated blacklists are used, "
        + "with --q ok, all available blacklists are used.",
    )

    # DEPRECATED
    # argparser.add_argument(
    #     "--max-tokens",
    #     action="store",
    #     type=int,
    #     help="Truncate to a certain number of embeddings for evaluation.",
    # )

    args = argparser.parse_args()
    return args


def main():
    args = parse_args()

    # Load embeddings (they act as feature vectors)
    emb_path = os.path.join("../runs", args.model, args.run, "embeddings.npy")
    embs = np.load(emb_path)

    # Remove "[UNK]" (first index in the embeddings, not in the vocabulary), and "" (for some reason, last index in the embeddings)
    embs = embs[1:-1]
    print("Loaded embs with shape:", embs.shape)

    # Load labels
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

    # Load vocabulary
    vocab_path = "/mnt/storage15/TI-2016/npy/tokenized/trivial/domain_vocab.txt"
    with open(vocab_path, "r") as f:
        vocab = [l.strip() for l in f.readlines()]

    # TODO It should not be used anymore. Always evaluate all learned embeddings
    # # if --max-tokens, truncate the vocabulary
    # if args.max_tokens:
    #     vocab = vocab[: args.max_tokens]
    #     print(f"Truncated vocab to first {args.max_tokens} tokens.")

    # check that embeddings and vocabulary have the same length
    if len(embs) != len(vocab):
        raise AssertionError(
            f"Embeddings and vocabulary should have the same length, but have {len(embs)} and {len(vocab)}"
        )

    # Only use labels for domains in embs (i.e. in the vocabulary)
    labels = labels[labels["domain"].isin(vocab)]
    labels = labels.reset_index()

    # Sort labels to match vocabs
    labels = labels.set_index("domain").reindex(vocab)

    # Set NaN values to 0 (non-malicious)
    labels = labels.fillna(0)

    # Select target category
    labels = labels[args.category, args.q]

    X = np.array(embs)
    y = np.array(labels)

    # Models, metrics, results initialization
    Models = {
        RandomForestClassifier: {"conf": {"oneclass": False}},
        SVC: {"conf": {"oneclass": False}, "kwargs": {"probability": True}},
        AdaBoostClassifier: {"conf": {"oneclass": False}},
        GaussianNB: {"conf": {"oneclass": False}},
        DecisionTreeClassifier: {"conf": {"oneclass": False}},
        ABOD: {"conf": {"oneclass": True}, "kwargs": {"contamination": 0.2}},
        HBOS: {"conf": {"oneclass": True}, "kwargs": {"contamination": 0.2}},
        VAE: {"conf": {"oneclass": True}, "kwargs": {"contamination": 0.2}},
        COPOD: {"conf": {"oneclass": True}, "kwargs": {"contamination": 0.2}},
        SUOD: {"conf": {"oneclass": True}, "kwargs": {"contamination": 0.2}},
    }
    # for oneclass models, ineligible metrics will be ignored
    metrics = [
        "f1-weighted",
        "f1-macro",
        "f1-micro",
        "acc",
        "auc",
    ]
    results = {Model.__name__: {} for Model in Models}

    # Load test fold
    test_domains = np.load(
        "/mnt/storage15/TI-2016/npy/tokenized/trivial/folds/partition-0/fold-0.npy"
    )
    test_indexes = np.sort(np.where(np.expand_dims(test_domains, axis=-1) == vocab)[1])

    train_X = np.delete(X, test_indexes, axis=0)
    train_y = np.delete(y, test_indexes, axis=0)
    test_X = X[test_indexes]
    test_y = y[test_indexes]

    # If --balanced, adjust label ratio of the training samples
    if args.balanced:
        print("Balancing train...")
        balanced_idx = get_balanced_indices(train_y)
        train_X = train_X[balanced_idx]
        train_y = train_y[balanced_idx]

    # Select only negative examples for oneclass models
    negative_idx = np.where(train_y == 0)[0]
    oneclass_train_X = train_X[negative_idx]

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
                "labels": test_y,
                "preds": probs,
                "pred_hard": preds,
            }
        )
        predictions_path = f"../runs/{args.model}+{Model.__name__}/"
        if not os.path.exists(
            os.path.join(predictions_path, args.model, "predictions")
        ):
            os.makedirs(os.path.join(predictions_path, args.model, "predictions"))
        df.to_csv(
            os.path.join(
                predictions_path, args.model, "predictions", f"preds-fold0.csv"
            )
        )

        # Evaluate NEW
        evaluation = Evaluation(os.path.join(predictions_path, args.model))
        evaluation.compute_metrics(
            df,
            plot_save_path=os.path.join(
                os.path.join(predictions_path, args.model),
                f"roc-{args.model}.png",
            ),
            verbose=True,
        )
        evaluation.save_metrics()

        print(evaluation.results)

    #     # Evaluate OLD  # TODO deprecate
    #     scores = model.evaluate(test_y, preds, probs, metrics)
    #     results[Model.__name__] = {
    #         metric: results[Model.__name__].get(metric, 0.0) + scores[metric]
    #         for metric in scores
    #     }
    #     print(scores)

    # # TODO old evaluation, deprecate
    # for Model in Models:
    #     results[Model.__name__] = {
    #         metric: results[Model.__name__][metric]
    #         for metric in results[Model.__name__]
    #     }

    # print(f"\n{Style.BRIGHT}Results:{Style.RESET_ALL}")
    # print(json.dumps(results, indent=4))


def get_balanced_indices(labels):
    if len(np.where(labels == 1)[0]) > len(np.where(labels == 0)[0]):
        # TODO implement the case where there are more positive than negative samples
        raise NotImplementedError(
            "For now, negative samples must be more than positive ones"
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
