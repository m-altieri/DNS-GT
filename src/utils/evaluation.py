import os
import pprint
import pickle
import sklearn
import pandas as pd
from colorama import Fore
import matplotlib.pyplot as plt


class Evaluation:
    def __init__(self, run_path: str = "../runs"):
        self.run_path = run_path
        self.results = None

    def compute_metrics(
        self,
        df: pd.DataFrame,
        plot_save_path: str = None,
        verbose: bool = False,
    ):
        """Compute metrics from a predictions DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame containing predictions. It must have 3
            columns, of which the first is the domain, and the other two are named
            exactly "labels" and "preds", containing the ground truth and the
            prediction for that domain, respectively.

            verbose (bool): whether to print results to stdout in addition to
            returning them.
        """

        results = {}

        # Confusion matrix
        preds = df.filter(regex="^preds-\d+$")
        discrete_preds = preds.apply(
            lambda row: row.index.get_loc(row.idxmax()), axis=1
        )
        confusion_matrix = sklearn.metrics.confusion_matrix(
            df["labels"], discrete_preds
        )
        print("SKlearn Confusion matrix:\n", confusion_matrix)
        print(
            "SKlearn Classification Report:\n",
            sklearn.metrics.classification_report(df["labels"], discrete_preds),
        )

        # results["confusion_matrix"] = {
        #     "tn": confusion_matrix[0, 0],
        #     "fp": confusion_matrix[0, 1],
        #     "fn": confusion_matrix[1, 0],
        #     "tp": confusion_matrix[1, 1],
        # }
        # if verbose:

        # # Precision, recall, F1, accuracy at different thresholds
        # for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        #     preds_hard = (df["preds"] > threshold).astype(int)

        #     precision = sklearn.metrics.precision_score(df["labels"], preds_hard)
        #     recall = sklearn.metrics.recall_score(df["labels"], preds_hard)
        #     f1_score = sklearn.metrics.f1_score(df["labels"], preds_hard)
        #     accuracy = sklearn.metrics.accuracy_score(df["labels"], preds_hard)

        #     results[threshold] = {
        #         "precision": precision,
        #         "recall": recall,
        #         "f1_score": f1_score,
        #         "accuracy": accuracy,
        #     }
        #     if verbose:
        #         print(f"-- {threshold} --")
        #         print(f"Precision: {precision:.3f}")
        #         print(f"Recall: {recall:.3f}")
        #         print(f"F1 score: {f1_score:.3f}")
        #         print(f"Accuracy: {accuracy:.3f}")
        #         print()

        # AUC
        # # take only pred columns from labels that are present
        # print([f"preds-{i}" for i in df["labels"].unique()])
        # if len(preds_columns.columns) != len(df["labels"].unique()):
        #     preds_columns = preds_columns[[f"preds-{i}" for i in df["labels"].unique()]]

        unique_classes = df["labels"].unique()
        unique_classes.sort()

        print(pd.get_dummies(df["labels"]))
        print(preds[[f"preds-{c}" for c in unique_classes]])
        auc = sklearn.metrics.roc_auc_score(
            pd.get_dummies(df["labels"]),
            preds[[f"preds-{c}" for c in unique_classes]],
        )

        results["auc"] = auc
        print(f"AUC: {auc:.3f}")

        # Plot ROC
        # fpr, tpr, _ = sklearn.metrics.roc_curve(df["labels"], preds)
        # if plot_save_path:
        #     plt.figure(figsize=(5, 5))
        #     plt.plot(
        #         fpr,
        #         tpr,
        #         label=f"ROC (AUC = {auc:.3f})",
        #     )
        #     plt.legend(loc="lower right")
        #     plt.savefig(plot_save_path)

        # results["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

        self.results = results

    def save_metrics(self):
        if self.results is None:
            raise ValueError(
                f"{Fore.RED}[ERROR] Results have not been computed yet.{Fore.RESET}"
            )

        # Save results in a human-readable format for quick access
        with open(os.path.join(self.run_path, "results.txt"), "w") as f:
            pprint.pprint(self.results, stream=f, sort_dicts=False)

        # Save results in a binary format for retrieval
        with open(os.path.join(self.run_path, "results.pkl"), "wb") as f:
            pickle.dump(self.results, f)

    @staticmethod
    def merge_rocs(
        models: list[str],
        runs: list[str],
        override_names: list[str] = None,
        runs_path: str = "../runs",
        save_folder: str = "../runs/_extras",
        name: str = "merged_rocs.png",
    ):
        plt.figure(figsize=(5, 5))
        for i, (model, run) in enumerate(zip(models, runs)):
            results_path = os.path.join(runs_path, model, run, "results.pkl")
            with open(results_path, "rb") as f:
                results = pickle.load(f)
            fpr = results["roc"]["fpr"]
            tpr = results["roc"]["tpr"]
            run_label = (
                override_names[i] if override_names is not None else f"{model}-{run}"
            )
            plt.plot(
                fpr,
                tpr,
                label=f"{run_label} (AUC = {results['auc']:.3f})",
            )
        plt.legend(loc="lower right", fontsize="small")

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plt.savefig(os.path.join(save_folder, name))

    @staticmethod
    def merge_rocs_manual(
        runs_path: list[str] = ["../runs/_extras/merged"],
        save_folder: str = "../runs/_extras/merged",
        name: str = "merged_rocs.png",
    ):
        plt.figure(figsize=(5, 5))
        for run_path in runs_path:
            results_path = os.path.join(run_path, "results.pkl")
            with open(results_path, "rb") as f:
                results = pickle.load(f)
            fpr = results["roc"]["fpr"]
            tpr = results["roc"]["tpr"]
            plt.plot(
                fpr,
                tpr,
                label=f"{run_path} (AUC = {results['auc']:.3f})",
            )
        plt.legend(loc="lower right")

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plt.savefig(os.path.join(save_folder, name))

    def merge_preds(
        self,
        models: list[str],
        runs: list[str],
        runs_path: str = "../runs",
        save_folder: str = "../runs/_extras",
        name: str = "merged",
    ):
        # Create a list with the predictions for each run
        preds_csvs = []
        for model, run in zip(models, runs):
            results_path = os.path.join(runs_path, model, run, "predictions")
            preds_csvs.append(
                pd.read_csv(os.path.join(results_path, os.listdir(results_path)[0]))
            )

        # Concatenate all predictions together
        merged = pd.concat([preds_csv for preds_csv in preds_csvs])

        # Save to CSV
        if not os.path.exists(os.path.join(save_folder, name)):
            os.makedirs(os.path.join(save_folder, name))
        merged.to_csv(os.path.join(save_folder, name, "predictions.csv"))

        # Compute metrics on the merged predictions
        self.compute_metrics(
            merged, plot_save_path=os.path.join(save_folder, name, "ROC.png")
        )

        # Save results in a human-readable format for quick access
        with open(os.path.join(save_folder, name, "results.txt"), "w") as f:
            pprint.pprint(self.results, stream=f, sort_dicts=False)

        # Save results in a binary format for retrieval
        with open(os.path.join(save_folder, name, "results.pkl"), "wb") as f:
            pickle.dump(self.results, f)

    @staticmethod
    def collect_preds(
        models: list[str],
        runs: list[str],
        runs_path: str = "../runs",
        save_folder: str = "../runs/_extras",
        name: str = "collected",
    ):
        # Create a list with the predictions for each run
        preds = None
        collected_results = {
            "model": [],
            "auc": [],
            "f1": [],
            "accuracy": [],
            "f1_best": [],
            "accuracy_best": [],
        }
        for model, run in zip(models, runs):
            preds_path = os.path.join(runs_path, model, run, "predictions")
            df = pd.read_csv(os.path.join(preds_path, os.listdir(preds_path)[0]))
            if preds is None:
                preds = df
            else:
                preds.merge(df, on="domains")

            results_path = os.path.join(runs_path, model, run, "results.pkl")
            with open(results_path, "rb") as f:
                results = pickle.load(f)
                collected_results["model"].append(f"{model}-{run}")
                collected_results["auc"].append(results["auc"])
                collected_results["f1"].append(results[0.5]["f1_score"])
                collected_results["accuracy"].append(results[0.5]["accuracy"])

                thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                per_threshold_f1 = [
                    results[threshold]["f1_score"] for threshold in thresholds
                ]
                best_threshold = thresholds[
                    per_threshold_f1.index(max(per_threshold_f1))
                ]
                collected_results["f1_best"].append(results[best_threshold]["f1_score"])
                collected_results["accuracy_best"].append(
                    results[best_threshold]["accuracy"]
                )

            print(
                f"Added preds for {model}/{run} of shape {df.shape}, new shape {preds.shape}"
            )

        print(preds)

        if not os.path.exists(os.path.join(runs_path, save_folder, name)):
            os.makedirs(os.path.join(runs_path, save_folder, name))

        preds.to_csv(os.path.join(runs_path, save_folder, name, "preds.csv"))

        collected_results = pd.DataFrame(collected_results)
        collected_results.to_csv(
            os.path.join(runs_path, save_folder, name, "results.csv")
        )
