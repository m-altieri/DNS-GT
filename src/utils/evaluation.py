import os
import json
import pprint
import pickle
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Style
import sklearn.metrics
from pytftk.formatting import indent


class Evaluation:
    def __init__(self, run_path: str = "../runs"):
        self.run_path = run_path
        self.results = None

    @staticmethod
    def _normalize_to_stochastic(x):
        """Make each row of x to add up to 1.

        Args:
            x (np.array): a 2-dimensional array.

        Returns:
            np.array: the same 2-dimensional array where each row adds up to 1.
        """
        return x / np.expand_dims(np.sum(x, axis=1), axis=-1)

    def compute_metrics(
        self,
        df: pd.DataFrame,
        plot_save_path: str = None,
        n_classes: int = None,
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
        print(df)
        results = {}

        labels = df["labels"]
        unique_labels = np.unique(labels)

        preds = df.filter(regex="^preds-\d+$")
        discrete_preds = preds.apply(
            lambda row: row.index.get_loc(row.idxmax()), axis=1
        )

        class_ids = (
            np.sort(discrete_preds.unique())
            if n_classes is None
            else list(range(n_classes))
        )

        print(
            "SKlearn Classification Report:\n",
            sklearn.metrics.classification_report(labels, discrete_preds),
        )

        confusion_matrix = sklearn.metrics.confusion_matrix(
            labels, discrete_preds, labels=class_ids
        )
        print("SKlearn Confusion matrix:\n", confusion_matrix)

        precision_classwise = sklearn.metrics.precision_score(
            labels, discrete_preds, labels=class_ids, average=None
        )
        precision_micro = sklearn.metrics.precision_score(
            labels, discrete_preds, labels=class_ids, average="micro"
        )
        precision_macro = sklearn.metrics.precision_score(
            labels, discrete_preds, labels=class_ids, average="macro"
        )
        precision_weighted = sklearn.metrics.precision_score(
            labels, discrete_preds, labels=class_ids, average="weighted"
        )
        recall_classwise = sklearn.metrics.recall_score(
            labels, discrete_preds, labels=class_ids, average=None
        )
        recall_micro = sklearn.metrics.recall_score(
            labels, discrete_preds, labels=class_ids, average="micro"
        )
        recall_macro = sklearn.metrics.recall_score(
            labels, discrete_preds, labels=class_ids, average="macro"
        )
        recall_weighted = sklearn.metrics.recall_score(
            labels, discrete_preds, labels=class_ids, average="weighted"
        )
        f1_classwise = sklearn.metrics.f1_score(
            labels, discrete_preds, labels=class_ids, average=None
        )
        f1_micro = sklearn.metrics.f1_score(
            labels, discrete_preds, labels=class_ids, average="micro"
        )
        f1_macro = sklearn.metrics.f1_score(
            labels, discrete_preds, labels=class_ids, average="macro"
        )
        f1_weighted = sklearn.metrics.f1_score(
            labels, discrete_preds, labels=class_ids, average="weighted"
        )
        accuracy = sklearn.metrics.balanced_accuracy_score(labels, discrete_preds)

        # if unique_labels != class_labels, some prediction scores will be
        # removed, making the prediction row not stochastic. here we normalize
        # the prediction of each instance to make it stochastic again
        preds_present_labels_only = preds[
            [f"preds-{c}" for c in unique_labels]
        ].to_numpy()
        preds_present_labels_only = __class__._normalize_to_stochastic(
            preds_present_labels_only
        )
        auc = sklearn.metrics.roc_auc_score(
            labels,
            preds_present_labels_only,
            multi_class="ovo",
        )

        results["confusion_matrix"] = confusion_matrix.tolist()
        results[0.5] = {
            "precision_classwise": precision_classwise.tolist(),
            "precision_micro": precision_micro,
            "precision_macro": precision_macro,
            "precision_weighted": precision_weighted,
            "recall_classwise": recall_classwise.tolist(),
            "recall_micro": recall_micro,
            "recall_macro": recall_macro,
            "recall_weighted": recall_weighted,
            "f1_classwise": f1_classwise.tolist(),
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "accuracy": accuracy,
        }
        results["auc"] = auc
        self.results = results

        # unique_classes = df["labels"].unique()
        # unique_classes.sort()

        # for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        #     preds_hard = (df["preds"] > threshold).astype(int)

        #     precision = sklearn.metrics.precision_score(
        #         df["labels"], preds_hard, average=None
        #     )
        #     recall = sklearn.metrics.recall_score(df["labels"], preds_hard)
        #     f1 = sklearn.metrics.f1_score(df["labels"], preds_hard)
        #     accuracy = sklearn.metrics.accuracy_score(df["labels"], preds_hard)

        #     results[threshold] = {
        #         "precision": precision,
        #         "recall": recall,
        #         "f1": f1,
        #         "accuracy": accuracy,
        # }
        #     if verbose:
        #         print(f"-- {threshold} --")
        #         print(f"Precision: {precision:.3f}")
        #         print(f"Recall: {recall:.3f}")
        #         print(f"F1 score: {f1:.3f}")
        #         print(f"Accuracy: {accuracy:.3f}")
        #         print()

        # AUC
        # # take only pred columns from labels that are present
        # print([f"preds-{i}" for i in df["labels"].unique()])
        # if len(preds_columns.columns) != len(df["labels"].unique()):
        #     preds_columns = preds_columns[[f"preds-{i}" for i in df["labels"].unique()]]

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

    def save_metrics(self):
        if self.results is None:
            raise ValueError(
                f"{Fore.RED}[ERROR] Results have not been computed yet.{Fore.RESET}"
            )

        with open(os.path.join(self.run_path, "results.json"), "w") as f:
            json.dump(self.results, f, indent=4)

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
    def collect_results(
        runs_path: str = "../runs",
        output_path: str = "../runs/results.csv",
        verbose: bool = False,
    ):
        """Computes results for all runs in runs_path.
        The runs folder is assumed to have structure:
        runs > {model} > {run} > predictions > {csv files}.
        It outputs a single csv file with results for all runs.

        Args:
            runs_path (str): path of the runs folder.
            output_path (str): path of the output csv file (including file name).
            verbose(bool): print debugging information. Defaults to False.

        Raises:
            ValueError: if runs_path doesn't exist or output_path's parent folder
            doesn't exist.
        """

        # check for non-existant input paths
        if not os.path.exists(runs_path):
            raise ValueError(f"Provided runs folder {runs_path} does not exist.")

        # create a list with the predictions for each run
        # preds = None
        collected_results = {}

        for model in os.listdir(runs_path):

            # skip non-directories, such as the default.yaml file
            if not os.path.isdir(os.path.join(runs_path, model)):
                if verbose:
                    print(
                        f"{Fore.RED}{indent(1)}Skipping model {Style.BRIGHT}{model}{Style.NORMAL}: \
it's not a model (not a folder).{Fore.RESET}"
                    )
                continue
            if verbose:
                print(
                    f"\n{indent(1)}Collecting results for model {Style.BRIGHT}{model}{Style.NORMAL}..."
                )

            for run in os.listdir(os.path.join(runs_path, model)):

                # skip non-directories, such as the default.yaml or .gitignore file
                if not os.path.isdir(os.path.join(runs_path, model, run)):
                    if verbose:
                        print(
                            f"{Fore.RED}{indent(2)}Skipping run {Style.BRIGHT}{run}{Style.NORMAL}: \
it's not a run (not a folder).{Fore.RESET}"
                        )
                    continue
                if verbose:
                    print(
                        f"{indent(2)}Collecting results for run {Style.BRIGHT}{run}{Style.NORMAL}..."
                    )

                # skip this run if there are no predictions, for example if it's
                # a pretraining run
                predictions_path = os.path.join(runs_path, model, run, "predictions")
                if not os.path.exists(predictions_path):
                    if verbose:
                        print(
                            f"{Fore.RED}{indent(2)}Skipping run {Style.BRIGHT}{run}{Style.NORMAL}: \
there are no predictions; it might be a pretraining run.{Fore.RESET}"
                        )
                    continue

                # <--- PREDICTIONS --- forse non c'è bisogno di collezionarle
                # prediction_files = os.listdir(predictions_path)
                # for prediction_file in prediction_files:
                #     # get partition and fold from the csv file name
                #     try:
                #         partition = re.findall(
                #             "predictions-partition(\d+)-fold\d+\.csv", prediction_file
                #         )[0]
                #         fold = re.findall(
                #             "predictions-partition\d+-fold(\d+)\.csv", prediction_file
                #         )[0]
                #     except:
                #         print(
                #             f"Error while extracting partition or fold from \
                #             prediction file {prediction_file}"
                #         )

                #     df = pd.read_csv(os.path.join(predictions_path, prediction_file))
                #     if preds is None:
                #         preds = df
                #     else:
                #         preds.merge(df, on="domains")
                # --->

                results_path = os.path.join(runs_path, model, run, "results.pkl")
                if not os.path.exists(results_path):  # skip if there are no results
                    if verbose:
                        print(
                            f"{Fore.RED}{indent(2)}Skipping run {Style.BRIGHT}{run}{Style.NORMAL}: \
there are no results; it might be a pretraining run.{Fore.RESET}"
                        )
                    continue

                with open(results_path, "rb") as f:
                    results = pickle.load(f)
                    print(results)

                    if "model" not in collected_results:
                        collected_results["model"] = []
                    collected_results["model"].append(model)

                    if "run" not in collected_results:
                        collected_results["run"] = []
                    collected_results["run"].append(run)

                    # store class-wise precision, recall and f1
                    if type(results[0.5]["precision_classwise"]) != list:
                        results[0.5]["precision_classwise"] = [
                            results[0.5]["precision_classwise"]
                        ]
                    for c, class_precision in enumerate(
                        results[0.5]["precision_classwise"]
                    ):
                        if f"precision-{c}" not in collected_results:
                            collected_results[f"precision-{c}"] = []
                        collected_results[f"precision-{c}"].append(class_precision)

                    if type(results[0.5]["recall_classwise"]) != list:
                        results[0.5]["recall_classwise"] = [
                            results[0.5]["recall_classwise"]
                        ]
                    for c, class_recall in enumerate(results[0.5]["recall_classwise"]):
                        if f"recall-{c}" not in collected_results:
                            collected_results[f"recall-{c}"] = []
                        collected_results[f"recall-{c}"].append(class_recall)

                    if type(results[0.5]["f1_classwise"]) != list:
                        results[0.5]["f1_classwise"] = [results[0.5]["f1_classwise"]]
                    for c, class_f1score in enumerate(results[0.5]["f1_classwise"]):
                        if f"f1-{c}" not in collected_results:
                            collected_results[f"f1-{c}"] = []
                        collected_results[f"f1-{c}"].append(class_f1score)

                    # store aggregated precision, recall and f1
                    if "precision_micro" not in collected_results:
                        collected_results["precision_micro"] = []
                    collected_results["precision_micro"].append(
                        results[0.5]["precision_micro"]
                    )

                    if "precision_macro" not in collected_results:
                        collected_results["precision_macro"] = []
                    collected_results["precision_macro"].append(
                        results[0.5]["precision_macro"]
                    )

                    if "precision_weighted" not in collected_results:
                        collected_results["precision_weighted"] = []
                    collected_results["precision_weighted"].append(
                        results[0.5]["precision_weighted"]
                    )

                    if "recall_micro" not in collected_results:
                        collected_results["recall_micro"] = []
                    collected_results["recall_micro"].append(
                        results[0.5]["recall_micro"]
                    )

                    if "recall_macro" not in collected_results:
                        collected_results["recall_macro"] = []
                    collected_results["recall_macro"].append(
                        results[0.5]["recall_macro"]
                    )

                    if "recall_weighted" not in collected_results:
                        collected_results["recall_weighted"] = []
                    collected_results["recall_weighted"].append(
                        results[0.5]["recall_weighted"]
                    )

                    if "f1_micro" not in collected_results:
                        collected_results["f1_micro"] = []
                    collected_results["f1_micro"].append(results[0.5]["f1_micro"])

                    if "f1_macro" not in collected_results:
                        collected_results["f1_macro"] = []
                    collected_results["f1_macro"].append(results[0.5]["f1_macro"])

                    if "f1_weighted" not in collected_results:
                        collected_results["f1_weighted"] = []
                    collected_results["f1_weighted"].append(results[0.5]["f1_weighted"])

                    if "auc" not in collected_results:
                        collected_results["auc"] = []
                    collected_results["auc"].append(results["auc"])

                    # accuracy should be a single value for all classes iirc
                    if "accuracy" not in collected_results:
                        collected_results["accuracy"] = []
                    collected_results["accuracy"].append(results[0.5]["accuracy"])

                    # <-- POTENZIALMENTE UTILE
                    # thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    # per_threshold_f1 = [
                    #     results[threshold]["f1"] if threshold in results else None
                    #     for threshold in thresholds
                    # ]

                    # # in realtà questi due possono essere derivati a posteriori
                    # # da ciò che sta sopra
                    # try:
                    #     best_threshold = thresholds[
                    #         per_threshold_f1.index(max(per_threshold_f1))
                    #     ]
                    # except:
                    #     best_threshold = -1
                    # collected_results["f1_best"].append(
                    #     results[best_threshold]["f1"]
                    #     if best_threshold in results
                    #     else None
                    # )
                    # collected_results["accuracy_best"].append(
                    #     results[best_threshold]["accuracy"]
                    #     if best_threshold in results
                    #     else None
                    # )
                    # POTENZIALMENTE UTILE --->

                if verbose:
                    print(
                        f"{Fore.GREEN}Collected results for model \
{Style.BRIGHT}{model}{Style.NORMAL} and run {Style.BRIGHT}{run}{Style.NORMAL}.{Fore.RESET}"
                    )
                # print(
                #     f"Added preds for {model}/{run} of shape {df.shape}, new shape {preds.shape}"
                # )

        # print(preds)

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        # preds.to_csv(output_path)

        print(collected_results)
        collected_results = pd.DataFrame(collected_results)
        print(collected_results)
        collected_results.to_csv(output_path)
