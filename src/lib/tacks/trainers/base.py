# -*- coding: utf-8 -*-
"""Base for model trainers.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import shutil
from abc import abstractmethod
from pathlib import Path

import yaml
from tacks.models import TacksModel

from ..optim.base import DEFAULT_LRS_PARAMS, DEFAULT_OPTIM_PARAMS


class BaseTrainer:
    """Trainer for TacksModel.

    Parameters
    ----------
    name : str
        Name of the trainer.
    workspace : tacks.utils.Workspace
        Workspace where to work.
    model : TacksModel or pathlib.Path or str
        Model, as a TorchModel instance on a path to a saved model.
    """

    def __init__(self, name, model, workspace):
        self.name = name
        self.model = model
        self.workspace = workspace
        self.logger = self.workspace.logger

        if isinstance(self.model, Path):
            self.model = TacksModel.load(self.model, logger=self.logger)

    @property
    def weights_path(self):
        return self.workspace.weights_path

    @property
    def outputs_path(self):
        return self.workspace.outputs_path

    @property
    def imgs_path(self):
        return self.workspace.outputs_path

    @property
    def trainingoutputs_path(self):
        return self.outputs_path / f"{self.name}_to.yaml"

    @property
    def optimparams_path(self):
        return self.outputs_path / f"{self.name}_op.yaml"

    @property
    def is_trained(self):
        return self.trainingoutputs_path.exists()

    def get_training_outputs(self):
        """Return the training outputs of the current run."""

        if self.is_trained:
            with open(self.trainingoutputs_path, "r") as infile:
                training_outputs = yaml.load(infile, Loader=yaml.FullLoader)
        else:
            training_outputs = None

        return training_outputs

    @abstractmethod
    def train(
        self,
        data_loaders,
        loss_function,
        optim_params=None,
        metrics=None,
        stopping_criteria=None,
        eval_interval=1,
        save_interval=1,
        reset=False,
    ):
        """Train the model.

        Each run is in a separate folder in the workspace folder, and is numbered
        starting from 000.

        If :param:`reset` is True, create a new run, other wise load the previous run.

        Parameters
        ----------
        data_loaders : dict
            Loaders for data sets.
        loss_function : torch.nn._Loss
            Loss function between predictions and labels.
        optim_params : dict or None
            Parameters of the optimization. If None or missing keys, populate the dict
            with default values (see :mod:`optim` for details).
        metrics : dict or None, optional
            Dict of metrics to use for evaluation. Keyed by metric name and valued by
            metric function.
        stopping_criteria: dict or None, optional
            List of stopping criteria, keyed by metric name and values by threshold.
            Different behaviours apply according to the type of metrics (default: None).
        eval_interval : int, optional
            Interval at which the model is evaluated on the valid set.
        save_interval : int, optional
            Interval at which the model is saved.
        reset : bool, optional
            Indicates if the training is reset or not.
        """

        # --
        # Initialization of empty parameters and checks

        # stopping criteria
        if stopping_criteria is None:
            stopping_criteria = {}
        self._stopping_criteria = stopping_criteria

        # list of metrics
        if metrics is None:
            metrics = {}
        self._metrics = {}

        # optimizer params
        if optim_params is None:
            optim_params = {}
        self._optim_params = optim_params

        if "optimizer_params" not in self._optim_params:
            self._optim_params["optimizer_params"] = {}

        self._optim_params.update(
            {
                key: value
                for key, value in DEFAULT_OPTIM_PARAMS.items()
                if key not in self._optim_params
            }
        )

        self._optim_params["optimizer_params"].update(
            {
                key: value
                for key, value in DEFAULT_OPTIM_PARAMS["optimizer_params"].items()
                if key not in self._optim_params["optimizer_params"]
            }
        )

        # update lrs params with default params
        self._optim_params["lrscheduler_params"].update(
            {
                key: value
                for key, value in DEFAULT_LRS_PARAMS[
                    self._optim_params["lrscheduler_name"]
                ].items()
                if key not in self._optim_params["lrscheduler_params"]
            }
        )

        # update params for lr schedeler in specific case
        if self._optim_params["lrscheduler_name"] == "cyclic":
            if "n_train_batches" not in self._optim_params:
                err_msg = (
                    "`n_train_batches` should be provided when lr scheduler is cyclic."
                )
                raise ValueError(err_msg)

            self._optim_params.update(
                {
                    "lrscheduler_params": {
                        "max_lr": self._optim_params["optimizer_params"]["lr"],
                        "epochs": self._optim_params["n_epochs"],
                        "steps_per_epoch": self._optim_params["n_train_batches"],
                    }
                }
            )

        # check that a train loader is in loaders
        if "train" not in data_loaders:
            err_msg = "Train loader is not provided."
            raise ValueError(err_msg)

        if not reset and self.is_trained:
            self.logger.info("Found existing training parameters.")

            # load existing parameters
            with open(self.trainingoutputs_path, "r") as infile:
                training_outputs = yaml.load(infile, Loader=yaml.FullLoader)

            self._start_epoch = training_outputs["epoch"] + 1
            self._metrics_values = training_outputs["metrics_values"]

            # load existing parameters from previous epoch
            weight_path = self.weights_path / "{:04d}".format(training_outputs["epoch"])
            self.logger.info("Loading weights at %s...", str(weight_path))

            if weight_path.exists():
                self.logger.info("Loading state at epoch %04d", self._start_epoch - 1)
                self.model.load_state(weight_path)
            else:
                err_msg = "Error when loading weights at {:s}. Please reset training."
                raise ValueError(err_msg.format(str(weight_path)))

            self.logger.info("Resuming at epoch %d", self._start_epoch)

        else:
            self.logger.info("Initialization of training parameters")

            # init training parameters
            self._start_epoch = 0

            self._metrics_values = {
                f"{split_name}_{meter_name}{metric_name}": []
                for meter_name, (_, metric_names) in metrics.items()
                for metric_name in metric_names
                for split_name in self._split_names
            }

            self._metrics_values.update(
                {f"{split_name}_loss": [] for split_name in self._split_names}
            )
            self._metrics_values.update({"learning_rate": [], "epoch_duration": []})

            # empty outputs and weights path
            shutil.rmtree(self.outputs_path)
            shutil.rmtree(self.weights_path)
            self.outputs_path.mkdir()
            self.weights_path.mkdir()
