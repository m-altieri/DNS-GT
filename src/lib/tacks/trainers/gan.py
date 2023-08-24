# -*- coding: utf-8 -*-
"""Trainers for GAN.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import time
import yaml
from functools import partial

import numpy as np
import torch
from tqdm.auto import tqdm

from .base import BaseTrainer
from ..metrics.base import Meter
from ..metrics.classification import ClassificationMeter
from ..optim.base import LIST_OPTIMIZERS, LIST_SCHEDULERS, DEFAULT_OPTIM_PARAMS


class TorchGANTrainer(BaseTrainer):
    """Class to train torch GAN models.

    Parameters
    ----------
    gen_model : torch.nn.Module
        Generator model.
    disc_model : torch.nn.Module
        Discriminator model.
    others :
        See :class:`TorchBaseTrainer`.
    """

    def __init__(self, workspace, gen_model, disc_model):
        super().__init__(workspace=workspace)

        self.gen_model = gen_model
        self.disc_model = disc_model

        self._split_names = ["gen", "disc"]

    @property
    def device(self):
        return self.gen_model.device

    def train(
        self,
        data_loader,
        gen_loss_function,
        disc_loss_function,
        optim_params=None,
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
        data_loader : torch.data.DataLoader
            Loader for the real data set.
        gen_loss_function : torch.Module
            Loss function for the generator model.
        disc_loss_function : torch.Module
            Loss function for the discriminator.
        optim_params%, stopping_criteria, eval_interval, save_interval, reset
            See :class:`TorchBaseTrainer`.
        """

        super().train(
            data_loaders=data_loader,
            loss_function=None,
            optim_params=optim_params,
            metrics=None,
            stopping_criteria=stopping_criteria,
            eval_interval=eval_interval,
            save_interval=save_interval,
            reset=reset,
        )

        ################################################################################
        # Loading workspace

        outputs_path = self.workspace.outputs_path
        weights_path = self.workspace.weights_path

        if not reset:
            # load the last weights of the generator if existing
            weight_path = weights_path / "gen_{:04d}".format(self._current_epoch)
            if weight_path.exists():
                self.gen_model.load_state(weight_path)

            # load the last weights of the discriminator if existing
            weight_path = weights_path / "gen_{:04d}".format(self._current_epoch)
            if weight_path.exists():
                self.disc_model.load_state(weight_path)

        ################################################################################
        # Training

        metrics_values = self._metrics_values
        current_epoch = self._current_epoch

        # init optimizers
        disc_optimizer = LIST_OPTIMIZERS[optim_params["optimizer_name"]](
            self.disc_model.parameters(), **optim_params["optimizer_params"]
        )

        gen_optimizer = LIST_OPTIMIZERS[optim_params["optimizer_name"]](
            self.gen_model.parameters(), **optim_params["optimizer_params"]
        )

        if optim_params["lrscheduler_name"] is not None:
            lr_scheduler = LIST_SCHEDULERS[optim_params["lrscheduler_name"]](
                disc_optimizer, **optim_params["lrscheduler_params"]
            )
        else:
            lr_scheduler = None

        # init running meters over batches
        running_meters = {"disc_loss": Meter(), "gen_loss": Meter()}

        n_epochs = self._optim_params["n_epochs"]

        if current_epoch >= n_epochs:
            self.logger.info("Training already done.")

        # test early stopping condition
        early_stopping = False
        if current_epoch > 0:
            for key, threshold in stopping_criteria.items():
                if key in ["train_acc", "valid_acc"]:
                    if metrics_values[key][-1] > threshold:
                        early_stopping = True
                        self.logger.info(f"Early stopping: {key} > {threshold:.2f}")

                elif key == "total_duration":
                    if sum(metrics_values["epoch_duration"]) > threshold:
                        early_stopping = True
                        self.logger.info(f"Early stopping: {key} > {threshold}s")

        self.logger.info("# Start training")

        # start training procedure
        desc_template = "LR:{:.1e} / L:{:.1e} / Acc:{:.2f}"

        # start training procedure
        for epoch in range(current_epoch, n_epochs):
            start_time = time.time()

            # prepare progress bar over batches
            pb_batches = tqdm(data_loader["train"], disable=not self.workspace.verbose)

            if early_stopping:
                break

            # store learning rate
            metrics_values["learning_rate"].append(disc_optimizer.param_groups[0]["lr"])

            # print messages
            self.logger.info("*** Epoch %d/%d", epoch + 1, n_epochs)

            # set model in train mode
            self.gen_model.train()
            self.disc_model.train()

            # reset running meters over batches
            for meter in running_meters.values():
                meter.reset()

            # start iteration over batches
            for train_instances, train_labels in pb_batches:
                # get batch size
                batch_size = train_instances.size(0)

                # send batch to device
                train_instances = train_instances.to(self.device)

                ########################################################################
                # Discriminator update

                # reset gradients
                disc_optimizer.zero_grad()

                # compute loss
                gen_inputs = self.gen_model.generate_input(batch_size).detach()
                gen_labels = self.gen_model.generate_labels(batch_size)
                if gen_labels is not None:
                    gen_labels = gen_labels.detach()

                gen_instances = self.gen_model(gen_inputs, gen_labels).detach()

                disc_train_scores = self.disc_model(train_instances, train_labels)
                disc_gen_scores = self.disc_model(gen_instances, gen_labels)
                disc_scores = torch.cat([disc_train_scores, disc_gen_scores], 0)
                disc_labels = torch.cat(
                    [torch.ones((batch_size, 1)), torch.zeros((batch_size, 1))], 0
                ).to(self.device)

                if disc_loss_function.name == "WassersteinLoss":
                    disc_loss = disc_loss_function(
                        disc_scores,
                        disc_labels,
                        train_instances,
                        gen_instances,
                        train_labels,
                        gen_labels,
                    )
                else:
                    disc_loss = disc_loss_function(disc_scores, disc_labels)

                disc_loss.backward()
                disc_optimizer.step()

                ########################################################################
                # Generator update

                gen_optimizer.zero_grad()

                gen_inputs = self.gen_model.generate_input(batch_size)
                gen_labels = self.gen_model.generate_labels(batch_size)
                gen_instances = self.gen_model(gen_inputs, gen_labels)

                gen_scores = self.disc_model(gen_instances, gen_labels)
                gen_labels = torch.ones((batch_size, 1)).to(self.device)
                gen_loss = gen_loss_function(gen_scores, gen_labels)

                gen_loss.backward()
                gen_optimizer.step()

                # update loss and metrics
                for meter_name, meter in running_meters.items():
                    if meter_name == "disc_loss":
                        meter([disc_loss.item() / batch_size] * batch_size)
                    elif meter_name == "gen_loss":
                        meter([gen_loss.item() / batch_size] * batch_size)
                    elif meter_name in metrics.keys():
                        meter(logits, gt_labels)
                    else:
                        raise ValueError(meter)

            # store metrics for the epoch
            metrics_values["gen_loss"].append(running_meters["gen_loss"]["avg"])
            metrics_values["disc_loss"].append(running_meters["disc_loss"]["avg"])

            # for meter_name, (_, metric_names) in metrics.items():
            #     for metric_name in metric_names:
            #         metrics_values[f'train_{meter_name}{metric_name}'].append(
            #             running_meters[meter_name][metric_name]
            #         )

            # print information about the epoch
            log_gen_loss = "{:.2e}".format(metrics_values["gen_loss"][-1])
            log_disc_loss = "{:.2e}".format(metrics_values["disc_loss"][-1])
            log_metrics = " - ".join(
                [
                    # '{}{}: {:.1e}'.format(
                    #     meter_name,
                    #     metric_name,
                    #     metrics_values[f'train_{meter_name}{metric_name}'][-1],
                    # )
                    # for meter_name, (_, metric_names) in metrics.items()
                    # for metric_name in metric_names
                ]
            )
            self.logger.info("# Generator loss: %s, %s", log_gen_loss, log_metrics)
            self.logger.info("# Discriminator loss: %s, %s", log_disc_loss, log_metrics)

            # save time of the epoch
            metrics_values["epoch_duration"].append(time.time() - start_time)

            if (epoch % save_interval) == 0:
                self.disc_model.save_state(weights_path / f"disc_{epoch:04d}")
                self.gen_model.save_state(weights_path / f"gen_{epoch:04d}")

            # update learning rate
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(metrics_values["valid_loss"][-1])
                else:
                    lr_scheduler.step()

            # update current epoch
            current_epoch = epoch + 1

            # save last epoch
            if current_epoch == n_epochs:
                self.disc_model.save_state(weights_path / f"disc_{epoch:04d}")
                self.gen_model.save_state(weights_path / f"gen_{epoch:04d}")

            # # check stopping criteria
            # for key, threshold in stopping_criteria.items():
            #     if key in ['train_acc', 'valid_acc']:
            #         if metrics_values[key][-1] > threshold:
            #             early_stopping = True
            #             self.logger.info(f'Early stopping: {key} > {threshold:.2f}')

            #     elif key == 'total_duration':
            #         if sum(metrics_values['epoch_duration']) > threshold:
            #             early_stopping = True
            #             self.logger.info(f'Early stopping: {key} > {threshold}s')

        ################################################################################
        # Post-training operations

        # get final metrics
        time_elapsed = np.sum(metrics_values["epoch_duration"]).item()
        info_template = "Training complete in {:.0f}m {:.0f}s"
        self.logger.info(info_template.format(time_elapsed // 60, time_elapsed % 60))

        # save training outputs
        self.logger.info("Save training outputs.")

        training_outputs = {
            "current_epoch": current_epoch,
            "metrics_values": metrics_values,
        }

        with open(outputs_path / "training_outputs.yaml", "w") as outfile:
            yaml.dump(training_outputs, outfile)

        # save optim params
        self.logger.info("Save optim params.")

        with open(outputs_path / "optim_params.yaml", "w") as outfile:
            yaml.dump(optim_params, outfile)

        self.is_trained = True


class DetectorTrainer(BaseTrainer):
    """Class to train torch detection models.

    Parameters
    ----------
    model : torch.nn.Module
        Classifier model.
    others :
        See :class:`TorchBaseTrainer`.
    """

    def __init__(self, model, workspace):
        super().__init__(workspace=workspace)

        self.model = model

        # move model to device
        self.model.to(self.model.device)

        # convert model to half precision
        if self.half_precision:
            self.model.half()

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
        raise NotImplementedError
        """Train the model.

        Each run is in a separate folder in the workspace folder, and is numbered
        starting from 000.

        If :param:`reset` is True, create a new run, other wise load the previous run.

        Parameters
        ----------
        data_loaders : dict
            Loaders for data sets.
        loss_function : torch.Module
            Loss function between predictions and labels.
        n_epochs : int
            Number of epochs.
        optim_params : dict or None
            Parameters of the optimization. If None, populate the dict with default
            values (see :mod:`tacks.optim` for details).
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
        self.logger.info("`%%% Training %%%")

        ################################################################################
        # Initialization of empty parameters and checks

        if stopping_criteria is None:
            stopping_criteria = {}

        if metrics is None:
            metrics = {}
            metrics.update(
                {
                    "": (
                        partial(ClassificationMeter, self.model.n_classes),
                        ("accuracy", "precision", "recall"),
                    )
                }
            )

        if optim_params is None:
            optim_params = {}

        if "optimizer_params" not in optim_params:
            optim_params["optimizer_params"] = {}

        optim_params.update(
            {
                key: value
                for key, value in DEFAULT_OPTIM_PARAMS.items()
                if key not in optim_params
            }
        )

        optim_params["optimizer_params"].update(
            {
                key: value
                for key, value in DEFAULT_OPTIM_PARAMS["optimizer_params"].items()
                if key not in optim_params["optimizer_params"]
            }
        )

        if optim_params["lrs_params"] is not None:
            optim_params["lrs_params"].update(
                {
                    key: value
                    for key, value in DEFAULT_OPTIM_PARAMS["lrs_params"].items()
                    if key not in optim_params["lrs_params"]
                }
            )

        # check that a train loader is in loaders
        if "train" not in data_loaders:
            err_msg = "Train loader is not provided."
            raise ValueError(err_msg)

        ################################################################################
        # Loading workspace

        outputs_path = self.workspace.outputs_path
        weights_path = self.workspace.weights_path

        # load the last weights if existing
        list_weights = sorted([item for item in weights_path.glob("*")])

        if len(list_weights) > 0:
            self.model.load_state(list_weights[-1])

        if not reset and (outputs_path / "training_outputs.yaml").exists():
            # load existing parameters
            with open(outputs_path / "training_outputs.yaml", "r") as infile:
                training_outputs = yaml.load(infile, Loader=yaml.FullLoader)

            current_epoch = training_outputs["current_epoch"]
            metrics_values = training_outputs["metrics_values"]

            self.logger.info("Load training outputs.")

        else:
            # init training parameters
            current_epoch = 0

            metrics_values = {
                f"{split_name}_{meter_name}{metric_name}": []
                for meter_name, (_, metric_names) in metrics.items()
                for metric_name in metric_names
                for split_name in ["train", "valid"]
            }

            metrics_values.update(
                {f"{split_name}_loss": [] for split_name in ["train", "valid"]}
            )
            metrics_values.update({"learning_rate": [], "epoch_duration": []})

        ################################################################################
        # Pre-training initialization

        # init running meters over batches
        running_meters = {"loss": Meter()}
        running_meters.update(
            {meter_name: meter() for meter_name, (meter, _) in metrics.items()}
        )

        # init optimization objects
        optimizer = LIST_OPTIMIZERS[optim_params["optimizer_name"]](
            self.model.parameters(), **optim_params["optimizer_params"]
        )

        if optim_params["lrs_name"] is not None:
            lr_scheduler = LIST_SCHEDULERS[optim_params["lrs_name"]](
                optimizer, **optim_params["lrs_params"]
            )
        else:
            lr_scheduler = None

        n_epochs = optim_params["n_epochs"]

        if current_epoch >= n_epochs:
            self.logger.info("Training already done.")

        # early stopping
        early_stopping = False

        ################################################################################
        # Training

        # test early stopping condition
        if current_epoch > 0:
            for key, threshold in stopping_criteria.items():
                if key in ["train_acc", "valid_acc"]:
                    if metrics_values[key][-1] > threshold:
                        early_stopping = True
                        self.logger.info(f"Early stopping: {key} > {threshold:.2f}")

                elif key == "total_duration":
                    if sum(metrics_values["epoch_duration"]) > threshold:
                        early_stopping = True
                        self.logger.info(f"Early stopping: {key} > {threshold}s")

        self.logger.info("# Start training")

        # start training procedure
        for epoch in range(current_epoch, n_epochs):
            start_time = time.time()

            if early_stopping:
                break

            # store learning rate
            metrics_values["learning_rate"].append(optimizer.param_groups[0]["lr"])

            # print messages
            self.logger.info("*** Epoch %d/%d", epoch + 1, n_epochs)
            self.logger.info(
                "Learning rate: {:.2}".format(metrics_values["learning_rate"][-1])
            )

            # set model in train mode
            self.model.train()

            # prepare progress bar over batches
            pb_batches = tqdm(data_loaders["train"], disable=not self.workspace.verbose)

            # reset running meters over batches
            for meter in running_meters.values():
                meter.reset()

            # start iteration over batches
            for instances, gt_labels in pb_batches:
                batch_size = instances.size(0)

                # send batch to device
                instances = instances.to(self.model.device)
                gt_labels = (
                    gt_labels.to(self.model.device) if gt_labels is not None else None
                )

                # reset gradients
                optimizer.zero_grad()

                # compute prediction and gradients
                with torch.set_grad_enabled(True):
                    logits = self.model(instances)
                    loss = loss_function(logits, gt_labels)

                loss.backward()
                optimizer.step()

                # update loss and metrics
                for meter_name, meter in running_meters.items():
                    if meter_name == "loss":
                        meter([loss.item() / batch_size] * batch_size)
                    elif meter_name in metrics.keys():
                        meter(logits, gt_labels)

            # store metrics for the epoch
            metrics_values["train_loss"].append(running_meters["loss"]["avg"])

            for meter_name, (_, metric_names) in metrics.items():
                for metric_name in metric_names:
                    metrics_values[f"train_{meter_name}{metric_name}"].append(
                        running_meters[meter_name][metric_name]
                    )

            # print information about the epoch
            log_loss = "{:.2e}".format(metrics_values["train_loss"][-1])
            log_metrics = " - ".join(
                [
                    "{}{}: {:.1e}".format(
                        meter_name,
                        metric_name,
                        metrics_values[f"train_{meter_name}{metric_name}"][-1],
                    )
                    for meter_name, (_, metric_names) in metrics.items()
                    for metric_name in metric_names
                ]
            )
            self.logger.info("# Train - loss: %s, %s", log_loss, log_metrics)

            # evaluation phase
            if (epoch % eval_interval) == 0:
                loss_value, eval_metrics_values = self.evaluate(
                    data_loaders["valid"], loss_function, metrics
                )

                # store metrics for the epoch
                metrics_values["valid_loss"].append(loss_value)
                for meter_name, (_, metric_names) in metrics.items():
                    for metric_name in metric_names:
                        metrics_values[f"valid_{meter_name}{metric_name}"].append(
                            eval_metrics_values[f"{meter_name}{metric_name}"]
                        )

                # print information about the epoch
                log_loss = "{:.2e}".format(metrics_values["valid_loss"][-1])
                log_metrics = " - ".join(
                    [
                        "{}{}: {:.1e}".format(
                            meter_name,
                            metric_name,
                            metrics_values[f"valid_{meter_name}{metric_name}"][-1],
                        )
                        for meter_name, (_, metric_names) in metrics.items()
                        for metric_name in metric_names
                    ]
                )

                self.logger.info("# Valid - loss: %s, %s", log_loss, log_metrics)

            # save time of the epoch
            metrics_values["epoch_duration"].append(time.time() - start_time)

            if (epoch % save_interval) == 0:
                self.model.save_state(weights_path / f"{epoch:04d}")

            # update learning rate
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(metrics_values["valid_loss"][-1])
                else:
                    lr_scheduler.step()

            # update current epoch
            current_epoch = epoch + 1

            # save last epoch
            if current_epoch == n_epochs:
                self.model.save_state(weights_path / f"{epoch-1:04d}")

            # check stopping criteria
            for key, threshold in stopping_criteria.items():
                if key in ["train_acc", "valid_acc"]:
                    if metrics_values[key][-1] > threshold:
                        early_stopping = True
                        self.logger.info(f"Early stopping: {key} > {threshold:.2f}")

                elif key == "total_duration":
                    if sum(metrics_values["epoch_duration"]) > threshold:
                        early_stopping = True
                        self.logger.info(f"Early stopping: {key} > {threshold}s")

        ################################################################################
        # Post-training operations

        # get final metrics
        time_elapsed = np.sum(metrics_values["epoch_duration"]).item()
        info_template = "Training complete in {:.0f}m {:.0f}s"
        self.logger.info(info_template.format(time_elapsed // 60, time_elapsed % 60))

        # save training outputs
        self.logger.info("Save training outputs.")

        training_outputs = {
            "current_epoch": current_epoch,
            "metrics_values": metrics_values,
        }

        with open(outputs_path / "training_outputs.yaml", "w") as outfile:
            yaml.dump(training_outputs, outfile)

        # save optim params
        self.logger.info("Save optim params.")

        with open(outputs_path / "optim_params.yaml", "w") as outfile:
            yaml.dump(optim_params, outfile)

        self.is_trained = True

    def test(self, test_loader, loss_function, metrics=None):
        """Test the model.

        Parameters
        ----------
        test_loader: torch.utils.data.DataLoader
            Loader for the test set.
        loss_function: torch.Module
            Loss function between predictions and labels.
        """

        self.logger.info("# Test")

        # init metrics if None
        if metrics is None:
            metrics = {}

            metrics.update(
                {
                    "": (
                        partial(ClassificationMeter, self.model.n_classes),
                        ("accuracy", "precision", "recall"),
                    )
                }
            )

        loss_value, eval_metrics_values = self.evaluate(
            test_loader, loss_function, metrics
        )

        metrics_values = {}
        metrics_values["test_loss"] = loss_value

        for meter_name, (_, metric_names) in metrics.items():
            for metric_name in metric_names:
                metrics_values[f"test_{meter_name}{metric_name}"] = eval_metrics_values[
                    f"{meter_name}{metric_name}"
                ]

        # writes loss and metrics as printable strings
        log_loss = "{:.3e}".format(loss_value)
        log_metrics = " - ".join(
            [
                "{}{}: {:.1e}".format(
                    meter_name,
                    metric_name,
                    eval_metrics_values[f"{meter_name}{metric_name}"],
                )
                for meter_name, (_, metric_names) in metrics.items()
                for metric_name in metric_names
            ]
        )

        # print the strings
        self.logger.info("# Test - loss: %s, %s", log_loss, log_metrics)

        return metrics_values

    def evaluate(self, loader, loss_function, metrics):
        """Evaluate the model on a dataset.

        Parameters
        ----------
        loader: torch.utils.data.DataLoader
            Loader for the dataset.
        loss_function: torch.Module
            Loss function between predictions and labels.
        metrics_func : dict
            Dict of metrics to use for evaluation. Keyed by metric name and valued by
            metric function.

        Returns
        -------
        tuple of floats
            Mean value for the loss and all metrics.
        """
        # set model in evaluation mode
        self.model.eval()

        # init running meters over batches
        running_meters = {"loss": Meter()}
        running_meters.update(
            {meter_name: meter() for meter_name, (meter, _) in metrics.items()}
        )

        # Iterate over data.
        pb_batches = tqdm(loader, disable=not self.workspace.verbose)
        for instances, gt_labels in pb_batches:
            instances = instances.to(self.model.device)
            if gt_labels is not None:
                gt_labels = gt_labels.to(self.model.device)

            batch_size = instances.size(0)

            with torch.set_grad_enabled(False):
                # forward pass the instance through the model
                logits = self.model(instances)
                _, preds = torch.max(logits, 1)

                # compute the loss
                loss = loss_function(logits, gt_labels)

            # update meters
            for meter_name, meter in running_meters.items():
                if meter_name == "loss":
                    meter([loss.item() / batch_size] * batch_size)
                elif meter_name in metrics.keys():
                    meter(logits, gt_labels)

        return (
            running_meters["loss"]["avg"],
            {
                f"{meter_name}{metric_name}": running_meters[meter_name][metric_name]
                for meter_name, (_, metric_names) in metrics.items()
                for metric_name in metric_names
            },
        )
