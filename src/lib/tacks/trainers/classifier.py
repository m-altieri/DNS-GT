# -*- coding: utf-8 -*-
"""Trainer for classifier.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from ..metrics.base import Meter
from ..metrics.classification import ClassificationMeter
from ..optim.base import LIST_OPTIMIZERS, LIST_SCHEDULERS
from .base import BaseTrainer


class ClassifierTrainer(BaseTrainer):
    """Trainer for TorchClassifier.

    Parameters
    ----------
    others :
        See :class:`TorchBaseTrainer`.
    """

    def __init__(self, name, model, workspace):
        super().__init__(name=name, model=model, workspace=workspace)

        self._split_names = ['train', 'valid']

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

        See :class:`TorchBaseTrainer`.
        """

        super().train(
            data_loaders=data_loaders,
            loss_function=None,
            optim_params=optim_params,
            metrics=metrics,
            stopping_criteria=stopping_criteria,
            eval_interval=eval_interval,
            save_interval=save_interval,
            reset=reset,
        )

        # add classification metrics if no metrics
        if len(self._metrics) == 0:
            self._metrics.update(
                {
                    '': (
                        partial(ClassificationMeter, self.model.n_classes),
                        ('accuracy', 'precision', 'recall'),
                    )
                }
            )

        metrics_values = self._metrics_values

        ################################################################################
        # Training

        # define optimizer and lr scheduler
        optimizer = LIST_OPTIMIZERS[self._optim_params['optimizer_name']](
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self._optim_params['optimizer_params'],
        )

        self.logger.info(
            'Number of trainable parameters: %d/%d',
            self.model.n_trainable_parameters,
            self.model.n_parameters,
        )

        lr_scheduler, lrs_update = LIST_SCHEDULERS[
            self._optim_params['lrscheduler_name']
        ]

        lr_scheduler = lr_scheduler(
            optimizer, **self._optim_params['lrscheduler_params']
        )

        # init running meters over batches
        running_meters = {'loss': Meter()}
        running_meters.update(
            {meter_name: meter() for meter_name, (meter, _) in metrics.items()}
        )

        n_epochs = self._optim_params['n_epochs']

        # test early stopping condition
        early_stopping = False

        if self._start_epoch > 0:
            for key, threshold in self._stopping_criteria.items():
                if key in ['train_accuracy', 'valid_accuracy']:
                    if metrics_values[key][-1] > threshold:
                        early_stopping = True
                        self.logger.info(f'Early stopping: {key} > {threshold:.2f}')

                elif key == 'total_duration':
                    if sum(metrics_values['epoch_duration']) > threshold:
                        early_stopping = True
                        self.logger.info(f'Early stopping: {key} > {threshold}s')

            if early_stopping:
                self.logger.info('Early stopping activated.')
                return

            if self._start_epoch >= n_epochs:
                self.logger.info('Number of epochs reached.')
                return

        self.logger.info('Start training.')

        # start training procedure
        desc_fields_length = [2 * len(str(n_epochs)) + 1, 7, 11, 10, 11, 10]

        header = '   '.join(
            [
                '#'.ljust(desc_fields_length[0]),
                'LR'.ljust(desc_fields_length[1]),
                'Train loss'.ljust(desc_fields_length[2]),
                'Train acc'.ljust(desc_fields_length[3]),
                'Valid loss'.ljust(desc_fields_length[4]),
                'Valid acc'.ljust(desc_fields_length[5]),
            ]
        )

        if self.workspace.verbose:
            print(header)

        for epoch in range(self._start_epoch, n_epochs):
            start_time = time.time()

            if early_stopping:
                self.logger.info('Early stopping activated.')
                break

            # store learning rate
            metrics_values['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # print messages
            self.logger.debug('*** Epoch %d/%d', epoch + 1, n_epochs)

            # set model in train mode
            self.model.train()

            # reset running meters over batches
            for meter in running_meters.values():
                meter.reset()

            # prepare progress bar over batches
            pb_batches = tqdm(data_loaders['train'], disable=not self.workspace.verbose)

            # start iteration over batches
            for instances, gt_labels in pb_batches:
                batch_size = instances.size(0)

                # send batch to device
                instances = instances.to(self.model.device)
                gt_labels = (
                    gt_labels.to(self.model.device) if gt_labels is not None else None
                )

                # set as half if required
                if self.model.half_precision:
                    instances = instances.half()

                # reset gradients
                optimizer.zero_grad()

                # compute prediction and gradients
                with torch.set_grad_enabled(True):
                    logits = self.model(instances)

                    if torch.any(logits.abs().sum(1) == 0):
                        raise ValueError('Null logits.')
                    loss = loss_function(logits, gt_labels)

                loss.backward()
                optimizer.step()

                # update loss and metrics
                for meter_name, meter in running_meters.items():
                    if meter_name == 'loss':
                        meter([loss.item() / batch_size] * batch_size)
                    elif meter_name in metrics.keys():
                        meter(logits, gt_labels)

                # update learning rate if update after each batch
                if lr_scheduler is not None and lrs_update == 'batch':
                    lr_scheduler.step()
                    metrics_values['learning_rate'].append(
                        optimizer.param_groups[0]['lr']
                    )

                pb_batches.desc = '   '.join(
                    [
                        f'{epoch+1:d}/{n_epochs}'.ljust(desc_fields_length[0]),
                        '{:.1e}'.format(metrics_values['learning_rate'][-1]).ljust(
                            desc_fields_length[1]
                        ),
                        '{:.1e}'.format(running_meters['loss'].avg).ljust(
                            desc_fields_length[2]
                        ),
                        '{:.3f}'.format(running_meters[''].get_accuracy()).ljust(
                            desc_fields_length[3]
                        ),
                        ''.ljust(desc_fields_length[4]),
                        ''.ljust(desc_fields_length[5]),
                    ]
                )

                # TODO: make adversarial collate update function work
                # ratio = np.random.uniform(0.,0.9)
                # cf = partial(at.adversarial_collate,
                #       model=self.model,
                #       ratio=ratio)
                # print('TEST COLLATE')
                # print('ratio: ', ratio)
                # pb_batches.iterable.collate_fn = cf
                # input(pb_batches.iterable.collate_fn)

            # store metrics for the epoch
            metrics_values['train_loss'].append(running_meters['loss']['avg'])

            for meter_name, (_, metric_names) in metrics.items():
                for metric_name in metric_names:
                    metrics_values[f'train_{meter_name}{metric_name}'].append(
                        running_meters[meter_name][metric_name]
                    )

            # print information about the epoch
            log_loss = '{:.2e}'.format(metrics_values['train_loss'][-1])
            log_metrics = ' - '.join(
                [
                    '{}{}: {:.1e}'.format(
                        meter_name,
                        metric_name,
                        metrics_values[f'train_{meter_name}{metric_name}'][-1],
                    )
                    for meter_name, (_, metric_names) in metrics.items()
                    for metric_name in metric_names
                ]
            )
            self.logger.debug('# Train - loss: %s, %s', log_loss, log_metrics)

            # evaluation phase
            if (epoch % eval_interval) == 0:
                loss_value, eval_metrics_values = self.evaluate(
                    data_loaders['valid'],
                    loss_function,
                    metrics,
                    desc_fields_length=desc_fields_length,
                )

                # store metrics for the epoch
                metrics_values['valid_loss'].append(loss_value)
                for meter_name, (_, metric_names) in metrics.items():
                    for metric_name in metric_names:
                        metrics_values[f'valid_{meter_name}{metric_name}'].append(
                            eval_metrics_values[f'{meter_name}{metric_name}']
                        )

                # print information about the epoch
                log_loss = '{:.2e}'.format(metrics_values['valid_loss'][-1])
                log_metrics = ' - '.join(
                    [
                        '{}{}: {:.1e}'.format(
                            meter_name,
                            metric_name,
                            metrics_values[f'valid_{meter_name}{metric_name}'][-1],
                        )
                        for meter_name, (_, metric_names) in metrics.items()
                        for metric_name in metric_names
                    ]
                )

                self.logger.debug('# Valid - loss: %s, %s', log_loss, log_metrics)

            # save time of the epoch
            metrics_values['epoch_duration'].append(time.time() - start_time)

            if (epoch % save_interval) == 0:
                # save weights of the model
                self.model.save_state(self.weights_path / f'{epoch:04d}')

                # save outputs of the training
                with open(self.trainingoutputs_path, 'w') as outfile:
                    yaml.dump(
                        {'epoch': epoch, 'metrics_values': metrics_values},
                        outfile,
                    )

            # update learning rate
            if lr_scheduler is not None and lrs_update == 'epoch':
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(metrics_values['valid_loss'][-1])
                else:
                    lr_scheduler.step()

            # check stopping criteria
            for key, threshold in self._stopping_criteria.items():
                if key in ['train_accuracy', 'valid_accuracy']:
                    if metrics_values[key][-1] > threshold:
                        early_stopping = True
                        self.logger.info(f'Early stopping: {key} > {threshold:.2f}')

                elif key == 'total_duration':
                    if sum(metrics_values['epoch_duration']) > threshold:
                        early_stopping = True
                        self.logger.info(f'Early stopping: {key} > {threshold}s')

        ################################################################################
        # Post-training operations

        # save last epoch if not saved
        if not (self.weights_path / f'{epoch:04d}').exists():
            self.logger.debug('Saving final state at epoch %04d', epoch)
            self.model.save_state(self.weights_path / f'{epoch:04d}')

        # save full model
        self.model.save(self.outputs_path / f'{self.name}_final.pt')

        # get final metrics
        time_elapsed = np.sum(metrics_values['epoch_duration']).item()
        info_template = 'Training complete in {:.0f}m {:.0f}s'
        self.logger.info(info_template.format(time_elapsed // 60, time_elapsed % 60))

        # save training outputs
        self.logger.info('Save training outputs.')

        with open(self.trainingoutputs_path, 'w') as outfile:
            yaml.dump({'epoch': epoch, 'metrics_values': metrics_values}, outfile)

        # save optim params
        self.logger.info('Save optim params.')

        with open(self.optimparams_path, 'w') as outfile:
            yaml.dump(self._optim_params, outfile)

    def test(self, test_loader, loss_function, metrics=None):
        """Test the model.

        Parameters
        ----------
        test_loader: torch.utils.data.DataLoader
            Loader for the test set.
        loss_function: torch.Module
            Loss function between predictions and labels.
        """

        # init metrics if None
        if metrics is None:
            metrics = {}

            metrics.update(
                {
                    '': (
                        partial(ClassificationMeter, self.model.n_classes),
                        ('accuracy', 'precision', 'recall'),
                    )
                }
            )

        loss_value, eval_metrics_values = self.evaluate(
            test_loader, loss_function, metrics
        )

        metrics_values = {}
        metrics_values['test_loss'] = loss_value

        for meter_name, (_, metric_names) in metrics.items():
            for metric_name in metric_names:
                metrics_values[f'test_{meter_name}{metric_name}'] = eval_metrics_values[
                    f'{meter_name}{metric_name}'
                ]

        # writes loss and metrics as printable strings
        log_loss = '{:.3e}'.format(loss_value)
        log_metrics = ' - '.join(
            [
                '{}{}: {:.1e}'.format(
                    meter_name,
                    metric_name,
                    eval_metrics_values[f'{meter_name}{metric_name}'],
                )
                for meter_name, (_, metric_names) in metrics.items()
                for metric_name in metric_names
            ]
        )

        # print the strings
        self.logger.info('# Test - loss: %s, %s', log_loss, log_metrics)

        return metrics_values

    def evaluate(self, loader, loss_function, metrics, desc_fields_length=None):
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
        running_meters = {'loss': Meter()}
        running_meters.update(
            {meter_name: meter() for meter_name, (meter, _) in metrics.items()}
        )

        # Iterate over data.
        pb_batches = tqdm(loader, disable=not self.workspace.verbose)

        for instances, gt_labels in pb_batches:
            # send to device
            instances = instances.to(self.model.device)

            if gt_labels is not None:
                gt_labels = gt_labels.to(self.model.device)

            # set as half if required
            if self.model.half_precision:
                instances = instances.half()

            batch_size = instances.size(0)

            with torch.set_grad_enabled(False):
                # forward pass the instance through the model
                logits = self.model(instances)

                # sanity check on the logits
                if torch.all(logits == 0):
                    raise ValueError('Invalid outputs: all logits are null.')

                _, preds = torch.max(logits, 1)

                # compute the loss
                loss = loss_function(logits, gt_labels)

            # update meters
            for meter_name, meter in running_meters.items():
                if meter_name == 'loss':
                    meter([loss.item() / batch_size] * batch_size)
                elif meter_name in metrics.keys():
                    meter(logits, gt_labels)

            if desc_fields_length is not None:
                pb_batches.desc = '   '.join(
                    [
                        ''.ljust(desc_fields_length[0]),
                        ''.ljust(desc_fields_length[1]),
                        ''.ljust(desc_fields_length[2]),
                        ''.ljust(desc_fields_length[3]),
                        '{:.1e}'.format(running_meters['loss'].avg).ljust(
                            desc_fields_length[4]
                        ),
                        '{:.3f}'.format(running_meters[''].get_accuracy()).ljust(
                            desc_fields_length[5]
                        ),
                    ]
                )

        return (
            running_meters['loss']['avg'],
            {
                f'{meter_name}{metric_name}': running_meters[meter_name][metric_name]
                for meter_name, (_, metric_names) in metrics.items()
                for metric_name in metric_names
            },
        )

    def plot_training_outputs(self):
        """Plot metrics values from training outputs."""

        self.workspace.logger.info('Plotting training outputs')

        # get metrics values
        training_outputs = self.get_training_outputs()
        metrics_values = training_outputs['metrics_values']

        plt.figure()

        plt.plot(metrics_values['train_loss'], label='Train loss')
        plt.plot(metrics_values['valid_loss'], label='Valid loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()

        plt.savefig(self.imgs_path / 'training_loss.png')
        plt.close()

        plt.figure()

        plt.plot(metrics_values['train_accuracy'], label='Train')
        plt.plot(metrics_values['valid_accuracy'], label='Valid')
        plt.plot(metrics_values['train_accuracy_3'], label='Train Top3')
        plt.plot(metrics_values['valid_accuracy_3'], label='Valid Top3')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()

        plt.savefig(self.imgs_path / 'training_accuracy.png')

        plt.figure()
        plt.semilogy(metrics_values['learning_rate'])

        plt.xlabel('Steps')
        plt.ylabel('Learning rate')
        plt.grid()

        plt.savefig(self.imgs_path / 'training_lr.png')

        self.workspace.logger.info('Plots saved in %s', self.imgs_path)
