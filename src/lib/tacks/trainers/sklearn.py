# -*- coding: utf-8 -*-
"""Handling sklearn models.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import pickle
import yaml
from pathlib import Path

import numpy as np

from sklearn.model_selection import ParameterGrid, cross_val_score

from ..utils import check_path, get_config


class SklearnModelHandler:
    """Class to handle SkLearn models.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Model to handle.
    others :
        See :class:`BaseModelHandler`.
    """

    def __init__(self, model, workspace, verbose=True, logger=None, **kwargs):
        self.model = model
        self.workspace = workspace
        self.logger = self.workspace.logger
        self.verbose = verbose

    @staticmethod
    def load(mh_path, logger=None):
        """Load a Sklearn model handler.

        Parameters
        ----------
        mh_path : str of pathlib.Path
            Path to the saved model handler. It can be either an absolute path,
            or relative to the model path defined in the configuration file.

        Returns
        -------
        SklearnModelHandler
        """
        mh_path = Path(mh_path)

        if not mh_path.is_absolute():
            mh_path = SklearnModelHandler.config.get_path('paths', 'models') / mh_path

        # check if the path is valid
        check_path(mh_path)

        # load the params of the model handler
        with open(mh_path / 'mh_params.yaml', 'r') as infile:
            mh_params = yaml.load(infile, Loader=yaml.FullLoader)

        # load the model
        from joblib import load

        model = load(str(mh_path / 'model.skl'))

        mh = SklearnModelHandler(model=model, **mh_params, logger=logger)

        mh.logger.info('Model handler loaded from %s', mh_path)

        return mh

    def save(self):
        """Save the Sklearn model handler in the workspace directory.

        Parameters
        ----------
        model_path : pathlib.Path or str
            Path to the saved model.
        """
        from joblib import dump

        # save the model
        dump(self.model, str(self.mh_path / 'model.skl'))

        # save the params of the model handler
        mh_params = {'task': self.task, 'name': self._name, 'verbose': self.verbose}

        with open(self.mh_path / 'mh_params.yaml', 'w') as outfile:
            yaml.dump(mh_params, outfile)

        self.logger.info('Model handler saved to %s', self.mh_path)

    def train(self, instances, labels):
        """Train the model.

        This corresponds to the :py:meth:`fit` method in :py:mod:`sklearn`.

        Parameters
        ----------
        instances : array-like
            Training data.
        labels : array-like
            Training label.
        """
        self.model.fit(instances, labels)

        self.logger.info('Training complete')
        self.evaluate(instances, labels)

    def evaluate(self, instances, labels, metrics):
        """Evaluate the model on instances and labels.

        Parameters
        ----------
        instances : array-like
            Training data.
        labels : array-like
            Training label.
        """

        outputs, (pred_labels, pred_scores) = self.predict(instances)

        metrics_value = {
            metric_name: metric_func(labels, outputs, pred_labels, pred_scores)
            for metric_name, metric_func in metrics.items()
        }

        log_metrics = ' - '.join(
            [
                '{}: {:.3f}'.format(metric_name.capitalize(), metric_value)
                for metric_name, metric_value in metrics_value.items()
            ]
        )

        self.logger.info(log_metrics)

        return metrics_value

    def predict(self, instances):
        """Predict the results of a given batch of instances.

        Parameters
        ----------
        features : array-like
            Features.

        Returns
        -------
        prediction
        """
        outputs = self.model.predict_proba(instances)
        pred_labels = np.argmax(outputs, 1)
        pred_probas = outputs[np.arange(pred_labels.shape[0]), pred_labels]

        return outputs, (pred_labels, pred_probas)

    def find_best_hyperparams(self, instances, labels, params):
        """Find the best hyperparameters for a dataset using cross-validation.

        Parameters
        ----------
        mh : SklearnModelHandler
            Model handler.
        instances : array-like
            Training data.
        labels : array-like
            Training label.
        params : dict
            Ranges for parameters.

        """
        # grid of parameters
        params_grid = list(ParameterGrid(params))

        scores = []

        for params in params_grid:
            if self.logger:
                self.logger.info('Next cross-validation step: %s', params)

            model = self.model.set_params(**params)

            for param in [param for param in model.__dict__.keys() if param[-1] == '_']:
                del model.__dict__[param]

            score = cross_val_score(model, instances, labels, cv=10).mean()
            scores.append(score)

            if self.logger:
                self.logger.info('CV score: %.2f', score)

        best_params = params_grid[np.argmax(scores)]
        print('best_params', best_params)

        for param in [param for param in model.__dict__.keys() if param[-1] == '_']:
            del model.__dict__[param]

        return best_params

    def publish_model(self, prefix='', suffix=''):
        """Publish the model.

        Parameters
        ----------
        See :meth:`BaseModelHandler.publish`.
        """
        # path to model folders
        models_path = get_config().get_path('paths', 'models')

        # get model name
        self.model.name = f'{prefix}{self.model.name}{suffix}'
        filename = f'{self.model.name}.pickle'
        with open(models_path / filename, 'wb') as outfile:
            pickle.dump(self.model, outfile)

        if self.logger:
            self.logger.info(f'Model published as {filename}')

    def __call__(self, instances):
        return self.model.predict_log_proba(instances)
