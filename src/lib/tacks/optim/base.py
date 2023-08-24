# -*- coding: utf-8 -*-
"""Util functions for optimization.



lrscheduler_name : str, optional
    Name of the scheduler of the learning rate (default: None).
lrscheduler_params : disc, optional
    Parameters of the scheduler of the learning rate (default: None).


Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

from torch import optim

LIST_OPTIMIZERS = {
    'adam': optim.Adam,
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'rmsprop': optim.RMSprop,
    'sgd': optim.SGD,
}
LIST_SCHEDULERS = {
    'reduceonplateau': (optim.lr_scheduler.ReduceLROnPlateau, 'epoch'),
    'cyclic': (optim.lr_scheduler.OneCycleLR, 'batch'),
    'none': (lambda optimizer: None, None),
    None: (lambda optimizer: None, None),
}


DEFAULT_OPTIM_PARAMS = {
    'optimizer_name': 'adam',
    'optimizer_params': {'lr': 1e-4, 'weight_decay': 0.0},
    'lrscheduler_name': 'none',
    'lrscheduler_params': {},
    'max_iter': 100,
    'with_early_stopping': True,
    'tolerance': 1e-7,
    'n_epochs': 10,
}

DEFAULT_LRS_PARAMS = {
    'none': {},
    'reduceonplateau': {'factor': 0.9, 'patience': 2, 'min_lr': 1e-5},
    'cyclic': {
        'max_lr': None,
        'epochs': None,
        'steps_per_epoch': None,
        'anneal_strategy': 'linear',
    },
}
