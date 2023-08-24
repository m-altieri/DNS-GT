# -*- coding: utf-8 -*-
"""Testing of JSMA attack.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>, <rhamon@protonmail.com>
"""
import pytest

import torch

# from tacks.archs.classification.images import
from tacks.attacks.jsma import JSMAAttack
from tacks.data.mnist import load_dataset
from tacks.utils import get_argparser, get_config, Workspace

from art.attacks.evasion import SaliencyMapMethod

TOL = 1e-8

workspace = Workspace('Testing', subworkspace_name='attacks.fgm', args={'debug': True})
model_logger = get_logger('Testing | Model')


@pytest.fixture(scope='class')
def get_data(request):

    # get MNIST loaders
    train_dataset = load_dataset('train')
    test_dataset = load_dataset('test')

    in_shape = train_dataset.in_shape
    n_classes = len(train_dataset.classes)

    request.cls.n_classes = n_classes
    request.cls.in_shape = in_shape
    request.cls.train_dataset = train_dataset
    request.cls.test_dataset = test_dataset


@pytest.fixture(scope='class')
def get_model(request):

    # load simple MNISt model
    tacks_model = TorchModel.load_model(
        get_config().get_path('paths', 'models') / 'MNIST_lenet.pt'
    )
    tacks_model.clip_values = (0, 1)
    tacks_model.with_softmax = False
    tacks_model.logger = model_logger
    tacks_model.eval()

    # ART classifier
    art_classifier = PyTorchClassifier(
        tacks_model,
        loss=torch.nn.CrossEntropyLoss(),
        clip_values=tacks_model.clip_values,
        input_shape=tacks_model.in_shape,
        nb_classes=tacks_model.n_classes,
    )

    # store data in class
    request.cls.tacks_model = tacks_model
    request.cls.art_classifier = art_classifier

# get argument parser
argparser = get_argparser(sys.modules[__name__].__doc__, flags=['debug', 'silent'])

args = argparser.parse_args()

workspace = Workspace(name='Testing', subworkspace_name='attacks.jsma', args=args)

#######################################################################################
# Data

batch_size = 8

# get MNIST loaders
train_dataset = load_dataset('train')
test_dataset = load_dataset('test')

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

in_shape = train_dataset.in_shape
n_classes = len(train_dataset.classes)

instances, labels = next(iter(test_loader))

#######################################################################################
# Model

# load simple MNIST model
model = torch.load(get_config().get_path('paths', 'models') / 'MNIST_lenet.pt')
model.clip_values = (0, 1)
model.logger = workspace.logger

#######################################################################################
# Attack

tg_labels = torch.zeros(batch_size).long()
tg_probas = torch.ones(batch_size) * 0.25

attack = JSMAAttack(model, workspace)
perturbation = attack.generate(instances, tg_labels=tg_labels, tg_probas=tg_probas)
