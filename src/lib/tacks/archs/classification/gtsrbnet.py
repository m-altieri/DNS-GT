# -*- coding: utf-8 -*-
"""GTSRB model as defined in the GRAPHITE implementation.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...models import TacksClassifier


class GTSRBNet(TacksClassifier):
    def __init__(self, device=None, logger=None):
        super().__init__(
            name='GTSRBNet',
            in_shape=(3, 32, 32),
            n_classes=43,
            device=device,
            logger=logger,
        )

        # define the architecture
        self.add_module('conv1_1', nn.Conv2d(3, 3, 1))

        self.add_module('conv2_1', nn.Conv2d(3, 32, 5, padding=2))
        self.add_module('conv2_2', nn.Conv2d(32, 32, 5, padding=2))
        self.add_module('pool2_1', nn.MaxPool2d(2))
        self.add_module('drop2_1', nn.Dropout2d(0.5))

        self.add_module('conv3_1', nn.Conv2d(32, 64, 5, padding=2))
        self.add_module('conv3_2', nn.Conv2d(64, 64, 5, padding=2))
        self.add_module('pool3_1', nn.MaxPool2d(2))
        self.add_module('drop3_1', nn.Dropout2d(0.5))

        self.add_module('conv4_1', nn.Conv2d(64, 128, 5, padding=2))
        self.add_module('conv4_2', nn.Conv2d(128, 128, 5, padding=2))
        self.add_module('pool4_1', nn.MaxPool2d(2))
        self.add_module('drop4_1', nn.Dropout2d(0.5))

        self.add_module('fc5_1', nn.Linear(4 * 4 * 128, 1024))
        self.add_module('drop5_1', nn.Dropout(0.5))
        self.add_module('fc6_1', nn.Linear(1024, 1024))
        self.add_module('drop6_1', nn.Dropout(0.5))
        self.add_module('fc7_1', nn.Linear(1024, self.n_classes))

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv2_1(x))

        x2 = self.drop2_1(self.pool2_1(F.relu(self.conv2_2(x))))

        x = F.relu(self.conv3_1(x2))
        x3 = self.drop3_1(self.pool3_1(F.relu(self.conv3_2(x))))

        x = F.relu(self.conv4_1(x3))
        x4 = self.drop4_1(self.pool4_1(F.relu(self.conv4_2(x))))

        x = x4.view(-1, 4 * 4 * 128)

        x = self.drop5_1(F.relu(self.fc5_1(x)))
        x = self.drop6_1(F.relu(self.fc6_1(x)))
        x = self.fc7_1(x)

        if self._debugger:
            self._debugger.trace(
                f'M_{self.name}_EndForward',
                x.mean().item(),
                x2.mean().item(),
                x3.mean().item(),
                x4.mean().item(),
            )

        return x
