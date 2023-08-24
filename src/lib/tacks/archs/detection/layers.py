# -*- coding: utf-8 -*-
"""Object detection layers.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import math
import torch
import torch.nn as nn

from ..layers import ConvBn2d


def space_to_depth(x, block_size=4, focus=False):
    """Rearranges blocks of spatial data into depth.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    block_size : int, optional
        Size of the block (default: 4).
    is_focus : bool, optional
        Reproduces focus behaviours, for :param:`block_size` equal to 2
        (default: False).

    Returns
    -------
    torch.Tensor
        Output tensor.

    References
    ----------
    [1] T. Ridnik, H. Lawen, A. Noy, I. Friedman, E. B. Baruch, and G. Sharir,
    ‘TResNet: High Performance GPU-Dedicated Architecture’, 2020, [Online].
    Available: http://arxiv.org/abs/2003.13630.
    """
    batch_size, n_channels, height, width = x.shape

    if height % block_size != 0 or width % block_size != 0:
        err_msg = (
            'The width ({}) and the height ({}) should be divisible by '
            'the size of the block ({}).'
        )
        raise ValueError(err_msg.format(width, height, block_size))

    if focus:
        if block_size != 2:
            err_msg = 'Block size should be equal to 2 to reproduce Focus'
            'behavious (currently: {})'
            raise ValueError(err_msg.format(block_size))

        return torch.cat(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ],
            1,
        )
    else:
        return x.view(
            -1, n_channels * 2 * block_size, height // block_size, width // block_size
        )


class Focus(ConvBn2d):
    """Focus information on width and height into reduced space.

    This consists in applying :py:func:`space_to_depth` on inputs, and through
    a convolutional layer.

    Parameters
    ----------
    See :py:class:`nn.Conv2d`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
    ):
        super(Focus, self).__init__(
            in_channels * 4,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            eps=1e-3,
            momentum=0.03,
            activation='hardswish',
        )

    def forward(self, x):
        return super().forward(space_to_depth(x, block_size=2, focus=True))


class Bottleneck(nn.Module):
    """Bottleneck layer.

    Parameters
    ----------
    in_channels, out_channels, kernel_size, stride, padding, dilation, groups
        See :py:class:`nn.Conv2d`.
    shortcut : bool, optional
        Indicates if a shortcut is done or not (default: True).
    expansion_factor : float, optional
        Expansion factor for the hidden layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        shortcut=True,
        expansion_factor=0.5,
    ):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion_factor)
        self.conv1 = ConvBn2d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            eps=1e-3,
            momentum=0.03,
            activation='hardswish',
        )
        self.conv2 = ConvBn2d(
            hidden_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            groups=groups,
            eps=1e-3,
            momentum=0.03,
            activation='hardswish',
        )
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class BottleneckCSP(nn.Module):
    """Cross Stage Partial Bottleneck layer.

    Parameters
    ----------
    in_channels, out_channels, groups
        See :py:class:`nn.Conv2d`.
    n_bottlenecks : int, optional
        Number of bottleneck layers (default: 1)
    shortcut : bool, optional
        Indicates if a shortcut is done or not (default: True).
    expansion_factor : float, optional
        Expansion factor.

    References
    ----------
    [1] C.-Y. Wang, H.-Y. M. Liao, Y.-H. Wu, P.-Y. Chen, J.-W. Hsieh, and I.-H.
    Yeh, ‘CSPNet: A New Backbone That Can Enhance Learning Capability of CNN’,
    2020, pp. 390–391, Accessed: Aug. 04, 2020.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        groups=1,
        n_bottlenecks=1,
        shortcut=True,
        expansion_factor=0.5,
    ):
        super(BottleneckCSP, self).__init__()

        # number of hidden channels
        hid_channels = int(out_channels * expansion_factor)

        self.conv1 = ConvBn2d(
            in_channels,
            hid_channels,
            kernel_size=1,
            stride=1,
            eps=1e-3,
            momentum=0.03,
            activation='hardswish',
        )
        self.conv2 = nn.Conv2d(
            in_channels, hid_channels, kernel_size=1, stride=1, bias=False
        )
        self.conv3 = nn.Conv2d(
            hid_channels, hid_channels, kernel_size=1, stride=1, bias=False
        )
        self.conv4 = ConvBn2d(
            2 * hid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            eps=1e-3,
            momentum=0.03,
            activation='hardswish',
        )

        self.bn = nn.BatchNorm2d(2 * hid_channels, eps=1e-3, momentum=0.03)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

        self.layers = nn.Sequential(
            *[
                Bottleneck(
                    hid_channels,
                    hid_channels,
                    groups=groups,
                    shortcut=shortcut,
                    expansion_factor=1.0,
                )
                for _ in range(n_bottlenecks)
            ]
        )

    def forward(self, x):
        y1 = self.conv3(self.layers(self.conv1(x)))
        y2 = self.conv2(x)
        x = self.activation(self.bn(torch.cat((y1, y2), dim=1)))
        return self.conv4(x)


class SPP(nn.Module):
    """Spatial pyramid pooling layer.

    Parameters
    ----------
    in_channels, out_channels
        See :py:class:`nn.Conv2d`.
    kernel_size : 3-tuple of ints
        Kernel size.
    """

    def __init__(self, in_channels, out_channels, kernel_size=(5, 9, 13)):
        super(SPP, self).__init__()

        hid_channels = in_channels // 2
        self.conv1 = ConvBn2d(
            in_channels,
            hid_channels,
            kernel_size=1,
            stride=1,
            eps=1e-3,
            momentum=0.03,
            activation='hardswish',
        )
        self.conv2 = ConvBn2d(
            hid_channels * (len(kernel_size) + 1),
            out_channels,
            kernel_size=1,
            stride=1,
            eps=1e-3,
            momentum=0.03,
            activation='hardswish',
        )
        self.layers = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernel_size]
        )

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.layers], 1))


class Detect(nn.Module):
    """Detection layer.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    anchors : list of ints
        List of anchors for the different sizes of the grid.
    list_in_channels : list of ints
        List of input channels of previous layers.
    """

    def __init__(self, n_outputs, n_grids, n_anchors, list_in_channels):
        super(Detect, self).__init__()

        self.n_outputs = n_outputs
        self.n_grids = n_grids
        self.n_anchors = n_anchors

        # define a conv layer for each size factor
        self.layers = nn.ModuleList(
            nn.Conv2d(in_channels, self.n_outputs * self.n_anchors, 1)
            for in_channels in list_in_channels
        )

    def forward(self, x):

        for ids in range(self.n_grids):
            # convolutional layer
            x[ids] = self.layers[ids](x[ids])

            batch_size, _, n_y, n_x = x[ids].shape
            x[ids] = (
                x[ids]
                .view(batch_size, self.n_anchors, self.n_outputs, n_y, n_x)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

        return x

    def initialize_biases(self, n_classes, size_factors, class_frequencies=None):
        """Initializes biases.

        Parameters
        ----------
        class_frequencies : torch.Tensor
            A priori distribution of class probabilities.

        """
        for layer, grid_size in zip(self.layers, size_factors):
            bias = layer.bias.view(self.n_anchors, -1)
            bias[:, 4] += math.log(8 / (640 / grid_size) ** 2)
            if class_frequencies is None:
                bias[:, 5:] += math.log(0.6 / (n_classes - 0.99))
            else:
                bias[:, 5:] += torch.log(class_frequencies / class_frequencies.sum())
            layer.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)
