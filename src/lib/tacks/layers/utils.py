# -*- coding: utf-8 -*-
"""Util functions for layers.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import torch.nn as nn


def get_activation_module(activation, act_params=None):
    """Return an activation module.

    Parameters
    ----------
    activation : ['linear', 'sigmoid', 'tanh', 'relu', 'leaky_relu']
        Activation function to apply to the outputs (default: 'linear').
    act_params : dict or None
        Parameters for the activation function.

    Returns
    -------
    nn.Module
        Module with the desired activation.
    """
    if act_params is None:
        act_params = {}

    if activation == 'linear':
        return nn.Identity()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'leaky_relu':
        if 'negative_slope' not in act_params:
            act_params['negative_slope'] = 0.1
        return nn.LeakyReLU(**act_params, inplace=True)
    elif activation == 'hardswish':
        return nn.Hardswish()
    else:
        raise ValueError(f'Unknown activation: {activation}')


def initialize_weights(weight, method, params):
    """Initialize weights using a given method.

    Parameters
    ----------
    method : {'uniform', 'normal', 'kaiming_normal', 'xavier_normal'}
        Method of initialization.
    params : dict
        Dict with parameters for the initializers.
    """

    if method == 'uniform':
        nn.init.uniform_(weight, **params)
    elif method == 'kaiming_normal':
        nn.init.kaiming_normal_(weight, **params)
    elif method == 'normal':
        nn.init.normal_(weight, **params)
    elif method == 'xavier_normal':
        nn.init.xavier_normal_(weight, **params)


def autopadding(kernel_size):
    """Finds the right padding, according to the size of the kernel, to
    preserve image dimensions (padding = 'same').

    Parameters
    ----------
    kernel_size : int
        Size of the kernel.

    Returns
    -------
    int
        Right padding.
    """
    return (
        kernel_size // 2
        if isinstance(kernel_size, int)
        else [x // 2 for x in kernel_size]
    )


def get_out_length_after_conv1d(
    in_length, kernel_size, stride=1, padding=0, dilation=1
):
    """Return the output length after passing a sequence of given length in a
    1D convolutional layer.

    Parameters
    ----------
    See torch.nn.Conv1d
    """
    return (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def get_out_length_after_convt1d(
    in_length, kernel_size, stride=1, padding=0, dilation=1
):
    """Return the output length after passing a sequence of given length in a 1D
    convolutional transpose layer.

    Parameters
    ----------
    See torch.nn.ConvTranspose1d
    """
    return (in_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1


def get_out_length_after_maxpool1d(
    in_shape, kernel_size, stride=None, padding=0, dilation=1
):
    """Return the output shape after passing a tensor of given shape in a
    1D maxpool layer.

    Parameters
    ----------
    See torch.nn.MaxPool1d
    """
    if stride is None:
        stride = kernel_size

    return get_out_length_after_conv1d(in_shape, kernel_size, stride, padding, dilation)


def get_out_shape_after_conv2d(in_shape, kernel_size, stride=1, padding=0, dilation=1):
    """Return the output shape after passing a tensor of given shape in a 2D
    convolutional layer.

    Parameters
    ----------
    in_shape : 2-tuple of int
        Size of the input.
    others
        See torch.nn.Conv2d
    """

    if len(in_shape) != 2:
        err_msg = 'Shape should be a 2-tuple: (given: {:s}).'
        raise ValueError(err_msg.format(str(in_shape)))

    kernel_size = (
        (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    )
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    return (
        get_out_length_after_conv1d(
            in_shape[0], kernel_size[0], stride[0], padding[0], dilation[0]
        ),
        get_out_length_after_conv1d(
            in_shape[1], kernel_size[1], stride[1], padding[1], dilation[1]
        ),
    )


def get_out_shape_after_convt2d(in_shape, kernel_size, stride=1, padding=0, dilation=1):
    """Return the output shape after passing a tensor of given shape in a 2D
    convolutional Transpose layer.

    Parameters
    ----------
    in_shape : 2-tuple of int
        Size of the input.
    others
        See torch.nn.ConvTranspose2d
    """

    if len(in_shape) != 2:
        err_msg = 'Shape should be a 2-tuple: (given: {:s}).'
        raise ValueError(err_msg.format(str(in_shape)))

    kernel_size = (
        (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    )
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    return (
        get_out_length_after_convt1d(
            in_shape[0], kernel_size[0], stride[0], padding[0], dilation[0]
        ),
        get_out_length_after_convt1d(
            in_shape[1], kernel_size[1], stride[1], padding[1], dilation[1]
        ),
    )


def get_out_shape_after_maxpool2d(
    in_shape, kernel_size, stride=None, padding=0, dilation=1
):
    """Return the output shape after passing a tensor of given shape in a
    2D maxpool layer.

    Parameters
    ----------
    in_shape : 2-tuple of int
        Size of the input.
    others
        See torch.nn.MaxPool2d
    """

    if len(in_shape) != 2:
        err_msg = 'Shape should be a 2-tuple: (given: {:s}).'
        raise ValueError(err_msg.format(str(in_shape)))


    if stride is None:
        stride = kernel_size

    return get_out_shape_after_conv2d(in_shape, kernel_size, stride, padding, dilation)
