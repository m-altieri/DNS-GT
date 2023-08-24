"""Definition of convolutional layers.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import torch
import torch.nn as nn

from .utils import (
    autopadding,
    get_out_shape_after_conv2d,
    get_out_shape_after_convt2d,
    get_activation_module,
    initialize_weights,
)


class ConvReg2d(nn.Module):
    """Convolutional layer followed by a regularization layer and an activation
    function.

    Parameters
    ----------
    conv_params : dict
        Parameters of the convolutional layer. See :py:class:`nn.Conv2d`.
    spectral_norm : bool, optional
        Indicates whether spectral normalization (i.e. normalization of the weight
        matrix of the convolutional layer by its spectral norm) is used or not (default:
        False).
    regularization : ['bn', 'dropout'] or None
        Type of regularization.
    reg_params : dict or None
        Parameters of the regularization layer. See :py:class:`nn.BatchNorm2d` or
        :py:class:`nn.Dropout2d`.
    activation, act_params
        See :py:func:`get_activation_function`.
    """

    def __init__(
        self,
        conv_params,
        regularization='none',
        reg_params=None,
        activation='linear',
        act_params=None,
    ):

        super().__init__()

        self.fused = False
        self.regularization = regularization
        self.spectral_norm = (
            conv_params.pop('spectral_norm')
            if 'spectral_norm' in conv_params
            else False
        )

        if 'in_channels' not in conv_params:
            raise ValueError('`in_channels` should be defined in `conv_params`')

        if 'out_channels' not in conv_params:
            raise ValueError('`out_channels` should be defined in `conv_params`')

        _conv_params = {
            'kernel_size': 1,
            'stride': 1,
            'padding': None,
            'dilation': 1,
            'groups': 1,
            'bias': False,
        }

        _conv_params.update(conv_params)

        if _conv_params['padding'] is None:
            _conv_params['padding'] = autopadding(_conv_params['kernel_size'])

        conv_layer = nn.Conv2d(**_conv_params)

        if self.spectral_norm:
            conv_layer = torch.nn.utils.spectral_norm(conv_layer)

        if self.regularization == 'bn':

            if reg_params is None:
                reg_params = {'eps': 1e-5, 'momentum': 0.1}

            reg_layer = nn.BatchNorm2d(conv_params['out_channels'], **reg_params)

        elif self.regularization == 'dropout':

            if reg_params is None:
                reg_params = {'p': 0.5}

            reg_layer = nn.Dropout2d(**reg_params)

        elif self.regularization == 'none':
            reg_layer = nn.Identity()

        else:
            err_msg = 'Unknown regularization: {}'
            raise ValueError(err_msg.format(regularization))

        act_layer = get_activation_module(activation, act_params)

        self.add_module('conv', conv_layer)
        self.add_module('reg', reg_layer)
        self.add_module('act', act_layer)

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def kernel_size(self):
        return self.conv.kernel_size

    @property
    def stride(self):
        return self.conv.stride

    @property
    def dilation(self):
        return self.conv.dilation

    @property
    def padding(self):
        return self.conv.padding

    def forward(self, x):
        return self.act(self.reg(self.conv(x)))

    def fuse(self):
        """Fuse convolutional layer with batch normalisation layer.

        See https://tehnokv.com/posts/fusing-batchnorm-and-conv/ for details.
        """

        if self.regularization == 'bn':
            with torch.no_grad():
                # init the fused conv
                fused_conv = nn.Conv2d(
                    self.conv.in_channels,
                    self.conv.out_channels,
                    kernel_size=self.conv.kernel_size,
                    stride=self.conv.stride,
                    padding=self.conv.padding,
                    bias=self.conv.bias,
                ).to(self.conv.weight.device)

                # prepare filters
                w_conv = self.conv.weight.clone().view(self.conv.out_channels, -1)
                w_bn = torch.diag(
                    self.reg.weight.div(torch.sqrt(self.reg.eps + self.reg.running_var))
                )
                fused_conv.weight.copy_(
                    torch.mm(w_bn, w_conv).view(fused_conv.weight.size())
                )

                # prepare spatial bias
                b_conv = (
                    torch.zeros(
                        self.conv.weight.size(0), device=self.conv.weight.device
                    )
                    if self.conv.bias is None
                    else self.conv.bias
                )
                b_bn = self.reg.bias - self.reg.weight.mul(self.reg.running_mean).div(
                    torch.sqrt(self.reg.running_var + self.reg.eps)
                )
                fused_conv.bias.copy_(
                    torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn
                )

            self.conv = fused_conv
            self.reg = None
            self.with_bn = False

    def initialize_weights(self, method=None, params=None):
        """Initialize the weights of the layer.

        Parameters
        ----------
        See 'func:`initialize_weights`.
        """

        if method is None:
            if self.act._get_name().lower() in ['leaky_relu', 'relu']:
                method = 'kaiming_normal'
            else:
                method = 'xavier_normal'

        if params is None:
            params = {}

        bn_bias = params.pop('bn_bias') if 'bn_bias' in params else 0

        if method == 'kaiming_normal':
            if 'mode' not in params:
                params['mode'] = 'fan_out'
            if 'nonlinearity' not in params:
                params['nonlinearity'] = self.act._get_name().lower()

        initialize_weights(self.conv.weight, method, params)
        if self.regularization == 'bn':
            initialize_weights(self.reg.weight, method, params)
            nn.init.constant_(self.reg.bias, bn_bias)

    def get_output_shape(self, in_shape):
        """Return the output shape, given an input shape.

        The output shape is calculated without feed-forward pass.

        Parameters
        ----------
        in_shape : 3-tuple of int
           Shape of the input, if the CWH format.

        Returns
        -------
        3-tuple of int
           Shape of the output, if the CWH format.
        """

        if in_shape[0] != self.conv.in_channels:
            err_msg = 'Number of input channels incorrect: {} (expected: {}).'
            raise ValueError(err_msg.format(in_shape[0], self.conv.in_channels))

        return (self.conv.out_channels,) + get_out_shape_after_conv2d(
            in_shape=in_shape[1::],
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
        )

    def __str__(self):
        string = ' - '.join((str(self.conv), str(self.reg), str(self.act)))
        return string


class ConvTransposeReg2d(nn.Module):
    """Transpose convolutional layer followed by a regularization layer and an
    activation function.

    Parameters
    ----------
    conv_params : dict
        Parameters of the convolutional layer. See :py:class:`nn.ConvTranspose2d`.
    spectral_norm : bool, optional
        Indicates whether spectral normalization (i.e. normalization of the weight
        matrix of the convolutional layer by its spectral norm) is used or not (default:
        False).
    regularization : ['bn', 'dropout'] or None
        Type of regularization.
    reg_params : dict or None
        Parameters of the regularization layer. See :py:class:`nn.BatchNorm2d` or
        :py:class:`nn.Dropout2d`.
    activation, act_params
        See :py:func:`get_activation_function`.
    """

    def __init__(
        self,
        convt_params,
        regularization='none',
        reg_params=None,
        activation='linear',
        act_params=None,
    ):

        super().__init__()

        self.fused = False
        self.regularization = regularization

        if 'in_channels' not in convt_params:
            raise ValueError('`in_channels` should be defined in `convt_params`')

        if 'out_channels' not in convt_params:
            raise ValueError('`out_channels` should be defined in `convt_params`')

        _convt_params = {
            'kernel_size': 1,
            'stride': 1,
            'padding': 0,
            'dilation': 1,
            'groups': 1,
            'bias': False,
        }

        _convt_params.update(convt_params)

        if _convt_params['padding'] is None:
            _convt_params['padding'] = autopadding(_convt_params['kernel_size'])

        convt_layer = nn.ConvTranspose2d(**_convt_params)

        if self.regularization == 'bn':

            if reg_params is None:
                reg_params = {'eps': 1e-5, 'momentum': 0.1}

            reg_layer = nn.BatchNorm2d(convt_params['out_channels'], **reg_params)

        elif self.regularization == 'dropout':

            if reg_params is None:
                reg_params = {'p': 0.5}

            reg_layer = nn.Dropout2d(**reg_params)

        elif self.regularization == 'none':
            reg_layer = nn.Identity()

        else:
            err_msg = 'Unknown regularization: {}'
            raise ValueError(err_msg.format(regularization))

        act_layer = get_activation_module(activation, act_params)

        self.add_module('convt', convt_layer)
        self.add_module('reg', reg_layer)
        self.add_module('act', act_layer)

    def forward(self, x):
        return self.activation(self.reg_layer(self.convt(x)))

    def fuse(self):
        raise NotImplementedError

    def initialize_weights(self, method=None, params=None):
        """Initialize the weights of the layer.

        Parameters
        ----------
        method : {'uniform', 'normal', 'kaiming_normal'}, optional
            Method of initialization. If None, use 'kaiming_normal' (default: None).
        params : dict, optional
            Dict with parameters for the initializers (default: None).
        """

        if method is None:
            method = 'kaiming_normal'
        if params is None:
            params = {}

        bn_bias = params.pop('bn_bias') if 'bn_bias' in params else 0

        if method == 'kaiming_normal':
            if 'mode' not in params:
                params['mode'] = 'fan_out'
            if 'activation' not in params:
                params['activation'] = self.activation

        initialize_weights(self.convt.weight, method, params)

        if self.with_bn:
            initialize_weights(self.bn.weight, method, params)
            nn.init.constant_(self.bn.bias, bn_bias)

    def get_output_shape(self, in_shape):
        """Return the output shape, given an input shape.

        The output shape is calculated without feed-forward pass.

        Parameters
        ----------
        in_shape : 3-tuple of int
           Shape of the input, if the CWH format.

        Returns
        -------
        3-tuple of int
           Shape of the output, if the CWH format.
        """

        if in_shape[0] != self.convt.in_channels:
            err_msg = 'Number of input channels incorrect: {) (expected: {}).'
            raise ValueError(err_msg.format(in_shape[0], self.conv.in_channels))

        return (self.convt.out_channels,) + get_out_shape_after_convt2d(
            in_shape=in_shape[1::],
            kernel_size=self.convt.kernel_size,
            stride=self.convt.stride,
            padding=self.convt.padding,
            dilation=self.convt.dilation,
        )


class Residual2dBlock(nn.Module):
    """Residual block made of two convolutional layers.

    Parameters
    ----------
    conv_params, regularization, reg_params, activation, act_params
        See :py:class:`ConvReg2d`.
    """

    def __init__(
        self,
        conv_params,
        regularization='none',
        reg_params=None,
        activation='linear',
        act_params=None,
    ):

        super().__init__()

        self.add_module(
            'convreg2d_1',
            ConvReg2d(
                conv_params=conv_params,
                regularization=regularization,
                reg_params=reg_params,
                activation=activation,
                act_params=act_params,
            ),
        )

        conv_params_2 = conv_params.copy()
        conv_params_2['in_channels'] = conv_params['out_channels']
        conv_params_2['stride'] = 1
        conv_params_2['padding'] = 1

        self.add_module(
            'convreg2d_2',
            ConvReg2d(
                conv_params=conv_params_2,
            ),
        )

        # add projection layer if out shape is different from input shape
        out_shape = self.get_output_shape((conv_params['in_channels'], 256, 256))

        conv_params_proj = conv_params.copy()
        conv_params_proj['kernel_size'] = 1
        conv_params_proj['padding'] = 0
        conv_params_proj['dilation'] = 1
        conv_params_proj['spectral_norm'] = False

        if out_shape != (conv_params['out_channels'], 256, 256):

            # define projection
            projection_layer = ConvReg2d(conv_params=conv_params_proj)

        else:
            projection_layer = nn.Identity()

        self.add_module('convreg2d_proj', projection_layer)

    def forward(self, x):
        x_conv = self.convreg2d_1(x)
        x_conv = self.convreg2d_2(x_conv)

        return x_conv + self.convreg2d_proj(x)

    def get_output_shape(self, in_shape):
        """Return the output shape, given an input shape.

        The output shape is calculated without feed-forward pass.

        Parameters
        ----------
        in_shape : 3-tuple of int
           Shape of the input, if the CWH format.

        Returns
        -------
        3-tuple of int
           Shape of the output, if the CWH format.
        """
        out_shape = self.convreg2d_1.get_output_shape(in_shape)
        out_shape = self.convreg2d_2.get_output_shape(out_shape)

        if in_shape[0] != self.convreg2d_1.in_channels:
            err_msg = 'Number of input channels incorrect: {} (expected: {}).'
            raise ValueError(err_msg.format(in_shape[0], self.convreg2d_1.in_channels))

        return (self.convreg2d_1.out_channels,) + get_out_shape_after_conv2d(
            in_shape=in_shape[1::],
            kernel_size=self.convreg2d_1.kernel_size,
            stride=self.convreg2d_1.stride,
            padding=self.convreg2d_1.padding,
            dilation=self.convreg2d_1.dilation,
        )

    def __str__(self):
        string = 'ResBlock: '
        string += ' - '.join(
            (str(self.convreg2d_1), str(self.convreg2d_2), str(self.convreg2d_proj))
        )
        return string
