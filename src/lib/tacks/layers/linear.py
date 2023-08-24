"""Definition of linear layers.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import torch.nn as nn

from .utils import get_activation_module, initialize_weights


class LinearReg(nn.Sequential):
    """Linear layer followed by a regularization layer and an activation function.

    Parameters
    ----------
    lin_params : dict
        Parameters of the linear layer. See :py:class:`nn.Linear`.
    regularization : ['bn', 'dropout'] or None
        Type of regularization.
    reg_params : dict or None
        Parameters of the regularization layer. See :py:class:`nn.BatchNorm1d` or
        :py:class:`nn.Dropout1d`.
    activation, act_params
        See :py:func:`get_activation_function`.
    """

    def __init__(
        self,
        lin_params,
        regularization="none",
        reg_params=None,
        activation="linear",
        act_params=None,
    ):

        super().__init__()

        self.fused = False
        self.regularization = regularization

        if "in_features" not in lin_params:
            raise ValueError("`in_features` should be defined in `lin_params`")

        if "out_features" not in lin_params or lin_params["out_features"] is None:
            raise ValueError("`out_features` should be defined in `lin_params`")

        _lin_params = {"bias": False}
        _lin_params.update(lin_params)

        lin_layer = nn.Linear(**_lin_params)

        if self.regularization == "bn":

            if reg_params is None:
                reg_params = {"eps": 1e-5, "momentum": 0.1}

            reg_layer = nn.BatchNorm1d(lin_params["out_features"], **reg_params)

        elif self.regularization == "dropout":

            if reg_params is None:
                reg_params = {"p": 0.5}

            reg_layer = nn.Dropout(**reg_params)

        elif self.regularization == "none":
            reg_layer = nn.Identity()

        else:
            err_msg = "Unknown regularization: {}"
            raise ValueError(err_msg.format(regularization))

        act_layer = get_activation_module(activation, act_params)

        self.add_module("lin", lin_layer)
        self.add_module("reg", reg_layer)
        self.add_module("act", act_layer)

    @property
    def in_features(self):
        return self.lin.in_features

    @property
    def out_features(self):
        return self.lin.out_features

    def forward(self, x):
        if self.fused:
            raise NotImplementedError
        else:
            x = self.lin(x)
            x = self.reg(x)
            x = self.act(x)
            return x

    def fuse(self):
        raise NotImplementedError

    def initialize_weights(self, method=None, params=None):
        """Initialize the weights of the layer.

        Parameters
        ----------
        See 'func:`initialize_weights`.
        """
        if method is None:
            if self.act._get_name().lower() in ["leaky_relu", "relu"]:
                method = "kaiming_normal"
            else:
                method = "xavier_normal"

        if params is None:
            params = {}

        lin_bias = params.pop("lin_bias") if "lin_bias" in params else 0
        bn_bias = params.pop("bn_bias") if "bn_bias" in params else 0

        if method == "kaiming_normal":
            if "mode" not in params:
                params["mode"] = "fan_out"
            if "nonlinearity" not in params:
                params["nonlinearity"] = self.act._get_name().lower()

        initialize_weights(self.lin.weight, method, params)

        if self.lin.bias is not None:
            nn.init.constant_(self.lin.bias, lin_bias)

        if self.regularization == "bn":
            initialize_weights(self.reg.weight, method, params)
            nn.init.constant_(self.reg.bias, bn_bias)
