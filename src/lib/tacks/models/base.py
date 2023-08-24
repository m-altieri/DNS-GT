# -*- coding: utf-8 -*-
"""Base class for TacksModels.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
from math import prod
from pathlib import Path

import torch
from torch import nn


from ..layers import ConvReg2d, ConvTransposeReg2d, LinearReg, Reshape, Residual2dBlock
from ..layers.base import LAYER_DIM_OUTPUT
from ..layers.utils import get_out_shape_after_convt2d, get_out_shape_after_maxpool2d
from ..utils import check_path, get_config, get_logger
from .utils import get_correct_device


class TacksModel(nn.Module):
    """Wrapper around torch.nn.Module to define models.

    Parameters
    ----------
    name : str
        Name of the model.
    in_shape : tuple of ints
        Shape of the inputs.
    device : ['cpu', 'cuda'] or int or None, optional
        Device to use. If None, select the best available device (default: None).
    half_precision : bool, optional
        Indicates if calculation is made using half precision or not (default: False).
    logger : logging.Logger, optional
        Logging system (default: None).
    """

    def __init__(
        self,
        name,
        in_shape,
        out_shape,
        device=None,
        half_precision=False,
        logger=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.name = name

        if logger is None:
            logger = get_logger(self.name, to_file=False)

        self.logger = logger
        self._debugger = None

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.half_precision = half_precision

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        # get correct device
        self._device = get_correct_device(device, self.logger)

        # send the model on the device
        self.to(self.device)

    @property
    def n_parameters(self):
        """Return the number of parameters of the model."""
        return sum(param.numel() for param in self.parameters())

    @property
    def n_trainable_parameters(self):
        """Return the number of trainable parameters of the model."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    @property
    def n_layers(self):
        """Return the number of layers"""
        return len(list(self.parameters()))

    @property
    def module_names(self):
        return [module[0] for module in self.named_modules()]

    def freeze_all_layers(self):
        """Freeze all layers of the model."""
        for param in self.parameters():
            param.requires_grad = False

    def info(self):
        """Print information about the model."""

        header_template = '{:5s} {:40s} {:9s} {:12s} {:20s} {:10s} {:10s}'
        param_template = '{:5d} {:40s} {:9s} {:12g} {:20s} {:10.3g} {:10.3g}'

        print(
            header_template.format(
                'layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'
            )
        )

        for idp, (name, param) in enumerate(self.named_parameters()):
            print(
                param_template.format(
                    idp,
                    name,
                    str(param.requires_grad),
                    param.numel(),
                    str(list(param.shape)),
                    param.mean(),
                    param.std(),
                )
            )

        print('Model Summary:')
        print(
            ' {:g} layers, {:g} parameters, {:g} trainable parameters.'.format(
                self.n_layers, self.n_parameters, self.n_trainable_parameters
            )
        )

    def fuse(self):
        """Fuse layers for better inference performances."""

        if self.logger:
            self.logger.debug('%s| Fusing layers.', self.name)

        for module in self.layers.modules():
            if hasattr(module, 'fuse'):
                module.fuse()
                if self.logger:
                    self.logger.debug('%s|    > %s fused.', module)

        return self

    @staticmethod
    def load(model_path, device=None, logger=None):
        """Load a full model.

        This includes the model state and the class name and attributes.

        Parameters
        ----------
        model_path : str or pathlib.Path
            Path to the file. If Path is relative, start from the default model folder
            set in the config.
        logger : logging.Logger or None
            Logging system.
        device :
            See :func:`get_correct_device`.
        """

        # convert into Path
        model_path = Path(model_path).expanduser()

        # add default model path if relative path
        if not model_path.is_absolute():
            model_path = get_config().get_path('paths', 'models') / model_path

        # init logger if None
        if logger is None:
            logger = get_logger(model_path.stem, to_file=False)

        # load the model in the device
        model = torch.load(model_path, map_location=get_correct_device(device, logger))

        if not isinstance(model, TacksModel):
            model = TacksModel(
                model,
                in_shape,
                out_shape,
                device=None,
                half_precision=False,
                logger=None,
                **kwargs,
            )

        model.device = device

        # set the logger
        model.logger = logger

        model._post_building()
        model.logger.info('%s| Loading model from %s', model.name, model_path)

        return model

    def save(self, model_path):
        """Save the model.

        This includes the model state and the class name and attributes.

        Parameters
        ----------
        model_path : str or pathlib.Path
            Path to the file.
        """

        # check that parent path exists
        check_path(model_path.parent)

        torch.save(self, model_path)

        if self.logger:
            self.logger.info('%s| Saving model to %s', self.name, model_path)

    def load_state(self, state):
        """Load the state of the model from path or dict.

        Parameters
        ----------
        state : str or pathlib.Path or dict
            Path to the file or dict containing the state.
        """
        if isinstance(state, str) or isinstance(state, Path):
            check_path(state)
            state = torch.load(state, map_location=torch.device('cpu'))

            if self.logger:
                self.logger.info('%s| Loading state from %s', self.name, state)

        elif isinstance(state, dict):
            if self.logger:
                self.logger.info('%s| Loading state from dictionary.', self.name)

        else:
            raise Exception('Unknown type for state: %s.', type(state))

        self.load_state_dict(state)
        self._post_building()

    def save_state(self, state_path):
        """Save the state of the model.

        Parameters
        ----------
        state_path : str or pathlib.Path
            Path to the file.
        """
        check_path(state_path.parent)

        torch.save(self.state_dict(), state_path)

        if self.logger:
            self.logger.debug('%s| Saving state to %s', self.name, state_path)

    def forward(self, x):
        raise NotImplementedError

    def initialize_weights(self, method=None, params=None):
        """Initialize the weights of the model.

        Parameters
        ----------
        method : {'uniform', 'normal', 'kaiming_normal'}
            Method of initialization.
        params : dict, optional
            Dict with parameters for the initializers (default: None).
        """
        if params is None:
            params = {}

        for layer_name, layer in self._modules.items():
            for module_name, module in layer._modules.items():
                if hasattr(module, 'initialize_weights'):
                    module.initialize_weights(method, params)

                    if self.logger:
                        self.logger.debug(
                            "%s| Module '%s' of type '%s': initialized with method '%s'",
                            self.name,
                            module_name,
                            type(module).__name__,
                            method,
                        )
                else:
                    if len(list(module.parameters())) > 0:
                        if self.logger:
                            self.logger.debug(
                                "%s| Module '%s' of type '%s': not initialized",
                                self.name,
                                module_name,
                                type(module).__name__,
                            )

    def predict(self, instances):
        """Return the prediction of a given batch of instances.

        This method is a wrapper around the __call__() method to make sure necessary
        operations (eval mode, device handling, etc.) before and after the computation
        are performed.

        Parameters
        ----------
        instances : torch.Tensor
            Batch of instances.

        Returns
        -------
        torch.Tensor
            Outputs of the model.
        """

        # check the shape of instances
        instances_shape = tuple(instances.shape[1::])
        if instances_shape != self.in_shape:
            err_msg = 'Wrong shape for inputs: got {}, expected {}'
            raise ValueError(err_msg.format(instances_shape, self.in_shape))

        # get original device of instances
        orig_device = instances.device

        with torch.no_grad():
            # set model in evaluation mode
            self.eval()

            # move if necessary instances to model device
            if orig_device != self.device:
                instances = instances.to(self.device)

            if self.half_precision:
                instances = instances.half()

            outputs = self(instances)

        # send to original device
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.to(orig_device)
        elif isinstance(outputs, list):
            outputs = [item.to(orig_device) for item in outputs]

        return outputs

    def publish(self, prefix=None, suffix=None):
        """Publish the model.

        The model is saved in the model folders under the name
        '{prefix}_{model_name}_{suffix}.pt'.

        Parameters
        ----------
        prefix : str or None
            Prefix to add to the name of the model.
        suffix : str or None
            Suffix to add to the name of the model.
        """
        # path to model folders
        model_path = get_config().get_path('paths', 'models')

        # default value for prefix
        if prefix is None:
            prefix = self.name

        # default value for suffix
        if suffix is None:
            suffix = ''
        else:
            suffix = f'_{suffix}'

        # get model name
        self.save(model_path / f'{prefix}{suffix}.pt')

    def get_features(self, instances, module_name, submodule_idx=None):
        """Return the features at the output of a given module, or at the output of a
        submodule of this module.

        Parameters
        ----------
        instances : torch.Tensor
            Batch of instances.
        module_name : str
            Name of the module.
        submodule_idx : int or None, optional
            Index of the submodule if the module is a Sequential structure. If None, use
            the output of the Sequential structure is returned (default: None).

        Returns
        -------
        torch.Tensor
        """

        self.logger.warn('This method hasn\'t been used for a long time.')

        def save_features_hook(module, inputs, outputs):
            self._features = outputs

        if module_name not in self.module_names:
            err_msg = 'Unknown module name: {}'
            raise KeyError(err_msg.format(module_name))

        # get the original device
        orig_device = instances.device

        # get the module to consider
        hier_mod_names = module_name.split('.')
        module = self
        for mod_name in hier_mod_names:
            module = module._modules[mod_name]

        if submodule_idx is not None:
            module = module[submodule_idx]

        # register a hook to the module
        hook_handler = module.register_forward_hook(save_features_hook)

        # evaluate the model
        self.eval()
        with torch.no_grad():
            self(instances.to(self.device))

        # remove the hook
        hook_handler.remove()

        return self._features.to(orig_device)

    def build_layers_from_arch_config(self, arch_config):
        """Build the layers of the model from an architecture config.

        Parameters
        ----------
        arch_config : list of tuple (str, dict)
            Ordered list of layers defining the architecture. First item is the name of
            the layer, second item is the parameters of the layer.

        """
        self.logger.info('%s| Start building layers...', self.name)
        self.logger.debug('%s| Input shape: %s', self.name, str(self.in_shape))

        # shape of the previous layer output (init: input shape)
        out_shape = self.in_shape

        for idl, list_modules in enumerate(arch_config):
            # counter for each types of module
            counters = {layer_type: 0 for layer_type in LAYER_DIM_OUTPUT.keys()}

            layer = nn.Sequential()

            for module_type, module_params in list_modules:
                self.logger.debug(
                    '%s| Adding module %s with parameters %s...',
                    self.name,
                    module_type,
                    str(module_params),
                )

                # name the layer
                module_name = f'{module_type}_{counters[module_type]}'

                # increase counter for the type of layer
                counters[module_type] += 1

                if LAYER_DIM_OUTPUT[module_type] is not None:
                    # check that `out_shape` is compatible with the layer
                    if len(out_shape) != LAYER_DIM_OUTPUT[module_type]:
                        err_msg = (
                            'Output before a {} layer should be {}D tuple (found: {}).'
                        )
                        raise ValueError(
                            err_msg.format(
                                module_type, LAYER_DIM_OUTPUT[module_type], out_shape
                            )
                        )

                if module_type == 'linreg':
                    # add the number of input features in parameters
                    module_params['lin_params'].update({'in_features': out_shape[0]})

                    # init the module
                    module = LinearReg(**module_params)

                    # get new output shape
                    out_shape = (module.out_features,)

                elif module_type == 'convreg2d':
                    # update the parameters of layer to add `in_channels`
                    module_params['conv_params'].update({'in_channels': out_shape[0]})

                    # init the module
                    module = ConvReg2d(**module_params)

                    # get new output shape
                    out_shape = module.get_output_shape(out_shape)

                elif module_type == 'convtreg2d':
                    # update the parameters of layer to add `in_channels`
                    module_params['convt_params'].update({'in_channels': out_shape[0]})

                    # init the module
                    module = ConvTransposeReg2d(**module_params)

                    # get new output shape
                    out_shape = module.get_output_shape(out_shape)

                elif module_type == 'residual2d':
                    # update the parameters of layer to add `in_channels`
                    module_params['conv_params'].update({'in_channels': out_shape[0]})

                    # init the module
                    module = Residual2dBlock(**module_params)

                    # get new output shape
                    out_shape = module.get_output_shape(out_shape)

                elif module_type == 'maxpool2d':
                    # init the module
                    module = nn.MaxPool2d(**module_params)

                    # get new output shape
                    out_shape = (out_shape[0],) + get_out_shape_after_maxpool2d(
                        out_shape[1::], **module_params
                    )

                elif module_type == 'dropout2d':
                    # init the module
                    module = nn.Dropout2d(**module_params)

                elif module_type == 'flatten':
                    # init the module
                    module = nn.Flatten(**module_params)

                    # get new output shape
                    out_shape = (prod(out_shape),)

                elif module_type == 'reshape':
                    # init the module
                    module = Reshape(**module_params)

                    # get new output shape
                    out_shape = module.shape

                else:
                    err_msg = 'Unknown layer type: {}'
                    raise ValueError(err_msg.format(module_type))

                layer.add_module(module_name, module)

                # log the operation for debug
                self.logger.debug(
                    '%s| Added %s: %s',
                    self.name,
                    module_name,
                    str(module),
                )
                self.logger.debug(
                    '%s| Output shape of layer %s: %s',
                    self.name,
                    module_name,
                    out_shape,
                )
            self.add_module(f'layer{idl}', layer)

        # check that out shape corresponds to the one provided by the user
        if list(self.out_shape) != list(out_shape):
            err_msg = 'Output shape {} does not match user-provided out shape {}.'
            raise ValueError(err_msg.format(out_shape, self.out_shape))

        # store the architecture
        self.arch_config = arch_config

        # log the final output shape
        self.logger.info('%s| Final output shape: %s', self.name, self.out_shape)
        self.logger.info('%s| Architecture built')

        self._post_building()

    def _post_building(self):
        """Perform operations after building the architecture."""

        # move model to device
        self.logger.debug('%s| Send model to device `%s`.', self.name, self.device)
        self.to(self.device)

        if self.half_precision:
            self.logger.debug('%s| Set model in half precision mode.', self.name)
            self.half()

        # set model in eval mode by default
        self.logger.debug('%s| Set model in evaluation mode.', self.name)
        self.eval()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__['logger'] = None
