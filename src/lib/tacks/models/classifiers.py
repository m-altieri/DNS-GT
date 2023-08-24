# -*- coding: utf-8 -*-
"""Classifier classes.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import torch

from ..data.imagenet import ID_TO_LABEL
from ..layers import LinearReg
from .base import TacksModel


class TacksClassifier(TacksModel):
    """Torch model for classification.

    Parameters
    ----------
    name, in_shape, device, half_precision, logger
        See :class:`TacksModel`.
    """

    def __init__(
        self,
        name,
        in_shape,
        n_classes,
        with_softmax=False,
        device=None,
        half_precision=False,
        logger=None,
        **kwargs,
    ):
        self.n_classes = n_classes
        self.with_softmax = with_softmax

        super().__init__(
            name=name,
            in_shape=in_shape,
            out_shape=(self.n_classes,),
            device=device,
            half_precision=half_precision,
            logger=logger,
            **kwargs,
        )

    @property
    def n_classes(self):
        return self._n_classes

    @n_classes.setter
    def n_classes(self, n_classes):
        if n_classes < 2:
            err_msg = 'Number of classes should be higher than 1 (given: {:d}'
            raise ValueError(err_msg.format(n_classes))

        self._n_classes = n_classes

    def predict(self, instances):
        """Return the outputs

        Parameters
        ----------
        instances : torch.Tensor
            See :class:`TacksModel`.

        Returns
        -------
        torch.Tensor
            Outputs of the model.
        tuple of torch.Tensor, optional
            Predicted labels and associated probabilities for the instances.
        """

        outputs = super().predict(instances)

        # if softmax is not already applied, apply it
        sm_outputs = (
            outputs
            if hasattr(self, 'with_softmax') and self.with_softmax
            else torch.softmax(outputs, 1)
        )

        # compute predictions on softmax outputs
        pred_probas, pred_labels = torch.max(sm_outputs, 1)

        return outputs, (pred_labels, pred_probas)

    def find_instance(self, data_loader, gt_label=None, pred_label=None):
        """Find an instance with given ground truth label and prescribed predicted label.

        The first instance to be found is returned. If no instance is found, return
        None.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
            Loader for the dataset.
        gt_label : int, or None, optional
            Ground truth label of the instance to find. If None, any label can be
            returned (default: None).
        pred_label : int, optional
            Predicted label of the instance to find. If None, any label can be
            predicted (default: None).

        Returns
        -------
        tuple of torch.Tensor or None
        """

        self.logger.warn('This method hasn\'t been used for a long time.')

        has_found = False
        for instance, label in data_loader:
            if gt_label is None or label[0] == gt_label:
                if pred_label is None:
                    has_found = True
                    break
                else:
                    _, (pred_labels, _) = self.predict(instance)
                    if pred_labels[0] == pred_label:
                        has_found = True
                        break

        if not has_found:
            self.logger.error('No instance found.')

        return instance, label

    def compute_gradients(
        self, instances, input_grad_module=None, grad_tensors=None, on_logits=False
    ):
        """Compute the derivatives of the outputs with respect to the inputs.

        The gradients of the features at the inputs of the prescribed module are
        returned.

        Parameters
        ----------
        instances : torch.Tensor
            Batch of instances.
        input_grad_module : str or None, optional
            Name of the module whose input features gradients are return. If None, the
            model is used as model, i.e. the gradients of the input data is returned
            (default: None).
        grad_tensors : torch.Tensor or None, optional
            Tensor whose gradients is computed with respect to. If None, a tensor is
            built to compute the gradients for the output with the highest value, i.e.
            the predicted output (default: None).
        on_logits : bool, optional
            Indicates if the gradient is computed on logits or not (default: False).

        Returns
        -------
        torch.Tensor
            Gradients.
        tuple of torch.Tensor
            Predicted labels with corresponding probabilities.
        """

        self.logger.warn('This method hasn\'t been used for a long time.')

        # define a hook to extract gradients
        def save_grads_hook(grads):
            self._grads = grads

        # define a hook to register the save gradients hook on the required input

        # WARNING: not working:
        # https://pytorch.org/docs/master/generated/torch.nn.Module.html?highlight=register#torch.nn.Module.register_backward_hook
        def register_backwardhook_hook(module, inputs):
            if not inputs[0].requires_grad:
                inputs[0].requires_grad = True
            self._hooks.append(inputs[0].register_hook(save_grads_hook))

        if input_grad_module is None:
            # if the gradient module name is None, select the model as grad module
            input_grad = self
        else:
            # otherwise select the corresponding module
            input_grad = self._modules[input_grad_module]

        # add the gradient hook on feature
        self._hooks = [input_grad.register_forward_pre_hook(register_backwardhook_hook)]

        # original device of instances
        orig_device = instances.device
        # size of the batch
        batch_size = instances.shape[0]

        # set the model in evaluation mode and clear gradients
        self.eval()
        self.zero_grad()

        # copy the inputs
        X = instances.clone().detach().to(self.device)

        # run the model
        logits = self(X)
        outputs = torch.softmax(logits, 1)

        # get predictions
        with torch.no_grad():
            pred_probas, pred_labels = torch.max(outputs, 1)

        pred_labels = pred_labels.squeeze().to(orig_device)
        pred_probas = pred_probas.squeeze().to(orig_device)

        # if the output on which the output is computed is not set, select the predicted
        # class
        if grad_tensors is None:
            grad_tensors = torch.zeros(outputs.shape, device=self.device)
            grad_tensors[torch.arange(batch_size), pred_labels] = 1

        grad_tensors = grad_tensors.to(self.device)

        if on_logits:
            outputs = logits

        if grad_tensors.ndim == outputs.ndim:
            outputs.backward(grad_tensors)
            grads = self._grads.clone().detach().to(orig_device)

        else:
            outputs.backward(grad_tensors[..., 0], retain_graph=True)
            grads = torch.zeros(self._grads.shape + grad_tensors.shape[-1:])
            grads[..., 0] = self._grads.clone().detach().to(orig_device)
            for ido in range(1, grad_tensors.shape[-1]):
                outputs.backward(grad_tensors[..., ido], retain_graph=True)
                grads[..., ido] = self._grads.clone().detach().to(orig_device)

        # remove hooks
        while self._hooks:
            self._hooks.pop().remove()

        return grads, (pred_labels, pred_probas)

    def get_predictions_as_strs(self, instances, id_to_label=None):
        """Return the predictions as a list of strings for each instance.

        Parameters
        ----------
        instances : torch.Tensor
            Batch of instances.
        id_to_label : dict or None
            Mapping from id to label. If None, not used.

        Returns
        -------
        list of str
            Prediction as text for each instance.
        """
        outputs, (pred_labels, pred_probas) = self.predict(instances)

        return [
            '{:02d}{:s} ({:.2f})'.format(
                pred_labels[ids].item(),
                '-' + id_to_label[pred_labels[ids].item()]
                if id_to_label is not None
                else '',
                pred_probas[ids].item(),
            )
            for ids in range(pred_labels.shape[0])
        ]


class Sequential2dClassifier(TacksClassifier):
    """Architecture made of sequential defined layers.

    Parameters
    ----------
    arch : list of tuple (str, dict)
        Ordered list of layers defining the architecture. First item is the name of the
        layer, second item is the parameters of the layer.
    n_classes : int
        Number of classes.
    with_softmax : bool, optional
        Indicates if a softmax activation is performed.
    name, in_shape, device, half_precision, logger :
        See :py:class:`TacksModel`.
    """

    def __init__(
        self,
        name,
        in_shape,
        arch,
        n_classes,
        with_softmax=False,
        device=None,
        half_precision=False,
        logger=None,
        **kwargs,
    ):
        # replace final layer with number of classes
        if (
            arch[-1][-1][0] == 'linreg'
            and arch[-1][-1][1]['lin_params']['out_features'] is None
        ):
            arch[-1][-1][1]['lin_params']['out_features'] = n_classes

        super().__init__(
            name=name,
            in_shape=in_shape,
            n_classes=n_classes,
            with_softmax=with_softmax,
            device=device,
            half_precision=half_precision,
            logger=logger,
            **kwargs,
        )

        self.build_architecture(arch)
        self._post_building()

    def forward(self, x):
        if hasattr(self, 'pre_process'):
            x = self.pre_process(x)

        for idl in range(len(self.arch)):
            layer_name = f'layer{idl}'

            self.logger.debug('%s| Forward: %s', self.name, layer_name)
            self.logger.debug('%s|    Input shape:  %s', self.name, x.shape)

            x = self._modules[layer_name](x)

            self.logger.debug('%s|    Output shape:  %s', self.name, x.shape)

        if hasattr(self, 'post_forward'):
            x = self.post_forward(x)

        if self.with_softmax:
            x = torch.softmax(x, 1)

        return x


class ImageNetClassifier(TacksClassifier):
    """Architecture to handle models pretrained on ImageNet.

    Parameters
    ----------
    imagenet_model : torch.Module
        Model used for the imagenet dataset.
    n_classes : int or None, optional
        Number of classes. If provided, the final layer is replaced (default: None).
    with_softmax : bool, optional
        Indicates if a softmax activation is performed.
    name, in_shape, device, half_precision, logger :
        See :py:class:`TacksModel`.
    Parameters

    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    id_to_label = ID_TO_LABEL

    def __init__(
        self,
        name,
        in_shape,
        imagenet_model,
        n_classes=None,
        with_softmax=False,
        device=None,
        half_precision=False,
        logger=None,
        **kwargs,
    ):
        if n_classes is None:
            n_classes = 1000

        super().__init__(
            name=name,
            in_shape=in_shape,
            n_classes=n_classes,
            with_softmax=with_softmax,
            device=device,
            half_precision=half_precision,
            logger=logger,
            **kwargs,
        )

        imagenet_model.eval()

        # add a first layer for the normalization of data
        normalization = torch.torch.nn.BatchNorm2d(3)
        normalization.running_mean[:] = torch.FloatTensor(self.mean)
        normalization.running_var[:] = torch.FloatTensor(self.std) ** 2

        # get the length of the output vector of the imagenet model
        out_length = imagenet_model(torch.zeros((1,) + self.in_shape)).shape[1]

        # add classifier layer of imagenet model
        if out_length != self.n_classes:
            classifier = LinearReg(
                {'in_features': out_length, 'out_features': self.n_classes},
                activation='linear',
            )
        else:
            classifier = torch.nn.Identity()

        self.add_module('normalization', normalization)
        self.add_module('imagenet_model', imagenet_model)
        self.add_module('classifier', classifier)

        self._post_building()

    def forward(self, x):
        x = self.normalization(x)
        x = self.imagenet_model(x)
        x = self.classifier(x)

        if self.with_softmax:
            x = torch.softmax(x, 1)

        return x

    def freeze_imagenet_model(self):
        """Freeze imagenet model parameters."""
        for param in self.imagenet_model.parameters():
            param.requires_grad = False

        for param in self.normalization.parameters():
            param.requires_grad = False
