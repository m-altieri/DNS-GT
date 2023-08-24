# -*- coding: utf-8 -*-
""" filters of convoluta

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import torch


def visualize_convolutionallayers(classifier, instances, gt_classes,
                                 baseline=None, n_steps=50, on_logits=False):
    """Compute the integrated gradients.

    Integrated gradients is a technique for attributing a model prediction
    to its input features. the method also returns some additional
    debugging information for sanity checking the computation.

    References
    ----------
    [STY2017Axiomatic] Sundararajan, M.; Taly, A. & Yan, Q. `Axiomatic
    attribution for deep networks`

    Parameters
    ----------
    classifier : Classifier
        Classifier.
    instances : torch.Tensor
        Batch of instances for which integrated gradients are computed.
    classes : int
        Classes for which integrated gradients are computed.
    baseline : torch.Tensor, optional
        Baseline input used in the integrated gradients computation. If
        None, the all zero tensor with the same shape as an instance is
        used.
    n_steps : int, optional
        Number of interpolation steps between the baseline and the input used
        in the integrated gradients computation. These steps along determine
        the integral approximation error.
    in_grad : nn.Module
        Module whose input features are used to get the gradients. If None, the
        first module is used.
    on_logits : bool, optional
        Indicates if the gradient is computed on logits or not.

    Returns
    -------
    torch.Tensor
        Integrated_gradients of the prediction for the provided class to
        the provided instances.
    tuple of torch.Tensor
        Best classes and associated probabilities for the scaled instances.

    Raises
    ------
    ValueError
        If the provided baseline and the instance do not have the same
        shape.
    """

    # get the batch size and the number of classes
    batch_size = instances.shape[0]
    n_classes = classifier.n_classes

    # get the representation features at the right layer
    if baseline is None:
        baseline = torch.zeros(instances.shape[1::])

    if baseline.shape != instances.shape[1::]:
        err_msg = 'Shape of the gradient input {} and shape of the '\
            'baseline {} are not the same.'
        raise ValueError(err_msg.format(list(baseline.shape),
                                        list(instances.shape[1::])))

    # scale the inputs
    scaling = torch.arange(0.0, n_steps + 1.0).repeat(batch_size) / n_steps
    while scaling.ndim < instances.ndim:
        scaling = scaling.unsqueeze(-1)

    scaled_features = scaling * (instances.repeat_interleave(n_steps + 1, 0) -
                                 baseline)
    # compute gradients
    out_grad = torch.zeros((scaled_features.shape[0], n_classes))
    out_grad[torch.arange(scaled_features.shape[0]), gt_classes] = 1
    grads, (pred_classes, pred_probas) = classifier.compute_gradients(
        scaled_features, grad_tensors=out_grad, on_logits=on_logits)

    # reshape grads to extract the step dimension
    grads_shape = (batch_size, n_steps + 1) + instances.shape[1::] + (1,)
    grads = grads.reshape(grads_shape)

    igrads = grads.mean(1)
    tiled_inputs = (instances - baseline).unsqueeze(-1).repeat_interleave(
        igrads.shape[-1], -1)
    igrads = (tiled_inputs * igrads).squeeze(-1)

    return igrads.detach(), (pred_classes, pred_probas)
