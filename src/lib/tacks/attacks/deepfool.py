import torch


def deepfool(model, adv_instance, gt_label, n_classes, max_iter=100):

    iteration = 0
    adv_label = gt_label

    # forward pass the instance through the model
    output = model(adv_instance)

    # perturbation
    adv_perturbation = torch.zeros(adv_instance.shape)

    while adv_label == gt_label and iteration < max_iter:

        # calculate gradients of model in backward pass
        output[0, gt_label].backward(retain_graph=True)
        grad_gtlabel = adv_instance.grad.clone()
        lowest_perturbation_norm = torch.norm(adv_instance)

        for label in [idl for idl in range(n_classes) if idl != gt_label]:

            # clear the gradients
            adv_instance.grad.zero_()
            # compute gradients with respect to the given label
            output[0, label].backward(retain_graph=True)
            grad = adv_instance.grad

            # linearize the problem
            w = grad - grad_gtlabel
            output_at_label = output[0, label] - output[0, gt_label]

            # compute the norm of the perturbation
            perturbation_norm = (torch.abs(output_at_label) /
                                 torch.norm(w.flatten()))

            if perturbation_norm < lowest_perturbation_norm:
                lowest_perturbation_norm = perturbation_norm
                lowest_norm_w = w

        adv_perturbation_iter = (lowest_perturbation_norm * lowest_norm_w /
                                 torch.norm(lowest_norm_w)).detach()

        adv_perturbation += 1.02 * adv_perturbation_iter
        adv_instance = adv_instance.detach()
        adv_instance = torch.Tensor(adv_instance +
                                    1.02 * adv_perturbation_iter)
        adv_instance.requires_grad = True

        output = model(adv_instance)
        _, adv_label = torch.max(output, 1)
        adv_label = adv_label.item()

        iteration += 1

    print('Adversarial found at iteration {}'.format(iteration))
    if iteration == max_iter:
        print('MAX ITER REACHED')

    return adv_perturbation
