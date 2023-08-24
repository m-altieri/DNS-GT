# -*- coding: utf-8 -*-
"""Test of the module :module:`tacks.explainability.integrated_gradients`

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>, <rhamon@protonmail.com>
"""
import unittest
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


from torchvision import models

from tacks.classifier import Classifier
from tacks.explainability import compute_integrated_gradients

HOME_PATH = Path('~').expanduser()

UTKU_PATH = HOME_PATH / 'repos' / 'pytorch-cnn-visualizations' / 'src'
sys.path.append(str(UTKU_PATH))


class TestImageExamples(unittest.TestCase):

    def setUp(self):
        self.images_path = Path('input_images/')

    def tearDown(self):
        pass

    def test_on_alexnet(self):

        from integrated_gradients import IntegratedGradients
        from misc_functions import preprocess_image

        # load the model
        utku_model = models.alexnet(pretrained=True)
        IG_utku = IntegratedGradients(utku_model)

        tacks_model = models.alexnet(pretrained=True)
        tacks_model.n_classes = 1000

        classifier = Classifier(tacks_model)

        for image_path in self.images_path.glob('spider.png'):

            if image_path.stem == 'snake':
                gt_class = 56
            elif image_path.stem == 'spider':
                gt_class = 72
            elif image_path.stem == 'cat_dog':
                gt_class = 243
            else:
                gt_class = None

            # open the image
            original_image = Image.open(image_path).convert('RGB')
            # process image
            prep_image = preprocess_image(original_image)

            instances = prep_image
            gt_classes = torch.LongTensor([gt_class])

            # utku approach
            grads_utku = IG_utku.generate_gradients(instances, gt_class)
            igrads_utku = IG_utku.generate_integrated_gradients(
                instances, gt_class, 50)
            igrads_utku *= instances.detach().squeeze(0).numpy()

            # tacks approach
            out_grad = torch.zeros((1, 1000))
            out_grad[0, gt_class] = 1
            grads, (pred_classes, pred_probas) = classifier.compute_gradients(
                instances, out_grad=out_grad, on_logits=True)
            igrads, _ = compute_integrated_gradients(
                classifier, instances, gt_classes, n_steps=50, on_logits=True)

            # test results
            np.testing.assert_almost_equal(grads_utku, grads[0, ...].numpy())
            np.testing.assert_almost_equal(
                igrads_utku, igrads[0, ...].numpy(), decimal=3)

            torch.cuda.empty_cache()
