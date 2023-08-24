# -*- coding: utf-8 -*-
"""Testing of attacks.graphite

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import pytest
import torch

import tacks.attacks.physical.graphite as tacks_graphite
from tacks.archs.classification.gtsrbnet import GTSRBNet
from tacks.data.image import load_image
from tacks.models import TacksModel
from tacks.utils.image import resize_image, opencv_to_torch_img
from tacks.utils import get_config, Workspace, set_seed

import tacks.external.graphite_bridge as graphite_bridge

workspace = Workspace(name="GRAPHITE", instance_name="Testing")
tacks_config = get_config()


@pytest.fixture(scope="class")
def get_gtsrb_dataset(request):
    """Get the GTSRB dataset."""

    from tacks.data.gtsrb import get_loaders

    img_size = 32
    n_workers = 4
    batch_size = 128

    (
        request.cls.data_loaders,
        request.cls.loader_sizes,
        request.cls.extras,
    ) = get_loaders(batch_size, n_workers, img_size=img_size, logger=workspace.logger)


@pytest.fixture(scope="class")
def get_models(request):
    """Get pretrained models for testing."""

    workspace.logger.info("Loading GTRBNet from GRAPHITE checkpoint...")
    from tacks.external.graphite_bridge import ORIG_GRAPHITE_PATH

    tacks_model = GTSRBNet(logger=workspace.logger)
    checkpoint_path = ORIG_GRAPHITE_PATH / "GTSRB" / "checkpoint_us.tar"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found {checkpoint_path}.")

    tacks_model.load_state(torch.load(checkpoint_path)["model_state_dict"])
    workspace.logger.info("Done.")

    tacks_model.clip_values = (0, 1)
    tacks_model.with_softmax = False
    tacks_model._debugger = None
    tacks_model.eval()

    orig_model = graphite_bridge.get_gtsrbnet_model()
    orig_model.to(tacks_model.device)
    orig_model.eval()

    request.cls.tacks_model = tacks_model
    request.cls.orig_model = orig_model


@pytest.fixture(scope="class")
def get_random_image(request):
    """Get pretrained models for testing."""

    request.cls.img = torch.rand(3, 244, 244)


@pytest.mark.usefixtures("get_gtsrb_dataset")
@pytest.mark.usefixtures("get_models")
class TestModel:
    @pytest.mark.parametrize("split_name", ["train", "valid", "test"])
    def test_inference(self, split_name):
        device = self.tacks_model.device

        for instances, gt_labels in self.data_loaders[split_name]:
            instances = instances.to(device)

            orig_logits = self.orig_model(instances)
            tacks_logits = self.tacks_model(instances)

            torch.testing.assert_close(tacks_logits, orig_logits)


@pytest.mark.usefixtures("get_models")
class TestGraphiteWhiteboxAttack:
    @pytest.mark.parametrize("gt_label", [14])
    @pytest.mark.parametrize("tg_label", [0])
    @pytest.mark.parametrize("gradient_shrink", [False])
    @pytest.mark.parametrize("aligned", [True])
    @pytest.mark.parametrize("step_size", [0.01, 0.0156862745, 0.02])
    @pytest.mark.parametrize("n_transforms", [1, 10, 50])
    @pytest.mark.parametrize("min_eot_threshold", [0.8])
    @pytest.mark.parametrize("n_iterations", [1, 20, 50])
    @pytest.mark.parametrize("seed", [42])
    @pytest.mark.parametrize("patch_removal_size", [-1])
    @pytest.mark.parametrize("patch_removal_interval", [15.25])
    def test_generate(
        self,
        patch_removal_interval,
        patch_removal_size,
        seed,
        n_iterations,
        min_eot_threshold,
        n_transforms,
        step_size,
        aligned,
        gradient_shrink,
        tg_label,
        gt_label,
    ):
        # load an image
        data_path = (
            graphite_bridge.ORIG_GRAPHITE_PATH
            / "inputs"
            / "GTSRB"
            / "images"
            / f"{gt_label}.png"
        )

        img_size = (244, 244)

        img = load_image(str(data_path))
        img = opencv_to_torch_img(img)

        imgs = img.unsqueeze(0)

        # load the mask
        mask = load_image(
            graphite_bridge.ORIG_GRAPHITE_PATH
            / "inputs"
            / "GTSRB"
            / "Hulls"
            / f"{gt_label}.png",
            color_repr=None,
        )
        mask, _ = resize_image(mask, img_size, best_interpolation=False)
        mask = opencv_to_torch_img(mask)
        mask = (mask > (128 / 255)).float()

        # tacks attack
        tacks_attack_params = {
            "gt_label": gt_label,
            "gradient_shrink": gradient_shrink,
            "aligned": aligned,
            "n_transforms": n_transforms,
            "min_eot_threshold": min_eot_threshold,
            "patch_removal_size": patch_removal_size,
            "patch_removal_interval": patch_removal_interval,
            "n_patches": 4,
        }

        tacks_optim_params = {
            "n_max_epochs": None,
            "step_size": step_size,
            "n_iterations": n_iterations,
            "n_iterations_first_epoch": 500,
            "loss_function": torch.nn.CrossEntropyLoss(),
        }

        set_seed(seed, deterministic=True)

        # for iterating the random generator as in GRAPHITE script
        _ = GTSRBNet()

        tacks_attack = tacks_graphite.GRAPHITEWhiteBoxAttack(
            self.tacks_model, original=True, logger=workspace.logger
        )
        tacks_adv_img = tacks_attack.generate(
            imgs,
            torch.tensor([tg_label]).long(),
            mask=mask,
            attack_params=tacks_attack_params,
            optim_params=tacks_optim_params,
        )

        tacks_mask = tacks_attack.mask
        tacks_eot_robustness = tacks_attack.eot_robustness
        tacks_nqueries = tacks_attack.n_queries

        # -- GRAPHITE

        graphite_attack_params = {
            "gradient_shrink": gradient_shrink,
            "aligned": aligned,
            "victim": gt_label,
            "target": tg_label,
            "step_size": step_size,
            "num_xforms": n_transforms,
            "min_tr": min_eot_threshold,
            "iters": n_iterations,
            "seed": seed,
            "patch_removal_size": patch_removal_size,
            "patch_removal_interval": patch_removal_interval,
        }

        graphite_outs = graphite_bridge.run_graphite_attack(graphite_attack_params)

        graphite_adv_img = graphite_outs["adv_img"]
        graphite_mask = graphite_outs["mask"]
        graphite_eot_robustness = graphite_outs["eot_robustness"]
        graphite_nqueries = graphite_outs["n_queries"]

        # testing
        torch.testing.assert_close(tacks_adv_img, graphite_adv_img)
        torch.testing.assert_close(tacks_mask, graphite_mask)
        assert (
            tacks_eot_robustness == graphite_eot_robustness
        ), f"EOT robustness: {tacks_eot_robustness} != {graphite_eot_robustness}"

        assert (
            tacks_nqueries == graphite_nqueries
        ), f"N queries: {tacks_nqueries} != {graphite_nqueries}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 5])
    @pytest.mark.parametrize("nps", [True, False])
    @pytest.mark.parametrize("gt_label", [0, 14, 28, 40])
    @pytest.mark.parametrize("n_transforms", [1, 10, 100])
    @pytest.mark.parametrize("blur_kernels", [[0], [0, 3], [0, 3, 5], [0, 3, 5, 7]])
    @pytest.mark.parametrize("max_dist", [5, 10, 15])
    @pytest.mark.parametrize("min_angle", [-40, -50, -60])
    @pytest.mark.parametrize("max_angle", [40, 50, 60])
    def test_transform(
        self,
        min_angle,
        max_angle,
        max_dist,
        blur_kernels,
        n_transforms,
        gt_label,
        nps,
        seed,
    ):
        set_seed(seed)

        pt_file = (
            graphite_bridge.ORIG_GRAPHITE_PATH
            / "inputs"
            / "GTSRB"
            / "Points"
            / f"{gt_label}.csv"
        )

        tacks_attack = tacks_graphite.GRAPHITEWhiteBoxAttack(
            self.tacks_model, original=True, logger=workspace.logger
        )
        # get transform params from tacks implementation
        tacks_attack.generate_transform_params(
            n_transforms=n_transforms,
            max_dist=max_dist,
            min_angle=min_angle,
            max_angle=max_angle,
            blur_kernel_sizes=blur_kernels,
            nps=nps,
        )

        transform_params = tacks_attack.tparams_list

        # load an image
        img = load_image(
            graphite_bridge.ORIG_GRAPHITE_PATH
            / "inputs"
            / "GTSRB"
            / "images"
            / f"{gt_label}.png"
        )
        img, _ = resize_image(img, (244, 244))
        imgs = opencv_to_torch_img(img).unsqueeze(0)

        mask = load_image(
            graphite_bridge.ORIG_GRAPHITE_PATH
            / "inputs"
            / "GTSRB"
            / "Hulls"
            / f"{gt_label}.png"
        )
        mask, _ = resize_image(mask, (244, 244))
        mask = opencv_to_torch_img(mask)
        mask = mask > 0.5

        # original implementation
        _, transform_wb = graphite_bridge.get_external_functions()
        graphite_imgs = transform_wb(
            imgs.clone(), imgs, mask, transform_params[0], pt_file
        )

        # tacks implementation
        tacks_imgs = tacks_attack._apply_transformation(
            orig=imgs,
            imgs=imgs.clone(),
            mask=mask,
            transform_params=transform_params[0],
            gt_label=gt_label,
        )

        torch.testing.assert_close(tacks_imgs, graphite_imgs)
