# -*- coding: utf-8 -*-
"""YOLO architecture.

References
----------
https://github.com/ultralytics/yolov5

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import math
import sys
import yaml
from pathlib import Path

import torch
import torch.nn as nn

from .layers import ConvBn2d, Focus, Bottleneck, BottleneckCSP, SPP, Detect
from .utils import non_max_suppression, convert_coords_format
from ..layers import Concat
from ..modules import TorchModule

LIST_MODULES = {
    'Upsample': nn.Upsample,
    'ConvBn2d': ConvBn2d,
    'Bottleneck': Bottleneck,
    'SPP': SPP,
    'Focus': Focus,
    'BottleneckCSP': BottleneckCSP,
    'Concat': Concat,
    'Detect': Detect,
}

MSCOCO_ANCHORS = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]


class YOLOv5(TorchModule):
    """YOLOv5 model.

    Format of outputs of the YOLO model have the following specifications:
        * List of Torch.tensor, with as much elements as grid sizes.
        * Each row is a vector of (n_classes + 5) representing a bounding box.
        * The first four columns are the coordinates, at format 'xywh'.
        * Coordinates are relative values.
        * The fifth column is the objectness score, i.e., the confidence score
        associated to the detection.
        * The remaining elements are the distribution of probability over the classes.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    in_channels : int
        Number of channels of the input image (default: 3).
    variant : ['s', 'm', 'l', 'x'], optional
        Variant of the architecture (detault: 's').
    anchors : list of list of ints or None, optional
        List of anchors for the different sizes of the grid. If None, use anchors used
        for the MS COCO dataset (optional: None).
    logger: Logger or None, optional
        Logging system (default: None).
    """

    name = 'YOLOv5'

    def __init__(
        self, n_classes, in_channels=3, variant='s', anchors=None, logger=None
    ):
        super(YOLOv5, self).__init__(in_shape=(in_channels,))

        self.logger = logger

        self.task = 'detection'

        self.n_classes = n_classes
        self.in_channels = in_channels
        self.n_outputs = n_classes + 5

        # compute anchors
        if anchors is None:
            anchors = MSCOCO_ANCHORS
        self.n_grids = len(anchors)
        self.n_anchors = len(anchors[0]) // 2

        # convert anchors to torch tensor and register as buffers
        self.register_buffer(
            'anchors', torch.tensor(anchors).float().view(self.n_grids, -1, 2)
        )
        # assign anchors to each grid
        self.register_buffer(
            'anchor_grid', self.anchors.clone().view(self.n_grids, 1, -1, 1, 1, 2)
        )

        # load config file for the desired variant
        yaml_file = Path(sys.modules[self.__module__].__file__).parent / 'yolov5.yaml'

        if not yaml_file.exists():
            err_msg = 'File {} for variant {} does not exist.'
            raise ValueError(err_msg.format(yaml_file, variant))

        with open(yaml_file) as infile:
            self.cfg = yaml.load(infile, Loader=yaml.FullLoader)

        if variant == 's':
            self.cfg['depth_multiple'] = 0.33
            self.cfg['width_multiple'] = 0.50
        elif variant == 'm':
            self.cfg['depth_multiple'] = 0.67
            self.cfg['width_multiple'] = 0.75
        elif variant == 'l':
            self.cfg['depth_multiple'] = 1.0
            self.cfg['width_multiple'] = 1.0
        elif variant == 'x':
            self.cfg['depth_multiple'] = 1.33
            self.cfg['width_multiple'] = 1.25
        else:
            raise ValueError(f'Unknown variant: {variant}')

        # get layers from config
        self.build_layers_from_config()

        # compute the size factors of the grids by computing the ratio between
        # the size of inputs and outputs
        input_size = 128

        with torch.no_grad():
            outputs = self.forward(torch.zeros(1, in_channels, input_size, input_size))
        self.size_factors = torch.tensor([input_size / x.shape[-2] for x in outputs])

        # compute relative sizes of anchors
        self.anchors /= self.size_factors.view(-1, 1, 1)

        # compute anchor areas
        anchor_areas = self.anchor_grid.prod(-1).view(-1)
        # compute the difference between the first and last anchors areas
        delta_area = anchor_areas[-1] - anchor_areas[0]
        delta_sizes = self.size_factors[-1] - self.size_factors[0]
        # small areas should correspond to small grid sizes. If not, reverse
        # anchors
        if delta_area.sign() != delta_sizes.sign():
            if self.logger:
                self.logger.info('Reversing anchor order.')
            self.anchors[:] = self.anchors.flip(0)
            self.anchor_grid[:] = self.anchor_grid.flip(0)

        self.size_factors = self.size_factors

        # self.layers[-1].initialize_biases(self.n_classes, self.size_factors)
        # self.initialize_weights()

    def forward(self, x):
        saved_features = []
        for module in self.layers:

            # if not from previous layers
            if module.f != -1:
                if isinstance(module.f, int):
                    x = saved_features[module.f]
                else:
                    x = [(x if j == -1 else saved_features[j]) for j in module.f]

            x = module(x)

            # save features if required
            saved_features.append(x if module.i in self.save else None)
        return x

    def build_layers_from_config(self):
        """Builds the layers of the model from a model config."""

        depth_multiple, width_multiple = (
            self.cfg['depth_multiple'],
            self.cfg['width_multiple'],
        )

        # keep track of successive number of channels
        list_in_channels = [self.in_channels]

        layers, save = [], []

        for idm, (from_ids, number, module_name, args) in enumerate(
            self.cfg['backbone'] + self.cfg['head']
        ):

            # calculate depth gain
            number = max(round(number * depth_multiple), 1) if number > 1 else number

            if module_name in [
                'ConvBn2d',
                'Bottleneck',
                'SPP',
                'Focus',
                'BottleneckCSP',
            ]:

                in_channels, out_channels = list_in_channels[from_ids], args[0]

                # scale number of filters, and make it divisible by 8
                out_channels = math.ceil(out_channels * width_multiple / 8) * 8
                dict_args = {
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                }

                if module_name in ['ConvBn2d', 'Focus', 'SPP']:
                    dict_args['kernel_size'] = args[1]
                    if module_name == 'ConvBn2d':
                        dict_args.update(
                            {
                                'kernel_size': args[1],
                                'stride': args[2],
                                'eps': 1e-3,
                                'momentum': 0.03,
                                'activation': 'hardswish',
                            }
                        )
                elif module_name == 'BottleneckCSP':
                    dict_args['n_bottlenecks'] = number
                    if len(args) == 2:
                        dict_args['shortcut'] = args[1]
                    number = 1

            elif module_name == 'Concat':
                out_channels = sum(
                    [list_in_channels[-1 if x == -1 else x + 1] for x in from_ids]
                )
                dict_args = {}
            elif module_name == 'Upsample':
                dict_args = {
                    'size': None if args[0] == 'None' else args[0],
                    'scale_factor': None if args[1] == 'None' else args[1],
                    'mode': args[2],
                }
            elif module_name == 'Detect':
                dict_args = {
                    'n_outputs': self.n_outputs,
                    'n_grids': self.n_grids,
                    'n_anchors': self.n_anchors,
                    'list_in_channels': [list_in_channels[idf + 1] for idf in from_ids],
                }

            module = (
                nn.Sequential(
                    *[LIST_MODULES[module_name](**dict_args) for _ in range(number)]
                )
                if number > 1
                else LIST_MODULES[module_name](**dict_args)
            )

            n_params = sum([x.numel() for x in module.parameters()])
            # attach index, 'from' index, type, number params
            module.i, module.f, module.n_params = (idm, from_ids, n_params)

            # append to savelist
            save.extend(
                item % idm
                for item in ([from_ids] if isinstance(from_ids, int) else from_ids)
                if item != -1
            )
            layers.append(module)
            list_in_channels.append(out_channels)

        self.layers = nn.Sequential(*layers)
        self.save = save

    def post_process(self, outputs, **params):
        """Post-processing of outputs of the model.

        Outputs consists of a prediction tensor for each size factor.

        Parameters
        ----------
        outputs : torch.Tensor
            Outputs of the YOLOv5 model.
        with_nms : bool, optional
            Indicates if NMS is performed or not (default: True).
        obj_thres : float, optional
            Objectness score threshold for NMS (default: 0.5).
        iou_thres : float, optional
            IOU threshold for NMS (default: 0.5).

        Returns
        -------
        torch.Tensor
            Predicted bounding boxes, with or without NMS processing.
        """
        with_nms = params['with_nms'] if 'with_nms' in params else True
        obj_thres = params['obj_thres'] if 'obj_thres' in params else 0.5
        iou_thres = params['iou_thres'] if 'iou_thres' in params else 0.5
        max_detections = params['max_detections'] if 'max_detections' in params else 300

        x = []

        # convert relative coordinates w.r.t. anchors to absolute coordinates
        for ids in range(self.n_grids):

            # pass all values through sigmoid function
            y = outputs[ids].sigmoid()

            # convert to image coordinates
            batch_size, _, n_y, n_x, _ = outputs[ids].shape

            # generate the grid of indices for each grid cell
            yv, xv = torch.meshgrid([torch.arange(n_y), torch.arange(n_x)])
            idx_grid = (
                torch.stack([xv, yv], 2)
                .view((1, 1, n_y, n_x, 2))
                .float()
                .to(outputs[ids].device)
            )

            anchor_grid = self.anchor_grid[ids].to(outputs[ids].device)

            # compute centre of the bounding box
            bbox_xy = (y[..., 0:2] * 2 - 0.5 + idx_grid) * self.size_factors[ids]
            b_wh = (y[..., 2:4] * 2) ** 2 * anchor_grid

            x.append(
                torch.cat([bbox_xy, b_wh, y[..., 4::]], -1).view(
                    batch_size, -1, self.n_outputs
                )
            )

        preds = torch.cat(x, 1)

        if not with_nms:
            return preds

        # NMS
        with torch.no_grad():

            pp_preds = preds.clone()
            nms_preds = []

            # rescale the confidence of labels with objectness score
            pp_preds[..., 5:] = pp_preds[..., 5:] * pp_preds[..., 4:5]

            # loop over samples of the batch
            for idp in range(pp_preds.shape[0]):

                # convert the boxes from (x_c, y_c, width, height) to (x1, y1, x2, y2)
                pp_preds[idp, :, 0:4] = convert_coords_format(
                    pp_preds[idp, :, 0:4], conversion='xywh2xyxy'
                )

                # run NMS algorithm
                nms_preds.append(
                    non_max_suppression(
                        pp_preds[idp, ...],
                        obj_thres=obj_thres,
                        iou_thres=iou_thres,
                        max_detections=max_detections,
                    ).to(preds.device)
                )

            return preds, nms_preds
