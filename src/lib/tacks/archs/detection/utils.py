# -*- coding: utf-8 -*-
"""Util functions.


Format of images
================

Images can have two formats:
    * 'numpy':
        * type: np.ndarray
        * dtype: np.uint8
        * format: HWC.
        * values between 0 and 255.
    * 'torch':
        * type: torch.Tensor
        * dtype: torch.float16 or torch.float32
        * format: CHW.
        * values between 0 and 1.

Format of labels
================

Labels have the following format:
    * Each row is a vector of 5 (ground truth) or 6 (predicted) values representing a
    bounding box.
    * The first four columns are the coordinates, at format 'xyxy', in absolute values.
    * The fifth column is the objectness score. It it -1 in the case of ground truth.
    * The sixth column is the class id.


Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import random

import cv2
import numpy as np
import torch
import torchvision


def convert_coords_format(bboxes, conversion='xywh2xyxy'):
    """Convert coordinates of bounding boxes from a format to another one.

    Two formats are considered:
        * 'xywh': [x, y, w, h] where (x, y) are the center of the bounding box;
        * 'xyxy': [x1, y1, x2, y2] where (x1, y1) and (x2, y2) are respectively the
        top-left and bottom-right of the bounding box.

    Parameters
    ----------
    bboxes : torch.Tensor
        Coordinates of the boxes.
    conversion : ['xywh2xyxy', 'xyxy2xywh'], optional
        Type of conversion (default: 'xywh2xyxy').
    """

    if bboxes.shape[1] != 4:
        err_msg = 'Boxes should have 4 coordinates.'
        raise ValueError(err_msg)

    new_bboxes = torch.zeros_like(bboxes)

    if conversion == 'xywh2xyxy':

        # top left coordinate
        new_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
        new_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2

        # bottom right coordinate
        new_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
        new_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2

    elif conversion == 'xyxy2xywh':

        # centre coordinate
        new_bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
        new_bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2

        # width
        new_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        # height
        new_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    else:
        raise ValueError('Unknown conversion: {}'.format(conversion))

    return new_bboxes


def convert_label_format(label, conversion='gt2pred'):
    """Convert the format of labels.

    Two formats are considered:
        * 'gt'
            * 1-4: coordinates at the 'xywh' format.
            * 5: class id
        * 'pred'
            * 1-4: coordinates at the 'xyxy' format.
            * 5: class id
            * 6: confidence score

    Parameters
    ----------
    label : torch.Tensor
        Label to convert.
    conversion : ['gt2pred', 'pred2gt'], optional
        Type of conversion (default: 'gt2pred').

    Returns
    -------
    torch.Tensor
    """
    if conversion == 'gt2pred':

        new_label = torch.zeros((label.shape[0], 6), device=label.device)
        # convert coordinates of the boxes into 'xyxy' format
        new_label[:, 0:4] = convert_coords_format(label[:, 0:4], conversion='xywh2xyxy')
        # add classes
        new_label[:, 4] = label[:, 4]
        # add ground truth confidence score
        new_label[:, 5] = 1.0

    elif conversion == 'pred2gt':

        new_label = torch.zeros((label.shape[0], 5), device=label.device)
        # add classes
        new_label[:, 0] = label[:, 5]
        # convert coordinates of the boxes into 'xywh' format
        new_label[:, 1:5] = convert_coords_format(label[:, 0:4], conversion='xyxy2xywh')

    else:
        raise ValueError('Unknown conversion: {}'.format(conversion))

    return new_label


def draw_bounding_boxes_on_image(
    img, label=None, class_names=None, class_colors=None, draw_label=True
):
    """Draw bounding boxes on an image.

    Ground truth labels can be passed by using an objectness below 0.

    Parameters
    ----------
    img : torch.Tensor or nd-array
        Image to plot, in torch or numpy format.
    label : torch.Tensor or None
        Label associated to the image, containing bounding boxes.
    class_names : list of str, or None, optional
        Names of the classes. If None, draw ids instead (default: None).
    class_colors : list of 3-tuple, or None, optional
        Colors of the box for each class. If None, colors are randomly drawn
        (default: None).
    draw_label : bool, optional
        Indicates if labels are drawn on top of bounding boxes or not (default:
        True).

    Returns
    -------
    nd-array
        Image ready to be plotted with :mod:`matplotlib` or :mod:`cv2`.
    """
    if class_names is None:
        class_names = [str(item) for item in range(100)]

    if class_colors is None:
        class_colors = [
            [random.randint(0, 255) for _ in range(3)] for _ in range(len(class_names))
        ]

    if isinstance(img, torch.Tensor):
        img = img.cpu().float().numpy().transpose(1, 2, 0) * 255
        img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)

    if label is not None:
        height, width, _ = img.shape

        n_boxes = label.shape[0]

        # drawing options
        thickness = max(1, round(0.001 * (height + width) / 2))
        font_size = max(thickness - 2, 1)

        # get coordinates of bounding boxes
        coords = label[:, :4].numpy()

        # get objectness score of bounding boxes
        objectness = label[:, 4].numpy()

        # get class ids of bounding boxes
        class_ids = label[:, 5].int().numpy()

        # define box images
        gtbox_img = np.zeros_like(img)

        for idb in range(n_boxes):

            class_id = class_ids[idb]
            color = class_colors[class_id]
            label = class_names[class_id]
            is_gt = objectness[idb] < 0.0

            # get coordinates (x, y) of top-left and bottom-right corners of the rectangle
            pt1 = (int(coords[idb, 0]), int(coords[idb, 1]))
            pt2 = (int(coords[idb, 2]), int(coords[idb, 3]))

            if not is_gt:
                # add confidence score if predicted box
                label += ' {:.1f}'.format(objectness[idb])

            cv2.rectangle(
                gtbox_img if is_gt else img,
                pt1,
                pt2,
                color=color,
                thickness=cv2.FILLED if is_gt else thickness,
                lineType=cv2.LINE_AA,
            )

            # draw the label
            if draw_label and not is_gt:
                text_size = cv2.getTextSize(
                    label, 0, fontScale=font_size / 2, thickness=thickness
                )[0]

                pt3 = pt1[0] + text_size[0], pt1[1] - text_size[1] - 3

                cv2.rectangle(
                    img,
                    pt1,
                    pt3,
                    color=color,
                    thickness=cv2.FILLED,
                    lineType=cv2.LINE_AA,
                )

                cv2.putText(
                    img,
                    label,
                    (pt1[0], pt1[1] - 2),
                    fontFace=0,
                    fontScale=font_size / 2,
                    color=[255, 255, 255],
                    thickness=font_size,
                    lineType=cv2.LINE_AA,
                )

        img = cv2.addWeighted(img, 1.0, gtbox_img, 0.6, 0)

    return img


def non_max_suppression(
    preds, obj_thres=0.1, iou_thres=0.6, class_agnostic=True, max_detections=300
):
    """Performs Non-Maximum Suppression (NMS) on predicted outputs obtained after
    inference of the YOLO model.

    Predictions specifications:
        * Each row is a vector of (n_classes + 5) representing a bounding box
        * The first four columns are the coordinates, at format 'xyxy'.
        * Coordinates are relative values.
        * The fifth column is the objectness score, i.e., the confidence score
        associated to the detection.
        * The remaining elements are the distribution of probability over the
        classes.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted bounding boxes for one sample.
    obj_thres : float, optional
        Threshold for the confidence scores.
    iou_thres : float, optional
        Threshold for the IOU measure.
    class_agnostic : bool, optional
        Indicates if the class is taken into account or not. If False, two overlapping
        with different classes will not be merged (default: True).
    max_detections : int, optional
        Maximum number  of detections (default: 300).

    Returns
    -------
    torch.Tensor
        NMS predictions.
    """
    max_img_size = 4096

    # get bounding boxes with sufficient level of confidence
    preds = preds[preds[..., 4] > obj_thres, :]

    # return empty tensor if no more boxes
    if preds.shape[0] == 0:
        return torch.zeros((0, 6))

    i, j = torch.nonzero(preds[:, 5:] > obj_thres, as_tuple=True)
    preds = torch.cat((preds[i, 0:4], preds[i, j + 5, None], j[:, None].float()), 1)

    # return empty tensor if no more boxes
    if preds.shape[0] == 0:
        return torch.zeros((0, 6))

    # add an offset to avoid two overlapping bounding boxes to be merged if not
    # agnostic
    cls_offset = preds[:, 5:6] * (0 if class_agnostic else max_img_size)

    bboxes, scores = preds[:, :4] + cls_offset, preds[:, 4]

    # run NMS
    nms_bboxes_idx = torchvision.ops.boxes.nms(bboxes, scores, iou_thres)

    if nms_bboxes_idx.shape[0] > max_detections:
        nms_bboxes_idx = nms_bboxes_idx[:max_detections]

    return preds[nms_bboxes_idx, :]


def scale_coords(coords, src_img_size, trg_img_size):
    """Scales coordinates from an image size to another image size.

    Coordinates should have 'xyxy' format.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates of bounding boxes.
    src_img_size : tuple of ints
        Size of the image in which coordinates apply.
    trg_img_size : tuple of ints
        Size of the image in which coordinates will be scale.

    Returns
    -------
    torch.Tensor
        New coordinates.
    """
    coords = coords.clone()

    # get the minimal ratio between heights and widths
    ratio = min(src_img_size[0] / trg_img_size[0], src_img_size[1] / trg_img_size[1])
    # padding to apply
    pad = (src_img_size[1] - trg_img_size[1] * ratio) / 2, (
        src_img_size[0] - trg_img_size[0] * ratio
    ) / 2

    # apply padding on height
    coords[:, [0, 2]] -= pad[0]
    # apply padding on width
    coords[:, [1, 3]] -= pad[1]
    # apply ratio
    coords[:, :4] /= ratio

    # clip coordinates with respect to the size of the image
    coords = torchvision.ops.boxes.clip_boxes_to_image(coords, trg_img_size)

    return coords


def write_summary(detections, class_names):
    """Returns a textual summary of the number of detections made for each class.

    Parameters
    ----------
    detections : torch.Tensor
        Tensor of detections.
    class_names : list of str.
        Names of classes.
    """
    n_detections = detections.shape[0]
    if n_detections > 0:

        count_per_class = [
            (
                int(class_id),
                class_names[int(class_id)],
                (detections[:, -1] == class_id).sum(),
            )
            for class_id in detections[:, -1].unique()
        ]

        summary = ' / '.join(
            [
                f'{count} {class_name} ({class_id})'
                for class_id, class_name, count in count_per_class
            ]
        )
    else:
        summary = 'nothing'

    return summary
