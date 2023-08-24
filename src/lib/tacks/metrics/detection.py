"""Detection metrics.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""

import torch
import torchvision


def compute_ap(valid_detections_per_iou, n_gt_boxes):
    """Computes the area under the Precision x Recall curve.

    Parameters
    ----------
    valid_detections_per_iou : torch.tensor
        For each predicted box and value of IOU, indicates if a detection is valid or
        not.

    Returns
    -------
    torch.tensor
        Area under the curve, for different values of IOU.
    """

    n_iou_values = valid_detections_per_iou.shape[1]

    # accumulate FPs and TPs
    fp_cumul = (~valid_detections_per_iou).int().cumsum(0)
    tp_cumul = valid_detections_per_iou.int().cumsum(0)

    # compute recall and precision curves
    recall = tp_cumul.float() / n_gt_boxes
    precision = tp_cumul.float() / (tp_cumul + fp_cumul).float()

    # append boundary values to calculate the area under the curve
    m_rec = torch.cat([torch.zeros((1, n_iou_values)), recall, recall[-2:-1, :]])
    m_prec = torch.cat([precision[0:1, :], precision, torch.zeros((1, n_iou_values))])

    # get the envelope of the recall precision curve
    max_prec, _ = m_prec.flip(0).cummax(0)
    max_prec = max_prec.flip(0)

    ap_cls = torch.zeros(n_iou_values)
    for idi in range(n_iou_values):
        # get the points where the recall changes
        mrec_changes_idx = torch.where(m_rec[1:, idi] != m_rec[:-1, idi])[0]

        # compute the area under the curve
        ap_cls[idi] = (
            (m_rec[mrec_changes_idx + 1, idi] - m_rec[mrec_changes_idx, idi])
            * max_prec[mrec_changes_idx + 1, idi]
        ).sum()

    return recall, precision, ap_cls


def compute_ap_per_class(valid_detections_per_iou, pred_classes, gt_classes):
    """Computes the average precision per class.

    References
    ----------
    Rafael Padilla, S. L. Netto, and E. A. B. da Silva ‘Survey on performance
    metrics for object-detection algorithms’, presented at the International
    Conference on Systems, Signals and Image Processing (IWSSIP), 2020.

    Parameters
    ----------
    valid_detections_per_iou : torch.tensor
        For each predicted box and value of IOU, indicates if a detection is valid or
        not.
    pred_classes : torch.Tensor
        Predicted classes for each prediction.
    gt_classes : torch.Tensor
        Ground truth classes.

    Returns
    -------
    torch.tensor
        Average precision for each class and each value of IOU.
    torch.tensor
        Recall for each value of IOU.
    torch.tensor
        Precision for each value of IOU.
    """

    n_iou_values = valid_detections_per_iou.shape[1]

    # find unique classes
    unique_classes = torch.unique(torch.cat([pred_classes, gt_classes]).int())
    n_unique_classes = len(unique_classes)

    # recall per class
    recall_cls = torch.zeros((n_unique_classes, n_iou_values))
    # precision per class
    precision_cls = torch.zeros((n_unique_classes, n_iou_values))
    # average precision per class
    ap_cls = torch.zeros((n_unique_classes, n_iou_values))
    # number of detection per class
    nboxes_gt_cls = torch.zeros(n_unique_classes, dtype=torch.int)
    nboxes_pred_cls = torch.zeros(n_unique_classes, dtype=torch.int)

    # loop over classes
    for idc, cls in enumerate(unique_classes):

        # get prediction indices
        pred_idx = pred_classes == cls

        # count number of predicted objects
        n_pred_boxes = pred_idx.sum()

        # count number of ground truth objects with given class
        n_gt_boxes = (gt_classes == cls).sum()

        # store number of boxes for the class
        nboxes_gt_cls[idc] = n_gt_boxes
        nboxes_pred_cls[idc] = n_pred_boxes

        if n_pred_boxes > 0 and n_gt_boxes > 0:

            (
                recall_per_iou,
                precision_per_iou,
                ap_cls_per_iou,
            ) = compute_ap(valid_detections_per_iou[pred_idx, :], n_gt_boxes)

            recall_cls[idc, :] = recall_per_iou[-1, :]
            precision_cls[idc, :] = precision_per_iou[-1, :]
            ap_cls[idc, :] = ap_cls_per_iou

    return (
        unique_classes,
        nboxes_gt_cls,
        nboxes_pred_cls,
        recall_cls,
        precision_cls,
        ap_cls,
    )


def get_valid_detections_per_iou(
    pred_label, gt_label, iou_values=None, class_agnostic=False
):
    """Returns valid detections from predicted label given ground truth label, for
    different values of IOU.

    Parameters
    ----------
    pred_label : torch.tensor
        Predicted label.
    gt_label : torch.tensor
        Ground truth label.
    iou_values : torch.tensor, optional
        IOU values to consider, as a sorted tensor. If None, consider 10 values from 0.5
        to 0.95 (default: None).
    class_agnostic : bool, optional
        Indicates whether the class is used as criterion of validity or not (default:
        False).

    Returns
    -------
    torch.tensor
    """

    if iou_values is None:
        iou_values = torch.linspace(0.5, 0.95, 10, device=pred_label.device)

    n_pred_boxes = pred_label.shape[0]
    n_gt_boxes = gt_label.shape[0]
    n_iou_values = iou_values.shape[0]

    # array indicating if a prediction is valid for all IOU values
    valid_detections_per_iou = torch.zeros(
        (n_pred_boxes, n_iou_values), dtype=torch.bool, device=pred_label.device
    )

    # if there is no prediction, return empty tensors
    if n_pred_boxes == 0 or n_gt_boxes == 0:
        return valid_detections_per_iou

    # compute IOU between predicted and ground truth boxes
    ious = torchvision.ops.box_iou(pred_label[:, 0:4], gt_label[:, 0:4])

    # for each gt box, sort IOUS in descending order
    sorted_ious, sorted_ious_idx = ious.sort(0, descending=True)

    for idg in range(n_gt_boxes):
        for idp in range(n_pred_boxes):

            # check if the detection is not already marked as valid
            if not valid_detections_per_iou[sorted_ious_idx[idp, idg], 0]:

                # if the IOU value is higher than the minimal IOU value
                if sorted_ious[idp, idg] >= iou_values[0]:

                    # if the class is the same or it is class_agnostic mode
                    if class_agnostic or (
                        pred_label[sorted_ious_idx[idp, idg], 5] == gt_label[idg, 5]
                    ):

                        valid_detections_per_iou[
                            sorted_ious_idx[idp, idg],
                            iou_values <= sorted_ious[idp, idg],
                        ] = True

                        # skip other pred boxes and go to the next gt box
                        break

    return valid_detections_per_iou


if __name__ == "__main__":

    # Example from https://github.com/rafaelpadilla/Object-Detection-Metrics

    gt_labels = torch.Tensor(
        [
            [0, 25, 16, 38, 56, -1, 0],
            [0, 129, 123, 41, 62, -1, 0],
            [1, 123, 11, 43, 55, -1, 0],
            [1, 38, 132, 59, 45, -1, 0],
            [2, 16, 14, 35, 48, -1, 0],
            [2, 123, 30, 49, 44, -1, 0],
            [2, 99, 139, 47, 47, -1, 0],
            [3, 53, 42, 40, 52, -1, 0],
            [3, 154, 43, 31, 34, -1, 0],
            [4, 59, 31, 44, 51, -1, 0],
            [4, 48, 128, 34, 52, -1, 0],
            [5, 36, 89, 52, 76, -1, 0],
            [5, 62, 58, 44, 67, -1, 0],
            [6, 28, 31, 55, 63, -1, 0],
            [6, 58, 67, 50, 58, -1, 0],
        ]
    )

    pred_labels = torch.Tensor(
        [
            [0, 5, 67, 31, 48, 0.88, 0],
            [0, 119, 111, 40, 67, 0.70, 0],
            [0, 124, 9, 49, 67, 0.80, 0],
            [1, 64, 111, 64, 58, 0.71, 0],
            [1, 26, 140, 60, 47, 0.54, 0],
            [1, 19, 18, 43, 35, 0.74, 0],
            [2, 109, 15, 77, 39, 0.18, 0],
            [2, 86, 63, 46, 45, 0.67, 0],
            [2, 160, 62, 36, 53, 0.38, 0],
            [2, 105, 131, 47, 47, 0.91, 0],
            [2, 18, 148, 40, 44, 0.44, 0],
            [3, 83, 28, 28, 26, 0.35, 0],
            [3, 28, 68, 42, 67, 0.78, 0],
            [3, 87, 89, 25, 39, 0.45, 0],
            [3, 10, 155, 60, 26, 0.14, 0],
            [4, 50, 38, 28, 46, 0.62, 0],
            [4, 95, 11, 53, 28, 0.44, 0],
            [4, 29, 131, 72, 29, 0.96, 0],
            [4, 29, 163, 72, 29, 0.23, 0],
            [5, 43, 48, 74, 38, 0.45, 0],
            [5, 17, 155, 29, 35, 0.84, 0],
            [5, 95, 110, 25, 42, 0.43, 0],
            [6, 16, 20, 101, 88, 0.48, 0],
            [6, 33, 116, 37, 49, 0.95, 0],
        ]
    )

    gt_labels[:, 3] += gt_labels[:, 1]
    gt_labels[:, 4] += gt_labels[:, 2]

    pred_labels[:, 3] += pred_labels[:, 1]
    pred_labels[:, 4] += pred_labels[:, 2]

    device = 'cpu'

    iou_values = torch.Tensor([0.29])
    n_iou_values = iou_values.shape[0]

    valid_detections_per_iou = torch.empty([0, n_iou_values], dtype=torch.bool)
    confidences = torch.empty(0)
    pred_classes = torch.empty(0)
    gt_classes = torch.empty(0)

    # get the valid detections for each sample
    for ids in range(7):

        gt_label = gt_labels[gt_labels[:, 0] == ids, :][:, 1::]
        pred_label = pred_labels[pred_labels[:, 0] == ids, :][:, 1::]

        valid_detections_per_iou = torch.cat(
            [
                valid_detections_per_iou,
                get_valid_detections_per_iou(pred_label, gt_label, iou_values),
            ]
        )
        confidences = torch.cat([confidences, pred_label[:, 4]])
        pred_classes = torch.cat([pred_classes, pred_label[:, 5]])
        gt_classes = torch.cat([gt_classes, gt_label[:, 5]])

    # sort by confidence
    _, sort_idx = confidences.sort(descending=True)
    valid_detections_per_iou = valid_detections_per_iou[sort_idx, :]
    confidences = confidences[sort_idx]
    pred_classes = pred_classes[sort_idx]

    # compute the metrics per class
    (
        unique_classes,
        nboxes_gt_cls,
        nboxes_pred_cls,
        recall_cls,
        precision_cls,
        ap_cls,
    ) = compute_ap_per_class(valid_detections_per_iou, pred_classes, gt_classes)

    # compute the metrics
    recall, precision, ap = compute_ap(valid_detections_per_iou, gt_labels.shape[0])

    # prints
    print('Computation by class: ')

    print(
        '# GT boxes: {:d} (expected: {:d})'.format(nboxes_gt_cls[0], gt_labels.shape[0])
    )
    print(
        '# Pred boxes: {:d} (expected: {:d})'.format(
            nboxes_pred_cls[0], pred_labels.shape[0]
        )
    )
    print('# Precision: {:.3f} (expected: {:.3f})'.format(precision_cls[0, 0], 0.2916))
    print('# Recall: {:.3f} (expected: {:.3f})'.format(recall_cls[0, 0], 0.4666))
    print('# AP: {:.3f} (expected: {:.3f})'.format(ap_cls[0, 0], 0.2456))
