# -*- coding: utf-8 -*-
"""Loss functions.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import math

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria =
    FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    """
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t)**self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class YOLOLoss(nn.modules.loss._Loss):
    """YOLO loss.

    The loss function is made of three components:
        ...

    """
    def __init__(self, n_classes, anchors, hyperparameters):
        super().__init__()

        self.hyperparameters = hyperparameters
        self.n_classes = n_classes
        self.anchors = anchors
        self.n_grids = len(self.anchors)

    def forward(self, outputs, gt_labels):

        # convert outputs to float32 if in half precision
        for idg in range(len(outputs)):
            if outputs[idg].dtype is torch.float16:
                outputs[idg] = outputs[idg].float()

        # get devices
        device = outputs[0].device

        # get sizes of the grids
        self.grid_sizes = [
            torch.tensor(outputs[idg].shape)[[2, 3]]
            for idg in range(self.n_grids)
        ]

        # class loss
        lcls = torch.zeros(1, device=device)
        # box class
        lbox = torch.zeros(1, device=device)
        # objectness class
        lobj = torch.zeros(1, device=device)

        tcls, tbox, indices, anchors = self.match_labels_with_grid(gt_labels)
        hyp = self.hyperparameters

        # define criteria
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([hyp['cls_pw']])).to(device)
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([hyp['obj_pw']])).to(device)

        # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        # define focal loss
        fl_gamma = hyp['fl_gamma']
        if fl_gamma > 0:
            BCEcls = FocalLoss(BCEcls, fl_gamma)
            BCEobj = FocalLoss(BCEobj, fl_gamma)

        n_gt_labels = 0  # number of gt_labels
        n_predictions = len(outputs)  # number of outputs

        balance = [4.0, 1.0, 0.4] if n_predictions == 3 else [
            4.0, 1.0, 0.4, 0.1
        ]  # P3-5 or P3-6

        # loop over predictions
        for idi, output in enumerate(
                outputs):  # layer index, layer predictions
            b, a, gj, gi = indices[idi]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(output[..., 0],
                                    device=device)  # target obj

            n = b.shape[0]  # number of gt_labels
            if n:
                # cumulative gt_labels
                n_gt_labels += n
                # prediction subset corresponding to gt_labels
                ps = output[b, a, gj, gi]

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2)**2 * anchors[idi]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                giou = bbox_iou(pbox.T, tbox[idi], x1y1x2y2=False,
                                CIoU=True)  # giou(prediction, target)
                lbox += (1.0 - giou).mean()  # giou loss

                # Objectness
                tobj[b, a, gj,
                     gi] = (1.0 - hyp['gr']) + hyp['gr'] * giou.detach().clamp(
                         0).type(tobj.dtype)

                # Classification
                if self.n_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn,
                                        device=device)  # gt_labels
                    t[range(n), tcls[idi]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE

            lobj += BCEobj(output[..., 4], tobj) * balance[idi]  # obj loss

        # output count scaling
        s = 3 / n_predictions

        lbox *= hyp['giou'] * s
        lobj *= hyp['obj'] * s * (1.4 if n_predictions == 4 else 1.)
        lcls *= hyp['cls'] * s
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls

        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def match_labels_with_grid(self, gt_labels):
        """Matches labels with grid.

        Parameters
        ----------
        gt_labels : torch.Tensor
            Ground truth labels.

        Returns
        -------

        """
        n_anchors = len(self.anchors[0])
        n_gt_labels = gt_labels.shape[0]
        anchor_t = self.hyperparameters['anchor_t']

        # containers
        gt_cls = []
        gt_coords = []
        indices = []
        anch = []
        gain = torch.ones(7, device=gt_labels.device)

        # duplicate ground truth labels for all anchors and append the index of
        # the anchor
        anchors_idx = torch.arange(n_anchors,
                                   device=gt_labels.device).float().view(
                                       n_anchors, 1).repeat(1, n_gt_labels)
        gt_labels = torch.cat(
            (gt_labels.repeat(n_anchors, 1, 1), anchors_idx[:, :, None]), 2)

        # bias
        g = 0.5
        offset = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
                              device=gt_labels.device).float() * 0.5

        # loop over grids
        for idg in range(self.n_grids):

            # get corresponding anchors
            anchors = self.anchors[idg]

            # get size of the grid
            gain[2:6] = self.grid_sizes[idg][[1, 0, 1, 0]]

            # un-normalize coordinates with respect to the size of the grid
            t = gt_labels * gain

            if n_gt_labels > 0:
                # find which box matches with the anchors
                wh_ratio = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(wh_ratio, 1. / wh_ratio).max(2)[0] < anchor_t
                # keep matched labels
                t = t[j]

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + offset[:, None])[j]
            else:
                t = gt_labels[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            gt_coords.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            gt_cls.append(c)  # class

        return gt_cls, gt_coords, indices, anch


def smooth_BCE(
    eps=0.1
):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE gt_labels
    return 1.0 - 0.5 * eps, 0.5 * eps


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw**2 + ch**2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) -
                    (b1_x1 + b1_x2))**2 / 4 + ((b2_y1 + b2_y2) -
                                               (b1_y1 + b1_y2))**2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v + 1e-16)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou
