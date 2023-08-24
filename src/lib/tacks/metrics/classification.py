# -*- coding: utf-8 -*-
"""Classification metrics.

Metrics
-------


Let 
For a class c \in \{1, \ldots, C\}:

    TPR = \frac{\sum_{1}^N 1_{}{}
TPR: True positive rate
    Number of positive samples that are correctly predicted.
TNR: True negative rate
    Number of negative samples that are correctly predicted.
FNR: False negative rate
    Number of negative samples that are correctly predicted.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import numpy as np
import torch


class ClassificationMeter:
    """Meter to store relevant classification metrics.

    A Meter stores scalar items and allows for various operations on them.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    """

    def __init__(self, n_classes):

        self.n_classes = n_classes

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.n_samples = None

        self.reset()

    def reset(self):
        """Reset the list of items."""
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.tn = [0] * self.n_classes
        self.fn = [0] * self.n_classes
        self.n_samples = 0

    def __call__(self, outputs, gt_labels, k=1, with_softmax=True):
        """Compute the number of true positive (TP), false positive (FP), true negative
        (TN) and false negative (FN) for each class from outputs of a classification
        model, looking at the first k top values.

        Parameters
        ----------
        outputs : torch.Tensor
            Output tensor.
        gt_labels : torch.Tensor
            Target tensor.
        k : int
            Top values to consider (default: 1).
        with_softmax : bool, optional
            Indicates if a softmax operation should be performed (default: True).
        """

        if torch.all(outputs == 0):
            raise ValueError(outputs)
        self.n_samples += outputs.shape[0]
        if with_softmax:
            outputs = torch.softmax(outputs, 1)
        _, pred_labels = outputs.topk(k, 1, True, True)

        gt_labels = gt_labels.unsqueeze(1)

        for idc in range(self.n_classes):

            self.tp[idc] += (
                torch.logical_and(gt_labels == idc, pred_labels == idc)[:, :k]
                .any(1)
                .sum()
                .item()
            )
            self.fp[idc] += (
                torch.logical_and(gt_labels != idc, pred_labels == idc)[:, :k]
                .all(1)
                .sum()
                .item()
            )
            self.tn[idc] += (
                torch.logical_and(gt_labels != idc, pred_labels != idc)[:, :k]
                .any(1)
                .sum()
                .item()
            )

            self.fn[idc] += (
                torch.logical_and(gt_labels == idc, pred_labels != idc)[:, :k]
                .all(1)
                .sum()
                .item()
            )

    def get_accuracy(self, average_method='micro'):
        """Get accuracy.

        Parameters
        ----------
        average_method : {'micro', 'macro', 'weighted'} or int or None
            Indicates how metrics are averaged over classes (default: None).

        Return
        ------
        float or list of floats.
        """
        if average_method == 'micro':
            return sum(self.tp) / self.n_samples
        else:
            acc_by_class = [
                (self.tp[idc] + self.tn[idc])
                / (self.tp[idc] + self.tn[idc] + self.fp[idc] + self.fn[idc])
                for idc in range(self.n_classes)
            ]

            if average_method == 'macro':
                return float(np.mean(acc_by_class))
            else:
                return acc_by_class

    def get_precision(self, average_method='micro'):
        """Get precision.

        Parameters
        ----------
        average_method : {'micro', 'macro', 'weighted'} or int or None
            Indicates how metrics are averaged over classes (default: None).

        Return
        ------
        float or list of floats.
        """
        if average_method == 'micro':
            sum_tp_fp = sum(self.tp) + sum(self.fp)
            return sum(self.tp) / sum_tp_fp if sum_tp_fp > 0 else np.nan

        else:
            prec_by_class = [
                self.tp[idc] / (self.tp[idc] + self.fp[idc])
                if (self.tp[idc] + self.fp[idc]) > 0
                else np.nan
                for idc in range(self.n_classes)
            ]

            if average_method == 'macro':
                return float(np.nanmean(prec_by_class))
            else:
                return prec_by_class

    def get_recall(self, average_method='micro'):
        """Get recall.

        Parameters
        ----------
        average_method : {'micro', 'macro', 'weighted'} or int or None
            Indicates how metrics are averaged over classes (default: None).

        Return
        ------
        float or list of floats.
        """

        if average_method == 'micro':
            sum_tp_fn = sum(self.tp) + sum(self.fn)
            return sum(self.tp) / sum_tp_fn if sum_tp_fn > 0 else np.nan

        else:
            rec_by_class = [
                self.tp[idc] / (self.tp[idc] + self.fn[idc])
                if (self.tp[idc] + self.fn[idc]) > 0
                else np.nan
                for idc in range(self.n_classes)
            ]
            if average_method == 'macro':
                return float(np.nanmean(rec_by_class))
            else:
                return rec_by_class

    def __str__(self):
        string = ''
        string += f'Accuracy: {self.get_accuracy()}\n'
        string += f'Precision: {self.get_precision()}\n'
        string += f'Recall: {self.get_recall()}\n'

        return string

    def __getitem__(self, index):

        index = index.split('_')

        metric_name = index[0]
        if len(index) == 2:
            if index[1] in ['micro', 'macro']:
                average_method = index[1]
                id_class = None
            else:
                id_class = int(index[1])
        else:
            average_method = 'micro'
            id_class = None

        if metric_name == 'accuracy':
            if id_class is None:
                return self.get_accuracy(average_method=average_method)
            else:
                return self.get_accuracy(average_method=None)[id_class]

        elif metric_name == 'precision':
            if id_class is None:
                return self.get_precision(average_method=average_method)
            else:
                return self.get_precision(average_method=None)[id_class]

        elif metric_name == 'recall':
            if id_class is None:
                return self.get_recall(average_method=average_method)
            else:
                return self.get_recall(average_method=None)[id_class]


def get_correct_among_ktops(outputs, gt_labels, list_k=(1,), with_softmax=True):
    """Return for each sample if the correct label is among the first k predicted labels
    for different values of k.

    Computation can be limited to a specific ground truth label.

    Parameters
    ----------
    outputs : torch.Tensor
        Output tensor.
    gt_labels : torch.Tensor
        Target tensor.
    list_k : list
        List of k (default: (1, )).
    specific_gt_class : int or None
        Specific ground truth class for which it is computed. If None, compute for all
        classes.
    with_softmax : bool, optional
        Indicates if a softmax operation should be performed (default: True).

    Returns
    -------
    list of int or list of list of int
        Indicates if the correct label is among the first k predicted samples, for all
        batches and different values of k. If only k has a single value, the returned
        value is a list of int, otherwise it returns a list of list of int.
    """

    if with_softmax:
        outputs = torch.softmax(outputs, 1)
    _, pred_labels = outputs.topk(max(list_k), 1, True, True)

    correct = pred_labels.eq(gt_labels.unsqueeze(1))
    values = [correct[:, :k].any(1).int().tolist() for k in list_k]

    return values[0] if len(list_k) == 1 else values


if __name__ == "__main__":

    from sklearn.metrics import precision_score, recall_score
    from sklearn.metrics import accuracy_score

    # data
    n_classes = 3
    n_samples = 1000

    gt_labels = torch.randint(n_classes + 1, (n_samples,))
    gt_labels[gt_labels == n_classes] = 0
    outputs = torch.rand((n_samples, n_classes))
    outputs[:, -1] *= 5
    outputs[gt_labels == 0, 0] *= 2

    _, pred_labels = outputs.max(1)

    ####################################################################################
    # TACkS

    print('@ manual')
    print('  Acc: {:.3f}'.format((gt_labels == pred_labels).float().mean().item()))

    print()
    for idc in range(n_classes):
        print(
            '  Acc_{}: {:.3f}'.format(
                idc,
                torch.logical_or(
                    torch.logical_and(gt_labels == idc, pred_labels == idc),
                    torch.logical_and(gt_labels != idc, pred_labels != idc),
                )
                .float()
                .mean()
                .item(),
            )
        )

        print()

    print()

    ####################################################################################
    # SKLEARN

    print('@ sklearn')
    print('  Acc: {:.3f}'.format(accuracy_score(gt_labels, pred_labels)))

    print()
    print(
        '  Prec (micro): {:.3f}'.format(
            precision_score(gt_labels, pred_labels, average='micro')
        )
    )
    print(
        '  Rec (micro): {:.3f}'.format(
            recall_score(gt_labels, pred_labels, average='micro')
        )
    )
    print()
    print(
        '  Prec (macro): {:.3f}'.format(
            precision_score(gt_labels, pred_labels, average='macro')
        )
    )
    print(
        '  Rec (macro): {:.3f}'.format(
            recall_score(gt_labels, pred_labels, average='macro')
        )
    )
    print()
    print(
        '  Prec (all): {}'.format(
            [
                f'{item:.3f}'
                for item in precision_score(gt_labels, pred_labels, average=None)
            ]
        )
    )
    print(
        '  Rec (all): {}'.format(
            [
                f'{item:.3f}'
                for item in recall_score(gt_labels, pred_labels, average=None)
            ]
        )
    )

    print()
    for idc in range(n_classes):
        print(
            '  Prec_{}: {:.3f}'.format(
                idc,
                precision_score(gt_labels, pred_labels, labels=[1], average=None)[0],
            )
        )
        print(
            '  Rec_{}: {:.3f}'.format(
                idc,
                recall_score(gt_labels, pred_labels, labels=[1], average=None)[0],
            )
        )

    ####################################################################################
    # TACkS

    classif_meter = ClassificationMeter(n_classes)

    classif_meter(outputs[0:250, :], gt_labels[0:250])
    classif_meter(outputs[500:, :], gt_labels[500:])
    classif_meter(outputs[250:500, :], gt_labels[250:500])

    print()
    print('@ tacks')
    print(
        '  Acc (micro): {:.3f}'.format(
            classif_meter.get_accuracy(average_method='micro')
        )
    )
    print(
        '  Prec (micro): {:.3f}'.format(
            classif_meter.get_precision(average_method='micro')
        )
    )
    print(
        '  Rec (micro): {:.3f}'.format(classif_meter.get_recall(average_method='micro'))
    )

    print()
    print(
        '  Acc (macro): {:.3f}'.format(
            classif_meter.get_accuracy(average_method='macro')
        )
    )
    print(
        '  Prec (macro): {:.3f}'.format(
            classif_meter.get_precision(average_method='macro')
        )
    )
    print(
        '  Rec (macro): {:.3f}'.format(classif_meter.get_recall(average_method='macro'))
    )

    print()
    print(
        '  Acc (all): {}'.format(
            [f'{item:.3f}' for item in classif_meter.get_accuracy(average_method=None)]
        )
    )
    print(
        '  Prec (all): {}'.format(
            [f'{item:.3f}' for item in classif_meter.get_precision(average_method=None)]
        )
    )
    print(
        '  Rec (all): {}'.format(
            [f'{item:.3f}' for item in classif_meter.get_recall(average_method=None)]
        )
    )

    print()
    print(
        '  Acc: {}'.format(
            [f'{item:.3f}' for item in classif_meter.get_accuracy(average_method=None)]
        )
    )
    print()
    for idc in range(n_classes):
        print(
            '  Acc_{}: {:.3f}'.format(
                idc,
                classif_meter.get_accuracy(average_method=None)[idc],
            )
        )
        print(
            '  Prec_{}: {:.3f}'.format(
                idc,
                classif_meter.get_precision(average_method=None)[idc],
            )
        )
        print(
            '  Rec_{}: {:.3f}'.format(
                idc,
                classif_meter.get_recall(average_method=None)[idc],
            )
        )
        print()
