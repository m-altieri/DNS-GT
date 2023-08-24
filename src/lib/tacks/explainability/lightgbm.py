# -*- coding: utf-8 -*-
"""Explainability of LightGBM models.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>, <rhamon@protonmail.com>
"""
import re

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tacks.utils import sigmoid


def browse_tree(tree, features, lookup_feature_id=-1):
    """Browse a tree and return the score achieved for a given set of features.

    if `lookup_feature_id` is provided, the two subtrees when this feature is
    used as splitting value are browsed.

    Parameters
    ----------
    tree :  dict
        Decision tree to browse.
    features : list
        Feature vector.
    lookup_feature_id : int
        Index of the feature to discriminate.

    Returns
    -------
    scalar or (scalar, (scalar, scalar))
        If `lookup_feature_id` is used as splitting value, return the score
        achieved for the two subtrees, and the value of the threshold.
        Otherwise, a unique score is returned.
    """
    if 'split_feature' in tree:
        split_feature_id = tree['split_feature']
        if split_feature_id == lookup_feature_id:
            return (tree['threshold'],
                    (browse_tree(tree['right_child'], features),
                     browse_tree(tree['left_child'], features)))

        else:
            next_tree = (tree['left_child'] if features[split_feature_id] <=
                         tree['threshold'] else tree['right_child'])
            return browse_tree(next_tree, features, lookup_feature_id)

    else:
        return tree['leaf_value']


def inspect_feature(trees, feature_id, features, vmin=-np.inf):
    """Compute the best value for a feature to decrease the total
    score.

    Parameters
    ----------
    trees : list of dict
        List of trees.
    feature_id : int
        Feature to inspect.
    features : list
        Feature vector.
    vmin : scalar, optional
        Minimal value for the feature.

    Returns
    -------
    array-like
        List of thresholds

    """
    thresholds = []
    true_contribs = []
    false_contribs = []

    # get contribution for all trees
    base = 0
    for idt, tree in enumerate(trees):

        contribution = browse_tree(
            tree['tree_structure'], features, feature_id)

        if isinstance(contribution, tuple):
            threshold, (true_contrib, false_contrib) = contribution
            thresholds.append(threshold)
            true_contribs.append(true_contrib)
            false_contribs.append(false_contrib)
        else:
            base += contribution

    thresholds = np.round(np.array(thresholds), 7)
    true_contribs = np.array(true_contribs)
    false_contribs = np.array(false_contribs)

    unique_thresholds = np.concatenate(
        [[vmin], np.sort(np.unique(thresholds))])

    contribution_by_threshold = np.array(
        [np.sum(true_contribs[thresholds <= threshold]) +
         np.sum(false_contribs[thresholds > threshold])
         for threshold in unique_thresholds]) + base

    return unique_thresholds, contribution_by_threshold


def get_bounds(feat_values, scores, min_value=-np.inf):
    """Get lower and upper bounds on the value of a feature in order to achieve
    the best score.

    Parameters
    ----------
    feat_values : array-like
        Values of the feature.
    scores : array-like
        Raw scores for each value of the features.
    min_value :  scalar
        Minimal value required for the feature.

    Returns
    -------
    tuple of scalar
        Lower and upper bounds for the value of the feature.
    scalar
        Best score achieved.
    """
    rectified_scores = scores.copy()
    rectified_scores[feat_values < min_value] = np.inf

    if np.all(feat_values < min_value):
        best_idx = scores.shape[0] - 1
    else:
        best_idx = np.argmin(rectified_scores)
    lb_value = feat_values[best_idx]

    ub_value = (feat_values[best_idx + 1] if (best_idx + 1) < len(feat_values)
                else np.inf)

    return (lb_value, ub_value), scores[best_idx]


def get_value_from_bounds(bounds, alpha=0.99, beta=1.01, gamma=1, as_int=True):
    """Get the best value to consider from bounds.

    Parameters
    ----------
    bounds : tuple of scalar
        Bounds to the value.
    alpha : scalar, optional
        Coefficient to apply when bounds are infinite.
    as_int : bool, optional
        Round the value and return an integer.

    Returns
    -------
    scalar
    """
    if bounds[0] == -np.inf:
        number = bounds[1] * alpha
    elif bounds[1] == np.inf:
        number = bounds[0] * beta
    else:
        number = (gamma * bounds[0] + bounds[1]) / (gamma + 1)

    return int(np.round(number)) if as_int else number


def plot_profile_feature(feat_values, scores, feature_label=None,
                         with_probability=False, feat_value=None,
                         raw_score=None, detection_threshold=0.8336):
    """Plot the profile of the feature.


    Parameters
    ----------
    feat_values : array-like
        Values of the feature.
    scores : array-like
        Raw scores for each value of the features.
    feature_label : str, optional
        Label of the feature.
    with_probability : bool, optional
        Indicates if raw scores or probabilities are displayed.
    feat_value : scalar, optional
        Current value of the feature.
    current_score : scalar, optional
        Current raw score.
    detection_threshold : scalar, optional
        Threshold for detection.
    """

    if feat_values[0] == -np.inf:
        if len(feat_values) == 1:
            feat_values = np.array([-100, 100])
            scores = np.repeat(scores, 2)
        else:
            feat_values[0] = feat_values[1] - np.abs(feat_values[1])

    if with_probability:
        scores = sigmoid(scores)

    plt.figure()
    plt.step(feat_values, scores, where='post')

    if raw_score:
        plt.plot(feat_value, raw_score, 'or')

    if feature_label is not None:
        plt.title('Profile feature "{}"'.format(feature_label))
    plt.xlabel('Value of the feature')
    plt.ylabel('Score')

    if with_probability:
        plt.axhspan(0., detection_threshold, facecolor='green', alpha=0.25,
                    label='Benign')
        plt.axhspan(detection_threshold, 1, facecolor='red',
                    alpha=0.25, label='Malware')
        plt.ylim(0, 1)
        plt.legend(fancybox=True, fontsize=14)


def compute_gains(list_feature_ids, list_min_values, trees, features,
                  raw_score):
    """Compute the gains of a list of features.

    Parameters
    ----------
    list_feature_ids : list of int
        List of feature ids.
    list_min_values : list of scalar
        List of minimal values for the corresponding feature id.
    trees : list of dict
        List of trees.
    features : list
        Feature vector.
    raw_score : int
        Current raw score for the given features.

    Returns
    -------
    array-like
        Gains for each feature.
    """
    gains = np.zeros(len(list_feature_ids))
    for idf, feature_id in enumerate(list_feature_ids):
        feat_values, scores = inspect_feature(
            trees, feature_id, features, vmin=0)
        _, best_score = get_bounds(
            feat_values, scores, min_value=list_min_values[idf])
        gains[idf] = raw_score - best_score

    return gains


def plot_importance_features(importances, feat_labels, n_max_features=50,
                             fontsize=14):
    """Plot the importances of features in decreasing order.

    Parameters
    ----------
    importances : array-like
        Importances of each feature.
    feat_labels : list of str
        Labels of features.
    n_max_features : int, optional
        Maximal number of features to plot.
    fontsize : int, optional
        Font size of labels.
    """
    sns.set()

    # get number of features to plot
    n_max_features = min(n_max_features, len(importances))

    # sort the feature in decreasing order of importance
    sorted_idx = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_idx]
    sorted_featlabels = feat_labels[sorted_idx]

    # plot the figure
    plt.barh(np.arange(n_max_features, 0, -1),
             sorted_importances[0:n_max_features],
             tick_label=sorted_featlabels[0:n_max_features])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize, rotation=45)
    plt.xlabel('Feature importance', fontsize=fontsize)


def create_graphviz_from_lightbgm_model(classifier, nodeid_max=100):
    """Create a graphviz instance from a lightgbm model.

    The graph is clipped to keep only a limited number of branches.

    Parameters
    ----------
    classifier : LightGBMClassifier
        Classifier with a LightGBM model.
    nodeid_max : int, optional

    """
    # create the graph
    graph = lgb.create_tree_digraph(classifier.model, precision=2)

    # copy the body of the graph
    body = graph.body.copy()

    # maximal ID number
    id_max = 100
    leaves = []
    for content in body:
        node = re.findall(r'\tsplit(\d+)', content)
        edge = re.findall(r'\tsplit(\d+) -> split(\d+)', content)
        edge_leaf = re.findall(r'\tsplit(\d+) -> leaf(\d+)', content)

        if edge:
            u, v = edge[0]
            if int(u) > id_max or int(v) > id_max:
                graph.body.remove(content)
        elif edge_leaf:
            u, v = edge_leaf[0]
            if int(u) > id_max:
                graph.body.remove(content)
            else:
                print(edge_leaf)
                leaves.append(int(v))
        elif node:
            if int(node[0]) > id_max:
                graph.body.remove(content)

    for content in body:
        leaf = re.findall(r'\tleaf(\d+)', content)
        if leaf and int(leaf[0]) not in leaves:
            graph.body.remove(content)

    return graph
