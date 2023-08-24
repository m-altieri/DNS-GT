# -*- coding: utf-8 -*-
"""Functions to determine the importance of features in a models.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>, <rhamon@protonmail.com>
"""


from eli5.sklearn import PermutationImportance


def compute_permutation_importance(mh, instances, labels, n_iterations=10,
                                   scoring=None):
    """Return the importance of the features using the permutation importance
    method.

    Parameters
    ----------
    mh : ModelHandler
        Model handler.
    instances : array-like
        Instances to consider.
    labels : array-like
        Corresponding labels.
    n_iterations : int
        Number of iterations.
    scoring : str
        Metric to use.
    """
    perm = PermutationImportance(mh.model, cv='prefit', scoring=scoring,
                                 n_iter=n_iterations)
    perm.fit(instances, labels)

    return perm.results_
