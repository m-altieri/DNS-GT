# -*- coding: utf-8 -*-
"""Utils classes and functions for analysis of experiment results.

.. moduleauthor:: Ronan Hamon
"""
import configparser
import importlib
import logging
import os
import stat
import sys
from pathlib import Path

from .utils import get_product_from_dict


def process_xresults(xresults, avg_axes=None, kept_axes=None):
    """Return xresults as list of arrays.

    This function returns a list of arrays for each possible combination of values for
    all axes. All axes are considered, except:
        * axes on which data are averaged;
        * axes kept in the final arrays.

    If data are averaged, the output is a 2-tuple containing mean and std values.

    Parameters
    ----------
    xresults : xarray
        Array to process.
    avg_axes : list of str or None
        List of axes on which to average on.
    kept_axes : list of str or None
        List of axes to keep in the arrays.

    Returns
    -------
    list of dict
        Values for discarded axes as dict.
    list of tuple of xarray or list of xarray
        Corresponding xarrays.
    """

    if avg_axes is None:
        avg_axes = []

    if kept_axes is None:
        kept_axes = []

    indexes_list = get_product_from_dict(
        {
            axis_name: axis_values.values.tolist()
            for axis_name, axis_values in xresults.indexes.items()
            if axis_name not in set(avg_axes).union(kept_axes)
        }
    )
    if len(indexes_list) == 1 and indexes_list[0] == {}:
        indexes_list = []

    # if there are averaging, return mean and std
    if len(avg_axes) > 0:
        avg_xresults = (xresults.mean(avg_axes), xresults.std(avg_axes))
    else:
        avg_xresults = xresults

    if len(indexes_list) > 0:
        out_xresults = (
            indexes_list,
            [
                [out_array.sel(index) for out_array in avg_xresults]
                if isinstance(avg_xresults, tuple)
                else avg_xresults.sel(index)
                for index in indexes_list
            ],
        )
    else:
        out_xresults = indexes_list, avg_xresults

    return out_xresults


if __name__ == "__main__":

    import random

    import matplotlib.pyplot as plt
    import numpy as np
    from tacks.utils import Workspace
    from tacks.yafe import Experiment

    def get_data(a, b):
        return {}

    def get_problem(c, d, e):
        def generate_problem(**kwargs):
            return {}, {}

        return generate_problem

    def get_solver(f):
        def generate_solver(**kwargs):
            return {}

        return generate_solver

    def get_measure(
        solution_data, solved_data, task_params, source_data, problem_data, **kwargs
    ):

        a = task_params['data_params']['a']
        e = task_params['problem_params']['e']

        return {'m1': a * random.random(), 'm2': e + random.random()}

    workspace = Workspace('Testing', instance_name='yafe.results', args={})

    # generate an experiment
    yafe_exp = Experiment(
        name='mock_xp',
        workspace=workspace,
        get_data=get_data,
        get_problem=get_problem,
        get_solver=get_solver,
        get_measure=get_measure,
        save_intermediate=False,
        reset=False,
    )

    # add tasks
    data_params = {'a': [0, 1, 2], 'b': ['alpha', 'beta']}
    problem_params = {'c': [1], 'd': [0.25, 0.11], 'e': range(10)}
    solver_params = {'f': [4, 2]}

    yafe_exp.add_tasks(
        data_params=data_params,
        problem_params=problem_params,
        solver_params=solver_params,
    )

    yafe_exp.generate_tasks()
    yafe_exp.launch_experiment()
    yafe_exp.collect_results()

    xresults_full = yafe_exp.load_results('xarray')

    avg_axes = ('data_a', 'problem_c')
    kept_axes = ('measure', 'problem_e')

    indexes_list, proc_xresults = process_xresults(xresults_full, avg_axes, kept_axes)

    # plotting
    x_name = 'problem_e'
    xticks = xresults_full.coords[x_name].to_index().to_list()
    x = np.arange(len(xticks))

    plt.figure()

    for idi, index in enumerate(indexes_list):

        plt.subplot(2, 4, idi + 1)
        plt.title(' / '.join([f'{name}: {value}' for name, value in index.items()]))
        plt.xlabel(x_name)
        plt.xticks(xticks)
        plt.ylabel('measure')

        for measure_name in ['m1', 'm2']:
            y = proc_xresults[idi][0].sel(measure=measure_name)
            yerr = proc_xresults[idi][1].sel(measure=measure_name)
            plt.errorbar(x, y, yerr=yerr, label=measure_name)

        plt.legend()
        plt.grid()

    plt.show()
