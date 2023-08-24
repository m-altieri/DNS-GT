# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2018
# -----------------
#
# * Laboratoire d'Informatique et Systèmes <http://www.lis-lab.fr/>
# * Université d'Aix-Marseille <http://www.univ-amu.fr/>
# * Centre National de la Recherche Scientifique <http://www.cnrs.fr/>
# * Université de Toulon <http://www.univ-tln.fr/>
#
# Contributors
# ------------
#
# * Ronan Hamon <firstname.lastname_AT_lis-lab.fr>
# * Valentin Emiya <firstname.lastname_AT_lis-lab.fr>
# * Florent Jaillet <firstname.lastname_AT_lis-lab.fr>
#
# Description
# -----------
#
# yafe: Yet Another Framework for Experiments.
#
# Licence
# -------
# This file is part of yafe.
#
# yafe is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
# ######### COPYRIGHT #########
"""Base functions and classes for experiments.

.. moduleauthor:: Ronan Hamon
.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Florent Jaillet

Designing an :class:`~yafe.base.Experiment`
===========================================

Designing an experiment consists in

* Creating an :class:`~yafe.base.Experiment` object which includes, as arguments of the
constructor:

    * a ``get_data`` function to load data,
    * a ``get_problem`` class or function to generate problems from the data,
    * a ``get_solver`` class or function to solve problems,
    * a ``measure`` function to compute performance.

* Adding tasks to the :class:`~yafe.base.Experiment` object by specifying the data,
problem and solver parameters.

The API for designing an experiment is described below and is illustrated in the
:ref:`yafe tutorial </_notebooks/yafe.ipynb>`.

Loading data: ``get_data`` and data parameters
----------------------------------------------
Experiments are often applied to several data items. Items may be entries loaded from a
dataset or data synthesized from some control parameters.

The input of function ``get_data`` is one or several parameters that characterize which
data item must be returned.

The output of function ``get_data`` is a dictionary with arbitrary keys and whom values
are the returned data.

Note that in the current version, ``get_data`` should have at least one parameter (which
may take a single value).

Examples:

* For a dataset of ``n`` items, one may design a ``get_data(i_item)`` function that
takes an integer ``i_item``, load item number ``i_item`` in a variable ``x`` and returns
a dictionary ``{'data_item': x}`` ; one may then add tasks using ``data_params =
{'i_item': np.arange(n)}``.

* For synthetic data, one may design a ``get_data(f0)`` function that takes an
real-valued frequency ``f0``, synthesize a sinusoid with frequency ``f0`` in a variable
``x`` and returns a dictionary ``{'signal': x}`` ; one may then add tasks using
``data_params = {'f0': np.linspace(0, 1, 10)}``.

Generating problems: ``get_problem`` and problem parameters
-----------------------------------------------------------
Problems to be solved are generated from the data. Several instances of a problem may be
generated from one data item, by varying some problem parameters, e.g., amount of noise.
Thus, designing the problem generation stage consists in specifying how to turn problems
parameters and data into problem data (and solution). This may be done equivalently by a
class or a function.

``get_problem`` may be a class, in which case:

* the inputs of the ``__init__`` method are the parameters of the problem,
* the parameters of the ``__call__`` method must match the keys of the
  dictionary obtained from ``get_data``,
* the output of the ``__call__`` method is a tuple
  ``(problem_data, solution_data)`` where ``problem_data`` is a dictionary
  containing the problem data for the solver and ``solution_data`` is a
  dictionary containing the solution of the problem for the performance
  measure.

Thus, an instance of that class is generated for each set of problem parameters and is
called on data to generate the problem data and its solution.

Alternatively, ``get_problem`` may be a function, in which case:

* its inputs are the parameters of the problem,
* its output must be a function that takes some data as inputs and returns the tuple
``(problem_data, solution_data)`` as described in the case above.

Note that in the current version, ``get_problem`` should have at least one parameter
(which may take a single value).

Solving problems: ``get_solver`` and solver parameters
------------------------------------------------------
A solver generates a solution from some problem data. Several instances of a solver may
be used by varying the solver's parameters, e.g., the order of the model. As for
problems, solvers can be implemented equivalently by a class or a function.

``get_solver`` may be a class, in which case:

* the inputs of the ``__init__`` method are the parameters of the solver,
* the parameters of the ``__call__`` method must match the keys of the dictionary
obtained from the problem generation method ``__call__``,
* the output of the ``__call__`` method is a dictionary containing the solution
generated by the solver.

Alternatively, ``get_solver`` may be a function, in which case:

* its inputs are the parameters of the solver,
* its output must be a function that takes some problem data as inputs and returns a
dictionary containing the solution generated by the solver, similarly as the
``__call__`` method described above.

Note that in the current version, ``get_solver`` should have at least one parameter
(which may take a single value).

Computing performance: ``measure``
----------------------------------
Several performance measures may be calculated from the estimated solution, using other
data like the original data, the problem data, or parameters of the data, problem or
solver.

These performance measures must be computed within a *function* ``measure`` as follows:

* its arguments should be dictionaries ``source_data``, ``problem_data``,
``solution_data`` and ``solved_data``, as returned by the data access function, the
``__call__`` methods of problem generation class and the ``__call__`` methods of solver
class, respectively; an additional argument ``task_params`` is a dictionary that
contains the data, problem and solver parameters;
* its output is a dictionary whose keys and values are the names and values of the
various performance measures.

Setting data, problem and solver parameters: adding tasks
---------------------------------------------------------

Once an instance of :class:`~yafe.base.Experiment` is created based on the four
processing blocks given by functions or classes  ``get_data``, ``get_problem``,
``get_solver`` and ``measure``, one must define the parameters that must be used for
each of these blocks. This is done by calling method
:meth:`~yafe.base.Experiment.add_tasks`: its inputs are three dictionaries
``data_params``, ``problem_params`` and ``solver_params`` whose keys match the input of
``get_data``, ``get_problem`` and ``get_solver`` respectively, and whose values are
array-like objects containing the parameters that must be used in the instances of those
processing blocks. A hypercube is automatically generated, based on the cartesian
product of all the arrays of parameters (internally stored in a so-called *schema*).
Each point in this hypercube is a full set of parameters to define a so-called *task* by
generating some particular instances of data, problem and solver.

Running an :class:`~yafe.base.Experiment`
=========================================

Once an :class:`~yafe.base.Experiment` is defined and tasks have been added, one can run
the tasks by calling method :meth:`~yafe.base.Experiment.launch_experiment`, which
executes all the tasks, except those that may have already been executed previously. At
any time, one may control the execution of the tasks by calling method
:meth:`~yafe.base.Experiment.display_status`.

Running an :class:`~yafe.base.Experiment` on a cluster
------------------------------------------------------
Tasks may also be executed in parallel, individually or in batches, by specifying the
parameter ``task_ids`` in :meth:`~yafe.base.Experiment.launch_experiment`. Function
:func:`~yafe.utils.generate_oar_script` may be used or taken as an example to split the
set of tasks among jobs that can be run by several cores in parallel.

Adding tasks
------------
Method :meth:`~yafe.base.Experiment.add_tasks` may be used to add new parameter values,
in which case the parameter hypercube is extended, recomputing all cartesian products
between parameters, and new tasks are created while results from already-executed tasks
are kept.

Collecting results
------------------
Method :meth:`~yafe.base.Experiment.collect_results` gathers the task results into a
single structure and saves it into a single file. Method
:meth:`~yafe.base.Experiment.load_results` then loads and returns the results so that
the user can process and display them. The main structure that contains the results is a
multi-dimensional array which dimensions match the parameters of the tasks (see above)
and the performance measures (last dimension). This array may be loaded as an
:class:`numpy.ndarray` or as a :class:`xarray.DataArray`.

Misc
====

Tasks IDs and parameters
------------------------
Task IDs are generated from the parameters and depend on the sequence of calls to method
:meth:`~yafe.base.Experiment.add_tasks`. Finding a task ID from a set of parameters and
vice-versa is not trivial and may be done by calling methods
:meth:`~yafe.base.Experiment.get_task_data_by_id` and
:meth:`~yafe.base.Experiment.get_task_data_by_params`.

Dealing with several solvers
----------------------------
In order to compare several solvers, one may design one experiment by solver, using the
same generators and parameters for data and problems, and merge the results.

Random seeds (reproducibility, consistent problem generation)
-------------------------------------------------------------
It is often important to control the random seeds, e.g., in order to reproduce the
experiments. Since all the tasks are run independently, controlling the random seed is
also required in order to have the exact same problem data and solution in all tasks
with the same data parameters and problem parameters, especially when noise is added.

This can be obtained by using the *random state* mechanism provided by
:class:`numpy.random.RandomState`. The random state seed may be used inside each block
(see ``get_problem`` in :ref:`yafe tutorial </_notebooks/yafe.ipynb>`) or passed as a
parameter to each processing block (as in ``scikit-learn``).

Files and folders
-----------------
A number of files and subfloders are generated in a folder named as the experiment's
name and located in the ``data_path`` set when creating the experiment. The user may not
need to handle those files since results are then collected automatically, unless in
case of tracking errors.

At the experiment level, a so-called *schema* is generated and contains all the data,
problem and solver parameters (file ``_schema.pickle``). Results from all tasks are
gathered in a single file ``results.npz``.

At the task level, a sub-directory is created for each task. It contains several files
generated when the task is created (parameters of the task in ``task_params.pickle``),
when the task is run (raw data in ``source_data.pickle``, problem data in
``problem_data.pickle``, true solution in ``solution_data.pickle``, solved data in
``solved_data.pickle``, performance measures in ``result.pickle``) and when an error
occurs during any processing step when running a task (``error_log``).

.. _yafe_configuration_file:

:mod:`yafe` configuration file
------------------------------

The :mod:`yafe` package uses a configuration file stored in ``$HOME/.config/yafe.conf``
to specify the default ``data_path`` used to store the experiment data, as well as the
default ``logger_path`` used to store the processing log.

You can create an empty configuration file by running the static method
:meth:`~yafe.utils.ConfigParser.generate_config` and then edit the generated file in
``$HOME/.config/yafe.conf``.

The resulting configuration file should have the following form::

    [USER]
    # user directory (data generated by experiments)
    data_path = /some/directory/path

    [LOGGER]
    # path to the log file
    path = /some/file/path/yafe.log

If you want to avoid the use of the :mod:`yafe` configuration file, you need to
systematically specify the ``data_path`` and ``logger_path`` parameters when creating an
:class:`~yafe.base.Experiment` (or alternatively for ``logger_path`` you can set the
flag ``log_to_file`` to ``False``).

"""
import inspect
import logging
import os
import shutil
import sys
import time
import traceback

import dill
import numpy as np
import torch
import torch.multiprocessing as mp

from ..utils import Workspace
from .utils import get_product_from_dict


def DEFAULT_GET_DATA():
    return {}


def DEFAULT_GET_PROBLEM():
    def generate_problem(**kwargs):
        return {}, {}

    return generate_problem


def DEFAULT_GET_SOLVER():
    def solve_problem(**kwargs):
        return {}

    return solve_problem


def DEFAULT_GET_MEASURE(
    solution_data, solved_data, task_params, source_data, problem_data, **kwargs
):
    return {}


class InvalidTask(Exception):
    """This exception is used to signal that the task should not be run."""

    pass


class Experiment:
    """Definition of a class to design experiments.

    Parameters
    ----------
    name : str
        Name of the experiment, used to store experiment data in an eponym subfolder.
    get_data : function
        A function that returns data from some data parameters.
    get_problem : callable class or function
        A class or function that takes some problem parameters and returns a callable
        object that takes input data from method ``get_data`` and returns some problem
        data and true solution data.
    get_solver : callable object or function
        A class or function that takes some solver parameters and returns a callable
        object that takes problem data from method ``get_problem`` and returns some
        solution data.
    get_measure : function
        Function that takes some task information and data returned by methods
        ``get_data``, ``get_problem``  and ``get_solver`` in order to compute one or
        several performance measures.
    reset : ['all', 'only_failed', 'no'] or bool, optional
        Flag forcing resetting the Experiment during initialization. If 'all', the
        previously computed data stored on disk for the Experiment with the same name
        are deleted. If 'only_failed', only computed data of failed data are deleted. If
        'no', the previously computed data are reused. If reset is bool, True means
        'all', False mean 'no' (default: False).
    data_path : None or str or pathlib.Path
        Path of the folder where the experiment data subfolder must be created.  If
        ``data_path`` is ``None``, the data path given in the yafe configuration file is
        used.
    save_intermediate : dict or None
        Dictionary keyed by ['data', 'problem', 'solver'] indicating if the intermediate
        data at each stage are saved or not. If None, all data are saved (defailt:
        None).
    logger: Logger or None, optional
        Logging system (default: None).
    log_to_file : bool
        Indicates if the processing and debug information must be logged in a file.
    log_to_console : bool
        Indicates if the processing information must be displayed in the console.
    logger_path : None or str or pathlib.Path
        Full file path where the logged data must be written when ``log_to_file ==
        True``.  If ``logger_path`` is ``None`` the logger path given in the yafe
        configuration file is used.
    """

    def __init__(
        self,
        name,
        workspace,
        get_data=None,
        get_problem=None,
        get_solver=None,
        get_measure=None,
        reset=False,
        save_intermediate=None,
    ):

        # check of parameters
        if isinstance(reset, bool):
            reset = 'all' if reset else 'no'

        if reset not in ['all', 'only_failed', 'no']:
            err_msg = 'Unknown reset option: {:s}'
            raise ValueError(err_msg.format(reset))

        if get_data is None:
            get_data = DEFAULT_GET_DATA

        if get_problem is None:
            get_problem = DEFAULT_GET_PROBLEM

        if get_solver is None:
            get_solver = DEFAULT_GET_SOLVER

        if get_measure is None:
            get_measure = DEFAULT_GET_MEASURE

        if save_intermediate is None:
            save_intermediate = {'data': True, 'problem': True, 'solver': True}

        # check that measure function has required parameters
        meas_params = inspect.signature(get_measure).parameters
        meas_expected_params = (
            'task_params',
            'source_data',
            'problem_data',
            'solution_data',
            'solved_data',
        )

        for param in meas_expected_params:
            if param not in meas_params:
                err_msg = 'Wrong signature for measure. Missing parameter: {}.'
                raise TypeError(err_msg.format(param))

        # store parameters as attributes
        self.name = str(name)

        self.workspace = workspace
        self.workspace_name = workspace.name
        self.xp_path = workspace.xps_path / name

        self.save_intermediate = save_intermediate

        self.get_data = get_data
        self.get_problem = get_problem
        self.get_solver = get_solver
        self.get_measure = get_measure

        # data handling
        if reset in ['only_failed', 'no']:

            self.logger.info('Load experiment schema...')

            # check that existing parameters are consistent with current schema
            if (self.xp_path / '_schema.pickle').exists():

                self.logger.info('Checking experiment schema...')
                # Check that the existing schema is coherent with the signature of the
                # provided functions

                schema_keys = [
                    ('data', par) for par in inspect.signature(self.get_data).parameters
                ]
                schema_keys += [
                    ('problem', par)
                    for par in inspect.signature(self.get_problem).parameters
                ]
                schema_keys += [
                    ('solver', par)
                    for par in inspect.signature(self.get_solver).parameters
                ]

                if set(self._schema.keys()) != set(schema_keys):
                    err_msg = 'Provided schema inconsistent with the schema of the stored in {}.'
                    raise RuntimeError(err_msg.format(str(self.xp_path)))

            else:
                self.logger.info(
                    'No _schema.pickle file found, resetting experiment %s.', self.name
                )
                reset = 'all'

        if reset == 'only_failed':

            self.logger.info('Reset failed tasks of experiment %s.', self.name)
            for task_id in self.get_pending_task_ids():
                self._delete_item(task_id, type_item='source_data')
                self._delete_item(task_id, type_item='problem_data')
                self._delete_item(task_id, type_item='solved_data')
                self._delete_item(task_id, type_item='error_log')

        if reset == 'all':

            self.logger.info('Reset all tasks of experiment %s.', self.name)

            # remove the xp folder and create directories and schema
            if self.xp_path.exists():
                shutil.rmtree(str(self.xp_path))

        if not self.xp_path.exists():
            self.logger.info('Creating directories and schema...')
            self._create_dirs_and_schema()

        self.logger.info('Initialization done.')

    # @property
    # def xp_path(self):
    #     return self.workspace.xps_path / self.name

    @property
    def logger(self):
        return self.workspace.logger

    def _create_dirs_and_schema(self):
        self.xp_path.mkdir()
        (self.xp_path / 'tasks').mkdir()
        schema = {}
        for data_param in inspect.signature(self.get_data).parameters:
            schema[('data', data_param)] = []
        for problem_param in inspect.signature(self.get_problem).parameters:
            schema[('problem', problem_param)] = []
        for solver_param in inspect.signature(self.get_solver).parameters:
            schema[('solver', solver_param)] = []
        self._schema = schema

    @property
    def _schema(self):
        """Schema of the parameters for each section of the experiment.

        Parameters
        ----------
        _schema : dict
            Schema of parameters of the experiment.

        Notes
        -----
        The content is written in a pickle file. Use I/O with caution.
        """
        schema_path = self.xp_path / '_schema.pickle'

        if not schema_path.exists():
            raise FileNotFoundError('Missing _schema file for xp {}'.format(self.name))
        else:
            with open(str(schema_path), 'rb') as fin:
                schema = dill.load(file=fin)

        return schema

    @_schema.setter
    def _schema(self, schema):
        schema_path = self.xp_path / '_schema.pickle'
        with open(str(schema_path), 'wb') as fout:
            dill.dump(schema, file=fout)

    # Task management (methods for internal use, should not be called by user)
    def _has_item(self, type_item, task_id):
        """Indicates if an item of a given task exists or not.

        Parameters
        ----------
        type_item : str
            Type of requested item ({'task_params', 'source_data',
            'problem_data', 'solved_data', 'result'})
        task_id : int
            Id of the item.

        Returns
        -------
        bool
            Return True if the item for the given task exists, otherwise
            False.
        """
        filename = type_item + '.pickle'
        path_item = self.xp_path / 'tasks' / '{:06}'.format(task_id) / filename
        return path_item.exists()

    def _read_item(self, type_item, task_id):
        """Load an item from the id.

        Parameters
        ----------
        type_item : str
            Type of requested item.
        task_id : int
            Id of the item.

        Returns
        -------
        dict or None
        """
        path_task = self.xp_path / 'tasks' / '{:06}'.format(task_id)
        if self._has_item(type_item, task_id):
            filename = type_item + '.pickle'
            with open(str(path_task / filename), 'rb') as fin:
                return dill.load(file=fin)
        else:
            return None

    def _write_item(self, type_item, task_id, content):
        """Write an item with the id.

        Parameters
        ----------
        type_item : str
            Type of requested item.
        id : int
            Id of the item.
        content : str
            Content to write. Should be JSON serializable.
        """
        task_path = self.xp_path / 'tasks' / '{:06}'.format(task_id)
        if not task_path.exists():
            task_path.mkdir()

        filename = type_item + '.pickle'
        with open(str(task_path / filename), 'wb') as fout:
            dill.dump(content, file=fout)

    def _delete_item(self, task_id, type_item=''):
        """Delete an item from the id of a task.

        Parameters
        ----------
        task_id : int
            Id of the item.
        type_item : str
            Type of requested item. If empty, delete all items of the task.
        """
        path_task = self.xp_path / 'tasks' / '{:06}'.format(task_id)
        if type_item == '':
            if path_task.exists():
                shutil.rmtree(str(path_task))
        else:
            if self._has_item(type_item, task_id):
                filename = type_item + '.pickle'
                os.remove(str(path_task / filename))

    def _get_all_items(self, type_item, only_ids=False):
        """Get all items of the experiment.

        Parameters
        ----------
        type_item : str
            Type of requested item.
        only_ids : bool, optional
            Indicates if only ids of items are retrieved.

        Returns
        -------
        [dict, list]
            List of ids or dict keyed by the ids.
        """
        # create an empty container
        items = [] if only_ids else {}

        # get each item and insert it
        for task_path in (self.xp_path / 'tasks').glob('*'):
            filename = type_item + '.pickle'
            if (task_path / filename).exists():
                task_id = int(task_path.stem)
                if only_ids:
                    items.append(task_id)
                else:
                    items[task_id] = self._read_item(type_item, task_id)

        return items

    # Experiment management

    def _handle_exception_in_run_task(self, exc, task_id, task_params, logger=None):
        if logger is not None:
            logger.exception(exc)
        exc_traceback = traceback.format_exception(*sys.exc_info())
        error_log = 'Task {}\n\nTask parameters:\n{}\n\n{}'.format(
            task_id, str(task_params), ''.join(exc_traceback)
        )
        self._write_item('error_log', task_id, error_log)

    def add_tasks(self, data_params=None, problem_params=None, solver_params=None):
        """Add tasks using combinations of parameters.

        Parameters
        ----------
        data_params : dict or None
            Parameters of generation of the dataset.
        problem_params : dict or None
            Parameters of generation of the problem.
        solver_params : dict or None
            Parameters of generation of the solver.

        Raises
        ------
        TypeError
            If parameters are not dict.
        ValueError
            If parameters contain unexpected keys.

        Notes
        -----
        * All parameters should be dict, keyed by the name of the parameter and
          valued by a list.
        * Tasks are generated as the cartesian product of the sets of
          parameters. If parameters are added by successive calls to
          :meth:`~yafe.base.Experiment.add_tasks`, the resulting set of tasks
          is the cartesian product of all updated sets of parameters (which is
          equivalent to the set of tasks obtained by calling
          :meth:`~yafe.base.Experiment.add_tasks` only once with all the
          parameters).
        """
        schema = self._schema

        def sort_key(a):
            return -1 if a is None else a

        if data_params is not None:

            # check that params are dict
            if not isinstance(data_params, dict):
                raise TypeError('data_params should be a dict')

            # convert non-iterable elements in list
            for key, value in data_params.items():
                if not hasattr(value, '__iter__') or isinstance(value, str):
                    data_params[key] = [value]
            for key, value in data_params.items():
                label = ('data', key)
                if label not in schema:
                    errmsg = '{} is not a parameter for getting data.'
                    raise ValueError(errmsg.format(key))
                else:
                    schema[label] = sorted(
                        list(set(schema[label]).union(set(value))), key=sort_key
                    )

        if problem_params is not None:

            # check that params are dict
            if not isinstance(problem_params, dict):
                raise TypeError('problem_params should be a dict')

            # convert non-iterable elements in list
            for key, value in problem_params.items():
                if not hasattr(value, '__iter__') or isinstance(value, str):
                    problem_params[key] = [value]

            for key, value in problem_params.items():
                label = ('problem', key)
                if label not in schema:
                    errmsg = '{} is not a parameter for getting the problem.'
                    raise ValueError(errmsg.format(key))
                else:
                    schema[label] = sorted(
                        list(set(schema[label]).union(set(value))), key=sort_key
                    )

        if solver_params is not None:

            # check that params are dict
            if not isinstance(solver_params, dict):
                raise TypeError('solver_params should be a dict')

            # convert non-iterable elements in list
            for key, value in solver_params.items():
                if not hasattr(value, '__iter__') or isinstance(value, str):
                    solver_params[key] = [value]

            for key, value in solver_params.items():
                label = ('solver', key)
                if label not in schema:
                    errmsg = '{} is not a parameter for getting the solver.'
                    raise ValueError(errmsg.format(key))
                else:
                    schema[label] = sorted(
                        list(set(schema[label]).union(set(value))), key=sort_key
                    )

        self._schema = schema

    def generate_tasks(self):
        """Generate the tasks from _schema.

        Already existing tasks are not rewritten.
        """

        # get _schema
        schema = self._schema

        # get the existing tasks
        existing_task_params = [
            dict(item) for item in self._get_all_items('task_params').values()
        ]

        n_tasks_added = 0
        n_existing_tasks = len(existing_task_params)

        data_params = dict()
        problem_params = dict()
        solver_params = dict()

        for label in schema.keys():
            cat, key = label

            if len(schema[label]) == 0:
                err_msg = (
                    "Cannot generate tasks. Missing values for '{0[0]:s}_{0[1]:s}'."
                )
                raise ValueError(err_msg.format(label))

            if cat == 'data':
                data_params[key] = schema[label]
            elif cat == 'problem':
                problem_params[key] = schema[label]
            else:  # cat == 'solver':
                solver_params[key] = schema[label]

        # get cartesian products of parameters
        parameters = get_product_from_dict(
            {
                'data_params': get_product_from_dict(data_params),
                'problem_params': get_product_from_dict(problem_params),
                'solver_params': get_product_from_dict(solver_params),
            }
        )

        # for each set of parameter, write the corresponding task
        for params in [item for item in parameters if item not in existing_task_params]:
            self._write_item('task_params', n_existing_tasks + n_tasks_added, params)
            n_tasks_added += 1

        self.logger.info(
            '%d task%s added. Total number of tasks: %d',
            n_tasks_added,
            's' if n_tasks_added > 0 else '',
            n_existing_tasks + n_tasks_added,
        )

    def run_task_by_id(self, task_id, extra_params=None):
        """Run a task given an id.

        Parameters
        ----------
        task_id : int
            Id of the task.
        """

        # create an instance for the task in the workspace
        workspace = Workspace(
            self.workspace_name, instance_name=f'{self.name}_task{task_id}'
        )

        workspace.logger.info('Loading task %s...', task_id)
        task_params = self._read_item('task_params', task_id)

        if extra_params is None:
            extra_params = {}

        extra_params.update(
            {'xp_name': self.name, 'task_id': task_id, 'workspace': workspace}
        )

        if task_params is None:
            err_msg = 'Unknown task: {}'.format(task_id)
            workspace.logger.error(err_msg)
            raise ValueError(err_msg)

        workspace.logger.info('    Data: %s', str(task_params['data_params']))
        workspace.logger.info('    Problem: %s', str(task_params['problem_params']))
        workspace.logger.info('    Solver: %s', str(task_params['solver_params']))

        if self._has_item('error_log', task_id):
            workspace.logger.info(
                'Task %s, remove error log from previous run', task_id
            )
            self._delete_item(task_id, 'error_log')

        try:
            if self._has_item('source_data', task_id):
                workspace.logger.info('Task %s, source data already generated', task_id)
                source_data = self._read_item('source_data', task_id)
            else:
                workspace.logger.info('Task %s, generating source data', task_id)
                source_data = self.get_data(**task_params['data_params'])

                if self.save_intermediate['data']:
                    self._write_item('source_data', task_id, source_data)
            workspace.logger.debug(
                'Task %s, source data : %s',
                task_id,
                {k: type(v) for k, v in source_data.items()},
            )

        except Exception as exc:
            self._handle_exception_in_run_task(
                exc, task_id, task_params, logger=workspace.logger
            )
            raise exc

        try:
            if self._has_item('problem_data', task_id) and self._has_item(
                'solution_data', task_id
            ):
                workspace.logger.info(
                    'Task %s, problem data and solution data already ' 'generated',
                    task_id,
                )
                problem_data = self._read_item('problem_data', task_id)
                solution_data = self._read_item('solution_data', task_id)
            else:
                workspace.logger.info('Task %s, loading problem', task_id)
                problem = self.get_problem(**task_params['problem_params'])

                workspace.logger.info('Task %s, generating problem', task_id)
                problem_params = inspect.signature(problem).parameters
                for param in source_data.keys():
                    if param not in problem_params:
                        msg = (
                            'Task {}, the output of get_data is not '
                            'compatible with the input of the problem '
                            'generated using get_problem. The parameter {} is '
                            'unexpected.'.format(task_id, param)
                        )
                        raise TypeError(msg)

                problem_data, solution_data = problem(**source_data, **extra_params)

                if self.save_intermediate['problem']:
                    self._write_item('problem_data', task_id, problem_data)
                    self._write_item('solution_data', task_id, solution_data)

            workspace.logger.debug(
                'Task %s, problem data : %s',
                task_id,
                {k: type(v) for k, v in problem_data.items()},
            )
            workspace.logger.debug(
                'Task %s, solution data : %s',
                task_id,
                {k: type(v) for k, v in solution_data.items()},
            )
        except Exception as exc:
            self._handle_exception_in_run_task(
                exc, task_id, task_params, logger=workspace.logger
            )
            raise exc

        try:
            if self._has_item('solved_data', task_id):
                workspace.logger.info('Task %s, problem already solved', task_id)
                solved_data = self._read_item('solved_data', task_id)
            else:
                workspace.logger.info('Task %s, loading solver', task_id)
                solver = self.get_solver(**task_params['solver_params'])

                workspace.logger.info('Task %s, solving problem', task_id)
                solver_params = inspect.signature(solver).parameters
                for param in problem_data.keys():
                    if param not in solver_params:
                        msg = (
                            'Task {}, the output of the problem generated '
                            'using get_problem is not compatible with the '
                            'input of the solver generated using get_solver. '
                            'The parameter {} is unexpected.'.format(task_id, param)
                        )
                        raise TypeError(msg)
                solved_data = solver(**problem_data, **extra_params)

                if self.save_intermediate['solver']:
                    self._write_item('solved_data', task_id, solved_data)

            workspace.logger.debug(
                'Task %s, solver data : %s',
                task_id,
                {k: type(v) for k, v in solved_data.items()},
            )
        except Exception as exc:
            self._handle_exception_in_run_task(
                exc, task_id, task_params, logger=workspace.logger
            )
            raise exc

        try:
            if self._has_item('result', task_id):
                workspace.logger.info('Task %s, performance already measured', task_id)
                result = self._read_item('result', task_id)
            else:
                workspace.logger.info('Task %s, measuring performance', task_id)
                result = self.get_measure(
                    task_params=task_params,
                    source_data=source_data,
                    problem_data=problem_data,
                    solution_data=solution_data,
                    solved_data=solved_data,
                    **extra_params,
                )
                self._write_item('result', task_id, result)

            workspace.logger.debug(
                'Task %s, result data: %s',
                task_id,
                {k: type(v) for k, v in result.items()},
            )
        except Exception as exc:
            self._handle_exception_in_run_task(
                exc, task_id, task_params, logger=workspace.logger
            )
            raise exc

        workspace.logger.info('Task %s done.', task_id)

    def get_pending_task_ids(self):
        """
        Get tasks that have not been run and successfully completed.

        Returns
        -------
        set
            Set of pending task ids
        """

        # get the list of task to do
        no_tasks = self._get_all_items('task_params', only_ids=True)
        no_results = self._get_all_items('result', only_ids=True)

        return set(no_tasks).difference(no_results)

    def launch_experiment(
        self, task_ids=None, n_max_processes=None, gpualloc_params=None
    ):
        """Launch an experiment.

        Parameters
        ----------
        task_ids : str or list of ints or None
            List of task ids to run. If str, task ids must be separated by a comma. If
            None, run pending tasks only to avoid running the same tasks several times.
        n_max_processes : int, optional
            Number maximal of processes. If None, set as the number of CPU cores if no
            GPU, otherwise set as the number of GPUs (default: None).
        gputil_params : dict or None
            Parameters for GPUtil.getAvailable function (default: None).
        """
        if task_ids is None:
            task_ids = self.get_pending_task_ids()
        elif isinstance(task_ids, str):
            task_ids = [int(task) for task in task_ids.split(',')]

        self.logger.info('Launch the experiment.')
        self.logger.info('{} tasks to be done.'.format(len(task_ids)))

        # prepare experiment to be run on GPUs
        if gpualloc_params is not None:

            if not torch.cuda.is_available():
                err_msg = 'CUDA is not available.'
                raise NotImplementedError(err_msg)

            from GPUtil import getAvailable, getGPUs, showUtilization

            torch.multiprocessing.set_start_method('spawn')

            # set default number of processes as number of gpus
            if n_max_processes is None:
                n_max_processes = len(getGPUs())

            # it is assumed all GPUs have the same amount of memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / (
                1024**2
            )

            max_memory_task = (
                gpualloc_params.pop('max_memory_task')
                if 'max_memory_task' in gpualloc_params
                else 0.9 * total_memory
            )

            sleep_time = (
                gpualloc_params.pop('sleep_time')
                if 'sleep_time' in gpualloc_params
                else 1
            )
            warmup_time = (
                gpualloc_params.pop('warmup_time')
                if 'warmup_time' in gpualloc_params
                else 1
            )

            gputil_params = {
                'order': 'memory',
                'limit': 1,
                'maxLoad': 0.9,
                'maxMemory': 1 - max_memory_task / total_memory,
                'includeNan': False,
                'excludeID': [],
                'excludeUUID': [],
            }

            gputil_params.update(gpualloc_params)

            on_gpu = True

        else:

            # set default number of processes
            if n_max_processes is None:
                n_max_processes = mp.cpu_count()

            on_gpu = False

        list_processes = []
        is_waiting = False

        for task_id in task_ids:

            # extra parameters to send to the run_task method
            extra_params = {}

            try:

                # wait for going below max number of processes
                while (
                    sum([process.is_alive() for process in list_processes])
                    >= n_max_processes
                ):
                    time.sleep(1)

                    if not is_waiting:
                        self.logger.info(
                            'Max number of processes reached (%d).', n_max_processes
                        )
                    is_waiting = True

                # reset flag when a slot is available
                is_waiting = False

                if on_gpu:

                    # wait for an available GPU
                    torch.cuda.empty_cache()
                    device = getAvailable(**gputil_params)

                    while len(device) == 0:

                        if not is_waiting:
                            self.logger.info('Waiting for a GPU.')
                            showUtilization()
                        is_waiting = True

                        time.sleep(sleep_time)

                        torch.cuda.empty_cache()
                        device = getAvailable(**gputil_params)

                    # reset flag when a GPU is available
                    is_waiting = False

                    # allocate device
                    extra_params['device'] = device[0]

                else:
                    extra_params['device'] = 'cpu'

                # start a new process
                process = mp.Process(
                    target=self.run_task_by_id,
                    args=(task_id, extra_params),
                    name=f'{self.name}_task{task_id}',
                )

                self.logger.info('Starting process %s...', process.name)
                process.start()

                list_processes.append(process)

                if on_gpu:
                    # wait before starting a new process to allow the computation on the
                    # GPU to start
                    time.sleep(warmup_time)

                # delete finished processes from list
                [
                    (
                        list_processes.pop(idp),
                        self.logger.info('Process %s ended.', process.name),
                    )
                    for idp, process in enumerate(list_processes)
                    if not process.is_alive()
                ]

            except Exception as exc:
                self.logger.error(exc)

        # wait for all unfinished processes
        [
            (
                process.join(),
                list_processes.pop(idp),
                self.logger.info('Process %s ended.', process.name),
            )
            for idp, process in enumerate(list_processes)
            if process.is_alive()
        ]

    def collect_results(self, batch_size=None):
        """Collect results from an experiment.

        Merge all the task results in a :class:`numpy.ndarray` and save the
        resulting data with axes labels and values in a file.

        The resulting data are stored in the data directory of the experiment
        in a file named ``results.npz``. This file contains three
        :class:`numpy.ndarray`:

          - results: the collected results,
          - axes_labels: the labels of the axes of results,
          - axes_values: the values corresponding to the indexes for all
            axes of results.

        The dimensions of the main array ``results`` match the parameters of
        the tasks and the performance measures (last dimension). More
        precisely, ``results[i_0, ..., i_j, ..., i_N]`` is the value of the
        performance measure axes_values[N][i_N] for parameter values
        ``axes_values[j][i_j]``, ``j < N``, related to parameter names
        ``axes_label[j]``.

        The stored data can be loaded using the method
        :meth:`~yafe.base.Experiment.load_results`.

        Raises
        ------
        RuntimeError
            If no results have been computed prior to running the method.
        """
        # get the tasks and the results
        tasks_params = self._get_all_items('task_params')
        results = self._get_all_items('result')

        if results:
            measures_names = list(next(iter(results.values())).keys())
        else:
            raise RuntimeError(
                'No result available. Results must be computed '
                'before running collect_results().'
            )

        self.logger.info('Collecting {} tasks.'.format(len(tasks_params)))

        # add axes corresponding to the parameters of the experiment
        schema = self._schema

        shape = [len(value) for value in schema.values()]
        shape.append(len(measures_names))

        if batch_size:
            shape.append(batch_size)

        dtype = np.float64
        array_results = np.empty(shape, dtype=dtype)
        array_results.fill(np.nan)

        # load the results and store them
        missing_tasks = dict()
        for measure in measures_names:
            missing_tasks[measure] = list()

        for idt, task in tasks_params.items():

            indices = []
            for (prefix, name), ticks in schema.items():
                indices.append(ticks.index(task[prefix + '_params'][name]))
            # CHECK: This slice seems to not do anything and also breaks
            # the latest run scripts Commented out.
            # indices.append(slice(None))
            indices = tuple(indices)

            if batch_size:
                result = np.empty((len(measures_names), batch_size))
            else:
                result = np.empty(len(measures_names))
            result.fill(np.nan)
            for idm, measure in enumerate(measures_names):
                try:
                    if batch_size:
                        result[idm, :] = results[idt][measure]
                    else:
                        result[idm] = results[idt][measure]
                except KeyError:
                    missing_tasks[measure].append(idt)

            if batch_size:
                array_results[indices, :] = result
            else:
                array_results[indices] = result

        for measure in measures_names:
            self.logger.info(
                'Measure `{}`: {} missing tasks.'.format(
                    measure, len(missing_tasks[measure])
                )
            )
            self.logger.debug(missing_tasks)

        axes_labels = [cat + '_' + param for (cat, param) in schema.keys()]
        axes_labels.append('measure')
        axes_labels = np.array(axes_labels, dtype=object)
        axes_values = np.empty_like(axes_labels, dtype=object)
        for ind, val in enumerate(schema.values()):
            axes_values[ind] = np.array(val)
        axes_values[-1] = np.array(measures_names, dtype=object)

        results_path = str(self.xp_path / 'results.npz')
        self.logger.info(f'Saving results at {results_path}...')

        np.savez_compressed(
            results_path,
            results=array_results,
            axes_labels=axes_labels,
            axes_values=axes_values,
        )
        self.logger.info('Done.')

    def load_results(self, array_type='numpy', batch_size=None):
        """Load the collected results from an experiment.

        Load the data stored by the method
        :meth:`~yafe.base.Experiment.collect_results` in the file
        ``results.npz``.

        Parameters
        ----------
        array_type : {'numpy', 'xarray'}
            Type of output. With the option ``'numpy'``, the output is split
            into three :class:`numpy.ndarray`, while with the option
            ``'xarray'`` the output is a single :class:`xarray.DataArray`.

        Returns
        -------
        (results, axes_labels, axes_values) : tuple of :class:`numpy.ndarray`
            Returned when using the default option ``array_type='numpy'``.
            See :meth:`~yafe.base.Experiment.collect_results` for a detailed
            description.
        xresults : :class:`xarray.DataArray`
            Returned when using the option ``array_type='xarray'``. This is
            equivalent to the case ``array_type='numpy'``, combining all
            three arrays into a single data structure for improved consistency
            and ease of use. ``xresults`` can be obtained from
            ``(results, axes_labels, axes_values)`` as ::

                xresults = xarray.DataArray(data=results,
                                            dims=axes_labels,
                                            coords=axes_values)

        Raises
        ------
        ImportError
            If the xarray package is not installed and the
            ``array_type='xarray'`` option is used.
        ValueError
            If an unrecognized value is given for ``array_type``.
        RuntimeError
            If no collected results file can be found.
        """
        self.logger.info('Loading yafe experiment results as %s...', array_type)

        if (self.xp_path / 'results.npz').exists():
            data = np.load(str(self.xp_path / 'results.npz'), allow_pickle=True)

            axes_labels = data['axes_labels'].tolist()
            axes_values = [item.tolist() for item in data['axes_values'].tolist()]

            if batch_size:
                axes_labels.append('batch')
                axes_values.append(list(range(batch_size)))

            if array_type == 'numpy':
                results = (data['results'], axes_labels, axes_values)
            elif array_type == 'xarray':

                try:
                    import xarray
                except ImportError:
                    raise ImportError(
                        'Please install the xarray package to '
                        'run the load_results() method with the '
                        'array_type="xarray" option.'
                    )
                results = xarray.DataArray(
                    data=data['results'], dims=axes_labels, coords=axes_values
                )
            else:
                errmsg = (
                    'Unrecognized value for array_type, the valid '
                    'options are "numpy" and "xarray".'
                )
                raise ValueError(errmsg)

        else:
            raise RuntimeError(
                'No file with collected results available. '
                'Results must be computed and collected before '
                'running load_results().'
            )

        self.logger.info('Done.')
        return results

    def get_status(self, task_id=None):
        """Return a string describing the status of the tasks.

        Parameters
        ----------
        summary : bool
            Indicates if only a summary of the status is provided or not.
        task_ids : list of ints or None
            List of tasks to which the status is returned.

        Returns
        -------
        str
            Status of the tasks.
        """
        ids_tasks = self._get_all_items('task_params', only_ids=True)
        ids_results = self._get_all_items('result', only_ids=True)
        ids_errors = self._get_all_items('error_log', only_ids=True)

        summary = task_id is None

        status = ''

        if summary:
            status += 'Status:\n'
            status += '\tNumber of tasks: {}\n'.format(len(ids_tasks))
            status += '\tNumber of finished tasks: {}\n'.format(len(ids_results))
            status += '\tNumber of failed tasks: {}\n'.format(len(ids_errors))
            status += '\tNumber of pending tasks: {}\n'.format(
                len(ids_tasks) - len(ids_results)
            )
        else:
            for id_task in [task_id] if task_id is not None else sorted(ids_tasks):
                width = 26
                nb_digits_id = 6
                id_format = '{{:0{nb}d}}'.format(nb=nb_digits_id)
                status += id_format.format(id_task)

                if id_task in ids_results:
                    msg = 'OK'
                    error = None
                else:
                    if id_task in ids_errors:
                        msg = 'ERROR'
                        error = self._read_item('error_log', id_task)
                    else:
                        msg = 'NOT CALCULATED'
                        error = None

                status += '.' * (width - nb_digits_id - len(msg)) + msg + '\n'
                if error is not None:
                    status += '-------------------------\n'
                    status += error
                    status += '-------------------------\n'
        return status

    def get_task_data_by_id(self, task_id):
        """Get data for a task specified by its id.

        Parameters
        ----------
        task_id : int
            Id of the task.

        Returns
        -------
        dict
            Task data.

        Raises
        ------
        ValueError
            If ``task_id`` is not a valid value.
        """
        task_params = self._read_item('task_params', task_id)
        if task_params is None:
            err_msg = 'Unknown task: {}'.format(task_id)
            self.logger.error(err_msg)
            raise ValueError(err_msg)
        source_data = self._read_item('source_data', task_id)
        problem_data = self._read_item('problem_data', task_id)
        solution_data = self._read_item('solution_data', task_id)
        solved_data = self._read_item('solved_data', task_id)
        result = self._read_item('result', task_id)
        return {
            'id_task': task_id,
            'task_params': task_params,
            'source_data': source_data,
            'problem_data': problem_data,
            'solution_data': solution_data,
            'solved_data': solved_data,
            'result': result,
        }

    def get_task_data_by_params(self, data_params, problem_params, solver_params):
        """Get data for a task specified by its parameters.

        Parameters
        ----------
        data_params : dict
            Parameters of generation of the dataset.
        problem_params : dict
            Parameters of generation of the problem.
        solver_params : dict
            Parameters of generation of the solver.

        Returns
        -------
        dict
            Task data.

        Raises
        ------
        TypeError
            If parameters are not dict.
        ValueError
            If parameters contain unexpected keys or unexpected values.
        """
        if (
            not isinstance(data_params, dict)
            or not isinstance(problem_params, dict)
            or not isinstance(solver_params, dict)
        ):
            msg = 'Parameters must be of type dict.'
            raise TypeError(msg)

        msg = 'Wrong keys in {}, the expected keys are: {}.'
        data_params_names = inspect.signature(self.get_data).parameters
        if set(data_params.keys()) != set(data_params_names):
            raise ValueError(msg.format('data_params', ', '.join(data_params_names)))

        problem_params_names = inspect.signature(self.get_problem).parameters
        if set(problem_params.keys()) != set(problem_params_names):
            raise ValueError(
                msg.format('problem_params', ', '.join(problem_params_names))
            )

        solver_params_names = inspect.signature(self.get_solver).parameters
        if set(solver_params.keys()) != set(solver_params_names):
            raise ValueError(
                msg.format('solver_params', ', '.join(solver_params_names))
            )

        id_task = None
        all_tasks_params = self._get_all_items('task_params')
        params = {
            'data_params': data_params,
            'problem_params': problem_params,
            'solver_params': solver_params,
        }
        for task_id, task_params in all_tasks_params.items():
            if params == task_params:
                id_task = task_id
                break

        if id_task is not None:
            return self.get_task_data_by_id(id_task)
        else:
            err_msg = 'No task exists for the given parameters.'
            self.logger.error(err_msg)
            raise ValueError(err_msg)

    def save(self):
        """Save the details of an experiment."""

        # save state
        state_path = self.xp_path / 'init.pickle'
        self.logger.info('Saving state of experiment at %s...', state_path)
        with open(state_path, 'wb') as outfile:
            dill.dump(self, outfile)
        self.logger.info('Done.')

    @staticmethod
    def load(name, workspace):
        """Load an experiment."""

        state_path = workspace.xps_path / name / 'init.pickle'

        if not state_path.exists():
            err_msg = 'Experiment workspace does not exist at {}'
            raise FileNotFoundError(err_msg.format(workspace.xps_path / name))

        workspace.logger.info('Loading state of experiment at %s...', state_path)
        with open(state_path, 'rb') as infile:
            experiment = dill.load(infile)
        workspace.logger.info('Done.')

        experiment.workspace = workspace
        experiment.xp_path = workspace.xps_path / name

        return experiment

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['workspace']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):

        string = '\nName: {}\n'.format(self.name)
        string += 'Data stored in: {}\n'.format(self.xp_path)

        logger_path = None

        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger_path = handler.baseFilename
                break

        if logger_path is None:
            string += 'Not storing any debug information\n'
        else:
            string += 'Debug information stored in: {}\n'.format(logger_path)

        string += 'Data provider: {}\n'.format(self.get_data)
        string += 'Problem provider : {}\n'.format(self.get_problem)
        string += 'Solver provider: {}\n'.format(self.get_solver)
        string += 'Measure: {}\n'.format(self.get_measure)
        string += '\n'
        string += 'Parameters:\n'

        data_param_str = ''
        problem_param_str = ''
        solver_param_str = ''
        for (cat, param_name), val in self._schema.items():
            if cat == 'data':
                data_param_str += '\t\t{}: {}\n'.format(param_name, val)
            elif cat == 'problem':
                problem_param_str += '\t\t{}: {}\n'.format(param_name, val)
            else:  # cat == 'solver':
                solver_param_str += '\t\t{}: {}\n'.format(param_name, val)

        string += '\tData parameters:\n'
        string += data_param_str
        string += '\tProblem parameters:\n'
        string += problem_param_str
        string += '\tSolver parameters:\n'
        string += solver_param_str
        string += '\n'
        string += self.get_status()

        return string
