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
# yafe is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ######### COPYRIGHT #########
"""Test of the module :module:`yafe.base`

.. moduleauthor:: Ronan Hamon
.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Florent Jaillet
"""
import contextlib
import json
import os
from pathlib import Path
from itertools import product
from shutil import rmtree
from io import StringIO
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from yafe.base import Experiment
from yafe.utils import ConfigParser


def create_temp_test_config():
    temp_dir = tempfile.mkdtemp(prefix='yafe_')
    ConfigParser._config_path = temp_dir / Path('yafe.conf')
    config_parser = ConfigParser()
    data_path = temp_dir / Path('data')
    data_path.mkdir()
    config_parser.set('USER', 'data_path', str(data_path))
    logger_path = temp_dir / Path('yafe.log')
    config_parser.set('LOGGER', 'path', str(logger_path))
    with open(str(ConfigParser._config_path), 'w') as config_file:
        config_parser.write(config_file)
    return temp_dir


def get_data_path():
    config = ConfigParser()
    data_path = config.get_path('USER', 'data_path')
    return data_path


# Mock component with arbitrary behavior
class MockSolver:
    def __init__(self, f, seed):
        self.f = f
        self.seed = seed

    def __call__(self, X):
        if self.f == 1000:
            raise ValueError('Exception in solve for testing')
        return {'X': X + self.f}


class MockProblem:
    def __init__(self, c, d, e, seed):
        self.value = c + d + len(e)
        self.seed = seed

    def __call__(self, X):
        np.random.seed(self.seed)
        if self.value == 1000:
            raise ValueError('Exception in generate for testing')
        problem_data = {'X': X + self.value}
        solution_data = {'X': X + self.value + 1}
        return problem_data, solution_data


def measure(task_params, source_data, problem_data, solution_data,
            solved_data):
    if task_params['data_params']['a'] == 10000:
        raise ValueError('Exception in measure for testing')
    return {'snr': 0.2, 'error': 1.6}


class MockExperiment(Experiment):
    def __init__(self, name, force_reset=False, data_path=None):

        def get_data(a, b, seed):
            if a == 1000:
                raise ValueError('Exception in get_data for testing')
            np.random.seed(seed)
            return {'X': np.arange(a + b)}

        def get_problem(c, d, e, seed):
            return MockProblem(c, d, e, seed)

        def get_solver(f, seed):
            return MockSolver(f, seed)

        super().__init__(name, get_data=get_data, get_problem=get_problem,
                         get_solver=get_solver, measure=measure,
                         force_reset=force_reset, data_path=data_path)


# Mock components to track the tasks contents
class MockTrackData:
    def __call__(self, data_param1):
        return {'mock_data': np.array([data_param1])}

    @staticmethod
    def get_params():
        return {
            'data_param1': [-1, 2, -4],  # not sorted
        }

    @staticmethod
    def get_additional_params():
        return {'data_param1': [6]}


class MockTrackProblem:
    # argument names not in lexicographic order
    def __init__(self, problem_param3, problem_param1, problem_param2):
        self.problem_param3 = problem_param3
        self.problem_param1 = problem_param1
        self.problem_param2 = problem_param2

    def __call__(self, mock_data):
        mock_pb_data = np.concatenate((mock_data, [self.problem_param3,
                                                   self.problem_param1,
                                                   self.problem_param2]))
        problem_data = {'mock_pb_data': mock_pb_data}
        mock_sol_data = np.concatenate((mock_data, [self.problem_param3,
                                                    self.problem_param1,
                                                    self.problem_param2]))
        solution_data = {'mock_sol_data': mock_sol_data}
        return problem_data, solution_data

    @staticmethod
    def get_params():
        return {
            'problem_param3': [10, 11, 12],  # sorted
            'problem_param1': [20],  # singleton
            'problem_param2': [33, 32],  # not sorted
        }

    @staticmethod
    def get_additional_params():
        return {'problem_param2': [20]}


class MockTrackSolver:
    # order of arguments solver_param2 and solver_param1 differ from order in
    # dict
    def __init__(self, solver_param2, solver_param1):
        self.solver_param2 = solver_param2
        self.solver_param1 = solver_param1

    def __call__(self, mock_pb_data):
        mock_solver_data = np.concatenate(
            (mock_pb_data, [self.solver_param2, self.solver_param1]))
        return {'mock_solver_data': mock_solver_data}

    @staticmethod
    def get_params():
        return {
            'solver_param1': [41],  # singleton
            'solver_param2': [54, 53],  # not sorted
        }

    @staticmethod
    def get_additional_params():
        return{
            'solver_param1': [39],
            'solver_param2': [54, 53.5],
        }


def mock_track_measure(solved_data, source_data=None, task_params=None,
                       problem_data=None, solution_data=None):
    return {
        'meas_data_param1': solved_data['mock_solver_data'][0],
        'meas_problem_param3': solved_data['mock_solver_data'][1],
        'meas_problem_param1': solved_data['mock_solver_data'][2],
        'meas_problem_param2': solved_data['mock_solver_data'][3],
        'meas_solver_param2': solved_data['mock_solver_data'][4],
        'meas_solver_param1': solved_data['mock_solver_data'][5],
    }


def check_tracked_results(results, axes_labels, axes_values):
    # Check shape
    np.testing.assert_array_equal(results.shape,
                                  [a.size for a in axes_values])

    # Check axes_labels
    measure_names = axes_values[axes_labels == 'measure'][0]
    for i, name in enumerate(measure_names):
        assert name.startswith('meas_')
        if axes_labels[i].startswith('data_'):
            assert name[len('meas_'):] == axes_labels[i][len('data_'):]
        elif axes_labels[i].startswith('problem_'):
            assert name[len('meas_'):] == axes_labels[i][len('problem_'):]
        elif axes_labels[i].startswith('solver_'):
            assert name[len('meas_'):] == axes_labels[i][len('solver_'):]
        else:
            raise ValueError(
                '{} does not begin with {}.'.format(
                    axes_labels[i], ('data_', 'problem_', 'solver_')))

    # Check first element in results
    value = results[(0,) * (results.ndim - 1)]
    for i in range(results.ndim-1):
        assert value[i] == axes_values[i][0]

    # Check all elements in results
    axes_values_params = axes_values[axes_labels != 'measure']
    indexes = [np.arange(a.size) for a in axes_values_params]
    for multi_index, multi_value in zip(product(*indexes),
                                        product(*axes_values_params)):
        for j in range(len(multi_value)):
            assert results[multi_index][j] == axes_values[j][multi_index[j]]


class TestExperiment(unittest.TestCase):

    def setUp(self):
        self.item = {'a': 1, 'e': 'alpha', 'f': [0, 2, 8], 'g': (4, 7)}

        self.data_params = {'a': [1, 2, 3], 'b': [4, 5], 'seed': 12}
        self.problem_params = {'c': [3, 2, 1],
                               'd': 5, 'e': ['alpha', 'beta'], 'seed': 12}
        self.solver_params = {'f': 31, 'seed': 12}

        all_values = list(self.data_params.values()) \
            + list(self.problem_params.values()) \
            + list(self.solver_params.values())
        all_len = [len(val) for val in all_values if isinstance(val, list)]
        self.expected_nb_tasks = np.product(all_len)

        self._temp_dir = create_temp_test_config()

    def tearDown(self):
        rmtree(self._temp_dir)

    def test_init(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:

            xp_name = Path(tmpdir).stem
            Path(tmpdir).rmdir()

            # Init a new experiment
            xp = MockExperiment(xp_name)

            self.assertEqual(str(xp.xp_path), tmpdir, msg='xp path')
            self.assertTrue(xp.xp_path.exists(), msg='existence of xp folder')
            self.assertTrue((xp.xp_path / 'tasks').exists(),
                            msg='existence of tasks folder')

            # add files in subfolders
            (xp.xp_path / 't.test').touch()
            (xp.xp_path / 'tasks' / 'tas.test').touch()

            # load existing xp
            xp = MockExperiment(xp_name)

            self.assertTrue((xp.xp_path / 't.test').exists(),
                            msg='file exists in xp folder ')
            self.assertTrue((xp.xp_path / 'tasks' / 'tas.test').exists(),
                            msg='file exists in tasks folder')

            # load existing xp without _schema
            os.remove(xp.xp_path / '_schema.pickle')
            xp = MockExperiment(xp_name)

            self.assertFalse((xp.xp_path / 't.test').exists(),
                             msg='file does not exists in xp folder')
            self.assertFalse((xp.xp_path / 'tasks' / 'tas.test').exists(),
                             msg='file does not exist in tasks folder')
            self.assertEqual(xp._schema,
                             {('data', 'a'): [],
                              ('data', 'b'): [],
                              ('data', 'seed'): [],
                              ('problem', 'c'): [],
                              ('problem', 'd'): [],
                              ('problem', 'e'): [],
                              ('problem', 'seed'): [],
                              ('solver', 'f'): [],
                              ('solver', 'seed'): []},
                             msg='_schema')

            # add files in subfolders
            (xp.xp_path / 't.test').touch()
            (xp.xp_path / 'tasks' / 'tas.test').touch()

            # check that the force_reset works
            xp = MockExperiment(xp_name, force_reset=True)

            self.assertFalse((xp.xp_path / 't.test').exists(),
                             msg='file does exists in xp folder ')
            self.assertFalse((xp.xp_path / 'tasks' / 'tas.test').exists(),
                             msg='file does exists in tasks folder')
            self.assertEqual(xp._schema,
                             {('data', 'a'): [],
                              ('data', 'b'): [],
                              ('data', 'seed'): [],
                              ('problem', 'c'): [],
                              ('problem', 'd'): [],
                              ('problem', 'e'): [],
                              ('problem', 'seed'): [],
                              ('solver', 'f'): [],
                              ('solver', 'seed'): []},
                             msg='_schema')

            # load existing xp with incoherent _schema.pickle file
            def get_data(a, b):
                return {'X': np.arange(a + b)}
            error_msg = ('The schema of the Experiment does not match the '
                         'schema of the Experiment with the same name stored '
                         'in ')
            with self.assertRaisesRegex(RuntimeError, error_msg):
                xp = Experiment(
                    xp_name, get_data=get_data, get_problem=MockProblem,
                    get_solver=MockSolver, measure=measure)

            os.remove(xp.xp_path / '_schema.pickle')
            with self.assertRaises(FileNotFoundError, msg='Missing _schema'):
                xp._schema

        tempfile.TemporaryDirectory(dir=str(get_data_path()))

        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem

            def measure_wrong_inputs(task_params, problem_data, solved_data):
                return {'snr': 0.2, 'error': 1.6}

            def get_data(a, b, seed):
                return {'X': np.arange(a + b)}

            error_msg = ('Wrong signature for measure. measure must accept '
                         'the following parameters: task_params, source_data, '
                         'problem_data, solution_data, solved_data.')
            with self.assertRaisesRegex(TypeError, error_msg):
                _ = Experiment(name=xp_name,
                               get_data=get_data,
                               get_problem=MockProblem,
                               get_solver=MockSolver,
                               measure=measure_wrong_inputs)

    def test_str(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)
            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()

            str_xp = str(xp)

            expected_start = (
                'Name: tmp.*\n'
                'Data stored in: /.*\n'
                'Debug information stored in: /.*\n'
                'Data provider: <function MockExperiment.__init__.<locals>.'
                'get_data at 0x[0-9a-f]*>\n'
                'Problem provider : <function MockExperiment.__init__.'
                '<locals>.get_problem at 0x[0-9a-f]*>\n'
                'Solver provider: <function MockExperiment.__init__.'
                '<locals>.get_solver at 0x[0-9a-f]*>\n'
                'Measure: <function measure at 0x[0-9a-f]*>\n')
            self.assertRegex(str_xp, expected_start)

            expected_end = (
                '\n'
                'Parameters:\n'
                ' Data parameters:\n'
                '  a: [1, 2, 3]\n'
                '  b: [4, 5]\n'
                '  seed: [12]\n'
                ' Problem parameters:\n'
                '  c: [1, 2, 3]\n'
                '  d: [5]\n'
                "  e: ['alpha', 'beta']\n"
                '  seed: [12]\n'
                ' Solver parameters:\n'
                '  f: [31]\n'
                '  seed: [12]\n'
                '\n'
                'Status:\n'
                'Number of tasks: 36\n'
                'Number of finished tasks: 0\n'
                'Number of failed tasks: 0\n'
                'Number of pending tasks: 36\n')
            self.assertEqual(str_xp[-len(expected_end):], expected_end)

        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            def get_data(a, b, seed):
                np.random.seed(seed)
                return {'X': np.arange(a + b)}
            xp_name = Path(tmpdir).stem
            xp = Experiment(xp_name,
                            get_data=get_data,
                            get_problem=MockProblem,
                            get_solver=MockSolver,
                            measure=measure,
                            log_to_file=False,
                            log_to_console=True)
            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()

            str_xp = str(xp)

            expected_start = (
                'Name: tmp.*\n'
                'Data stored in: /.*\n'
                'Not storing any debug information\n'
                'Data provider: <function TestExperiment.test_str.<locals>.'
                'get_data at 0x[0-9a-f]*>\n'
                'Problem provider : '
                "<class 'yafe.tests.test_base.MockProblem'>\n"
                "Solver provider: <class 'yafe.tests.test_base.MockSolver'>\n"
                'Measure: <function measure at 0x[0-9a-f]*>\n')
            self.assertRegex(str_xp, expected_start)

    def test_reset(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            Path(tmpdir).rmdir()

            # Init a new experiment
            xp = MockExperiment(xp_name)

            # add files in subfolders
            (xp.xp_path / 't.test').touch()
            (xp.xp_path / 'tasks' / 'tas.test').touch()

            # Reset the xp
            xp.reset()

            self.assertFalse((xp.xp_path / 't.test').exists(),
                             msg='file does not exist in xp folder')
            self.assertFalse((xp.xp_path / 'tasks' / 'tas.test').exists(),
                             msg='file does not exist in tasks folder')

            # The following is needed to reach 100% coverage
            rmtree(str(xp.xp_path))
            xp.reset()

            self.assertFalse((xp.xp_path / 't.test').exists(),
                             msg='file does not exist in xp folder')
            self.assertFalse((xp.xp_path / 'tasks' / 'tas.test').exists(),
                             msg='file does not exist in tasks folder')

    def test_write_read_delete_item(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            xp._write_item('name_item', 10, self.item)

            self.assertTrue(
                (Path(tmpdir) / 'tasks' / '000010').exists(),
                msg='existence of folder')

            self.assertTrue(
                (Path(tmpdir) / 'tasks' / '000010' /
                 'name_item.pickle').exists(),
                msg='existence of item in folder')

            self.assertEqual(
                json.dumps(xp._read_item('name_item', 10), sort_keys=True),
                json.dumps(self.item, sort_keys=True),
                msg='content of item')

            self.assertTrue(
                xp._read_item('name_item', 11) is None,
                msg='error if missing task folder')

            self.assertTrue(
                xp._read_item(
                    'unknown',
                    10, ) is None,
                msg='error if unknown item')

            xp._write_item('name_item', 4, self.item)
            xp._write_item('name_item', 7, self.item)
            xp._delete_item(11)
            xp._delete_item(10)
            self.assertTrue(
                not (Path(tmpdir) / 'tasks' / '000010').exists(),
                msg='supression of folder')
            xp._delete_item(7, 'wrong_name')
            self.assertTrue(
                (Path(tmpdir) / 'tasks' /
                 '000007'/'name_item.pickle').exists(),
                msg='supression of folder')
            xp._delete_item(7, 'name_item')
            self.assertTrue(
                (Path(tmpdir) / 'tasks' / '000007').exists(),
                msg='supression of folder')
            self.assertTrue(
                not (Path(tmpdir) / 'tasks' /
                     '000007'/'name_item.pickle').exists(),
                msg='supression of folder')

    def test_get_all_items(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            xp._write_item('name_item', 0, self.item)
            xp._write_item('name_item', 1, self.item)
            xp._write_item('name_item', 2, self.item)

            items = xp._get_all_items('name_item')

            self.assertTrue(
                isinstance(items, dict), msg='if output is dict')
            self.assertEqual(len(items), 3, msg='number of items')
            for idi, item in items.items():
                self.assertEqual(
                    item, self.item, msg='item {}'.format(idi))

            items = xp._get_all_items('name_item', only_ids=True)
            self.assertTrue(
                isinstance(items, list), msg='if output is list')
            self.assertEqual(len(items), 3, msg='number of items')
            self.assertEqual(set(items), set(range(3)), 'items')

    def test_add_tasks(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:

            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            # Add tasks
            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)

            schema = xp._schema
            self.assertTrue(isinstance(schema, dict), msg='type _schema')
            expected_schema = {('data', 'a'): [1, 2, 3],
                               ('data', 'b'): [4, 5],
                               ('data', 'seed'): [12],
                               ('problem', 'c'): [1, 2, 3],
                               ('problem', 'd'): [5],
                               ('problem', 'e'): ['alpha', 'beta'],
                               ('problem', 'seed'): [12],
                               ('solver', 'f'): [31],
                               ('solver', 'seed'): [12]}
            self.assertEqual(schema, expected_schema, msg='Value _schema')

            # Add new tasks
            new_data_params = {'a': [10, 11]}

            xp.add_tasks(new_data_params, self.problem_params,
                         self.solver_params)

            expected_schema = {('data', 'a'): [1, 2, 3, 10, 11],
                               ('data', 'b'): [4, 5],
                               ('data', 'seed'): [12],
                               ('problem', 'c'): [1, 2, 3],
                               ('problem', 'd'): [5],
                               ('problem', 'e'): ['alpha', 'beta'],
                               ('problem', 'seed'): [12],
                               ('solver', 'f'): [31],
                               ('solver', 'seed'): [12]}

            schema = xp._schema
            self.assertEqual(
                schema[('data', 'a')], [1, 2, 3, 10, 11], msg='update _schema')

            error_msg = 'data_params should be a dict'
            msg = 'error if wrong data params'
            with self.assertRaisesRegex(TypeError, error_msg, msg=msg):
                xp.add_tasks([2, 5], self.problem_params, self.solver_params)

            error_msg = 'problem_params should be a dict'
            msg = 'error if wrong problem params'
            with self.assertRaisesRegex(TypeError, error_msg, msg=msg):
                xp.add_tasks(self.data_params, [2, 5], self.solver_params)

            error_msg = 'solver_params should be a dict'
            msg = 'error if wrong solver params'
            with self.assertRaisesRegex(TypeError, error_msg, msg=msg):
                xp.add_tasks(self.data_params, self.problem_params, [2, 5])

            error_msg = 'wrong_param is not a parameter for getting data.'
            msg = 'error if wrong data param name'
            with self.assertRaisesRegex(ValueError, error_msg, msg=msg):
                xp.add_tasks({'wrong_param': 10}, self.problem_params,
                             self.solver_params)

            error_msg = 'wrong_param is not a parameter for getting the ' \
                        'problem.'
            msg = 'error if wrong problem param name'
            with self.assertRaisesRegex(ValueError, error_msg, msg=msg):
                xp.add_tasks(self.data_params, {'wrong_param': 10},
                             self.solver_params)

            error_msg = 'wrong_param is not a parameter for getting the ' \
                        'solver.'
            msg = 'error if wrong solver param name'
            with self.assertRaisesRegex(ValueError, error_msg, msg=msg):
                xp.add_tasks(self.data_params, self.problem_params,
                             {'wrong_param': 10})

    def test_generate_tasks(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:

            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)

            xp.generate_tasks()
            tasks = list(xp._get_all_items('task_params', only_ids=True))

            self.assertEqual(len(tasks), self.expected_nb_tasks,
                             msg='number of tasks')

            # Select a random task
            tasks = list(xp._get_all_items('task_params').values())
            idt = np.random.randint(len(tasks))
            task = tasks[idt]

            self.assertTrue(
                'data_params' in task, msg='data_params in task')
            self.assertTrue(
                task['data_params']['a'] in self.data_params['a'], msg='a')
            self.assertTrue(
                task['data_params']['b'] in self.data_params['b'], msg='b')

            self.assertTrue(
                'problem_params' in task, msg='problems_params in task')
            self.assertTrue(
                task['problem_params']['c'] in self.problem_params['c'],
                msg='c')
            self.assertTrue(
                task['problem_params']['d'] in self.problem_params['d'],
                msg='d')
            self.assertTrue(
                task['problem_params']['e'] in self.problem_params['e'],
                msg='e')

            self.assertTrue(
                'solver_params' in task, msg='solver_params in task')
            self.assertTrue(
                task['solver_params']['f'] in self.solver_params['f'],
                msg='e')

            xp.add_tasks({'a': 24}, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()
            tasks = list(xp._get_all_items('task_params', only_ids=True))

            all_values = list(self.data_params.values()) \
                + list(self.problem_params.values()) \
                + list(self.solver_params.values())
            all_len = [len(val) for val in all_values if isinstance(val, list)]
            all_len[0] += 1
            expected_nb_tasks = np.product(all_len)

            self.assertEqual(len(tasks), expected_nb_tasks,
                             msg='number of tasks')

    def test_run_task_by_id(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()
            xp.run_task_by_id(10)

            results = xp._get_all_items('result')
            self.assertTrue(10 in results, msg='the presence of a result')

            self.assertEqual(results[10], {'snr': 0.2, 'error': 1.6})

            # Rerun the task to check that already computed data are not
            # recomputed which is confirmed by checking the coverage report for
            # the corresponding part of the code.
            xp.run_task_by_id(10)

            error_msg = 'Unknown task: 100'
            msg = 'error if unknown task'
            with self.assertRaisesRegex(ValueError, error_msg, msg=msg):
                xp.run_task_by_id(100)

        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)
            data_params = {'a': 1000, 'b': 0, 'seed': 12}
            problem_params = {'c': 1, 'd': 5, 'e': 'alpha', 'seed': 12}
            solver_params = {'f': 31, 'seed': 12}
            xp.add_tasks(data_params, problem_params, solver_params)
            xp.generate_tasks()
            error_msg = 'Exception in get_data for testing'
            with self.assertRaisesRegex(ValueError, error_msg):
                xp.run_task_by_id(0)

        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)
            data_params = {'a': 1, 'b': 0, 'seed': 12}
            problem_params = {'c': 1000, 'd': 0, 'e': '', 'seed': 12}
            solver_params = {'f': 31, 'seed': 12}
            xp.add_tasks(data_params, problem_params, solver_params)
            xp.generate_tasks()
            error_msg = 'Exception in generate for testing'
            with self.assertRaisesRegex(ValueError, error_msg):
                xp.run_task_by_id(0)

        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)
            data_params = {'a': 1, 'b': 0, 'seed': 12}
            problem_params = {'c': 1, 'd': 0, 'e': 'alpha', 'seed': 12}
            solver_params = {'f': 1000, 'seed': 12}
            xp.add_tasks(data_params, problem_params, solver_params)
            xp.generate_tasks()
            error_msg = 'Exception in solve for testing'
            with self.assertRaisesRegex(ValueError, error_msg):
                xp.run_task_by_id(0)

        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)
            data_params = {'a': 10000, 'b': 0, 'seed': 12}
            problem_params = {'c': 1, 'd': 0, 'e': 'alpha', 'seed': 12}
            solver_params = {'f': 0, 'seed': 12}
            xp.add_tasks(data_params, problem_params, solver_params)
            xp.generate_tasks()
            error_msg = 'Exception in measure for testing'
            with self.assertRaisesRegex(ValueError, error_msg):
                xp.run_task_by_id(0)

        def get_data_failing(a, b, seed):
            raise ValueError('get_data failed')

        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)
            get_data_original = xp.get_data
            xp.get_data = get_data_failing
            data_params = {'a': 1, 'b': 0, 'seed': 12}
            problem_params = {'c': 1, 'd': 5, 'e': 'alpha', 'seed': 12}
            solver_params = {'f': 31, 'seed': 12}
            xp.add_tasks(data_params, problem_params, solver_params)
            xp.generate_tasks()
            error_msg = 'get_data failed'
            with self.assertRaisesRegex(ValueError, error_msg):
                xp.run_task_by_id(0)
            self.assertTrue(xp._has_item('error_log', 0))
            xp.get_data = get_data_original
            xp.run_task_by_id(0)
            self.assertFalse(xp._has_item('error_log', 0))

        def get_data_wrong_output(a, b, seed):
            return {'Y': np.arange(a + b)}

        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)
            xp.get_data = get_data_wrong_output
            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()
            error_msg = (r'Task 10, the output of get_data is not compatible '
                         r'with the input of the problem generated using '
                         r'get_problem. The parameter Y is unexpected.')
            with self.assertRaisesRegex(TypeError, error_msg):
                xp.run_task_by_id(10)

        class ProblemWrongOutput:
            def __init__(self, c, d, e, seed):
                pass

            def __call__(self, X):
                problem_data = {'Y': X}
                solution_data = dict()
                return problem_data, solution_data

        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)
            xp.get_problem = ProblemWrongOutput
            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()
            error_msg = (r'Task 10, the output of the problem generated using '
                         r'get_problem is not compatible with the input of '
                         r'the solver generated using get_solver. '
                         r'The parameter Y is unexpected.')
            with self.assertRaisesRegex(TypeError, error_msg):
                xp.run_task_by_id(10)

    def test_launch_experiment(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()
            xp.launch_experiment()
            results = xp._get_all_items('result')

            self.assertEqual(len(results), self.expected_nb_tasks,
                             msg='the number of results')

        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()
            xp.launch_experiment([2, 4, 8, 9])

            results = xp._get_all_items('result')
            self.assertEqual(len(results), 4, msg='the number of results')

        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()
            xp.launch_experiment('1,3,7')

            results = xp._get_all_items('result')
            self.assertEqual(len(results), 3, msg='the number of results')

        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()

            xp.launch_experiment([1, 100])

            results = xp._get_all_items('result')
            # 100 is not an id of the generated tasks, so only 1 result is
            # expected
            self.assertEqual(len(results), 1, msg='the number of results')

    def test_collect_results_full_xp(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:

            # launch experiment
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()

            values = list(xp._schema.values())

            error_msg = ('No result available. Results must be computed '
                         'before running collect_results().')
            with self.assertRaisesRegex(RuntimeError, error_msg):
                xp.collect_results()

            xp.launch_experiment()
            xp.collect_results()

            self.assertTrue(
                (xp.xp_path / 'results.npz').exists(),
                msg='the presence of file')

            file_npz = np.load(str(xp.xp_path / 'results.npz'))
            self.assertTrue('results' in file_npz, msg='the file')

            results = file_npz['results']
            self.assertTrue(
                isinstance(results, np.ndarray),
                msg='if results is nd-array')
            self.assertEqual(results.dtype, np.float64)
            self.assertEqual(
                len(results.shape), len(values) + 1, msg='shape of results')

            for ida, n_ticks in enumerate(results.shape[:-1]):
                self.assertEqual(
                    n_ticks,
                    len(values[ida]),
                    msg='size of dimension {}'.format(ida))

            measure_output = [0.2, 1.6]
            self.assertEqual(results.shape[-1], len(measure_output))

            self.assertTrue(np.all(results[..., 0] == measure_output[0]))
            self.assertTrue(np.all(results[..., 1] == measure_output[1]))

    def test_collect_results_partial_xp(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()

            values = list(xp._schema.values())

            xp.run_task_by_id(10)
            xp.run_task_by_id(20)
            xp.collect_results()

            self.assertTrue(
                (xp.xp_path / 'results.npz').exists(),
                msg='the presence of file')

            file_npz = np.load(str(xp.xp_path / 'results.npz'))
            self.assertTrue(
                'results' in file_npz, msg='if file contains results')

            results = file_npz['results']
            self.assertTrue(
                isinstance(results, np.ndarray),
                msg='if results is nd-array')
            self.assertEqual(results.dtype, np.float64)
            self.assertEqual(
                len(results.shape), len(values) + 1, msg='shape of results')

            for ida, n_ticks in enumerate(results.shape[:-1]):
                self.assertEqual(
                    n_ticks,
                    len(values[ida]),
                    msg='size of dimension {}'.format(ida))

            measure_output = [0.2, 1.6]
            self.assertEqual(results.shape[-1], len(measure_output))

            subarray = results[..., 0]
            self.assertTrue(
                np.all(subarray[~np.isnan(subarray)] == measure_output[0]))
            subarray = results[..., 1]
            self.assertTrue(
                np.all(subarray[~np.isnan(subarray)] == measure_output[1]))

    def test_load_results(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()

            values = list(xp._schema.values())

            xp.run_task_by_id(10)
            xp.run_task_by_id(20)

            error_msg = ('No file with collected results available. '
                         'Results must be computed and collected before '
                         'running load_results().')
            with self.assertRaisesRegex(RuntimeError, error_msg):
                xp.load_results()

            xp.collect_results()

            results, axes_labels, axes_values = xp.load_results()

            self.assertTrue(
                isinstance(results, np.ndarray),
                msg='if results is nd-array')
            self.assertEqual(results.dtype, np.float64)
            self.assertEqual(
                len(results.shape), len(values) + 1, msg='shape of results')

            for ida, n_ticks in enumerate(results.shape[:-1]):
                self.assertEqual(
                    n_ticks,
                    len(values[ida]),
                    msg='size of dimension {}'.format(ida))
                np.testing.assert_array_equal(values[ida], axes_values[ida])

            measure_output = [0.2, 1.6]
            self.assertEqual(results.shape[-1], len(measure_output))

            subarray = results[..., 0]
            self.assertTrue(
                np.all(subarray[~np.isnan(subarray)] == measure_output[0]))
            subarray = results[..., 1]
            self.assertTrue(
                np.all(subarray[~np.isnan(subarray)] == measure_output[1]))

            expected_labels = ('data_a', 'data_b', 'data_seed', 'problem_c',
                               'problem_d', 'problem_e', 'problem_seed',
                               'solver_f', 'solver_seed', 'measure')
            np.testing.assert_array_equal(axes_labels, expected_labels)

            error_msg = ('Unrecognized value for array_type, the valid '
                         'options are "numpy" and "xarray".')
            with self.assertRaisesRegex(ValueError, error_msg):
                _ = xp.load_results('wrong_flag')

            error_msg = (r'Please install the xarray package to '
                         r'run the load_results\(\) method with the '
                         r'array\_type="xarray" option.')
            with self.assertRaisesRegex(ImportError, error_msg):
                with patch.dict(sys.modules, {'xarray': None}):
                    _ = xp.load_results('xarray')

    def test_load_results_as_xarray(self):
        try:
            import xarray
        except ImportError:
            msg = ("As the xarray package is not available, the "
                   "array_type='xarray' option of Experiment.load_results() "
                   "could not be tested.")
            warnings.warn(msg)
            return

        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()

            values = list(xp._schema.values())

            xp.run_task_by_id(10)
            xp.run_task_by_id(20)

            xp.collect_results()

            results = xp.load_results('xarray')

            self.assertTrue(
                isinstance(results, xarray.DataArray),
                msg='if results is nd-array')
            self.assertEqual(results.dtype, np.float64)
            self.assertEqual(
                len(results.shape), len(values) + 1, msg='shape of results')

            for ida, n_ticks in enumerate(results.shape[:-1]):
                self.assertEqual(
                    n_ticks,
                    len(values[ida]),
                    msg='size of dimension {}'.format(ida))
                np.testing.assert_array_equal(
                    values[ida], results.coords[results.dims[ida]])

            measure_output = [0.2, 1.6]
            self.assertEqual(results.shape[-1], len(measure_output))

            subarray = results[..., 0].values
            self.assertTrue(
                np.all(subarray[~np.isnan(subarray)] == measure_output[0]))
            subarray = results[..., 1].values
            self.assertTrue(
                np.all(subarray[~np.isnan(subarray)] == measure_output[1]))

            expected_labels = ('data_a', 'data_b', 'data_seed', 'problem_c',
                               'problem_d', 'problem_e', 'problem_seed',
                               'solver_f', 'solver_seed', 'measure')
            self.assertEqual(results.dims, expected_labels)

    def test_display_status(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            data_params = {'a': [1000, 2, 3], 'b': [4, 5], 'seed': 12}

            xp.add_tasks(data_params, self.problem_params, self.solver_params)
            xp.generate_tasks()

            nb_tasks = self.expected_nb_tasks

            nb_tasks_success = 7
            xp.launch_experiment(range(nb_tasks_success))

            nb_tasks_fail = 3
            xp.launch_experiment(range(nb_tasks - nb_tasks_fail, nb_tasks))

            tmp_stdout = StringIO()
            with contextlib.redirect_stdout(tmp_stdout):
                xp.display_status()
            out = tmp_stdout.getvalue().strip()
            expected_out = 'Number of tasks: {}\n' \
                           'Number of finished tasks: {}\n' \
                           'Number of failed tasks: {}\n' \
                           'Number of pending tasks: {}'
            expected_out = expected_out.format(
                nb_tasks, nb_tasks_success, nb_tasks_fail,
                nb_tasks - nb_tasks_success)
            self.assertEqual(out, expected_out)

            tmp_stdout = StringIO()
            with contextlib.redirect_stdout(tmp_stdout):
                xp.display_status(False)
            out = tmp_stdout.getvalue().strip()
            expected_out = ''
            for task_id in range(nb_tasks_success):
                expected_out += '{:06d}..................OK\n'.format(task_id)
            for task_id in range(nb_tasks_success, nb_tasks - nb_tasks_fail):
                expected_out += '{:06d}......NOT CALCULATED\n'.format(task_id)
            for task_id in range(nb_tasks - nb_tasks_fail, nb_tasks):
                expected_out += '{:06d}...............ERROR\n'.format(task_id)
            expected_out = expected_out[:-1]
            self.assertEqual(out, expected_out)

    def test_get_task_data_by_id(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()
            task_data = xp.get_task_data_by_id(0)

            expected_dict_keys = {'id_task', 'task_params', 'source_data',
                                  'problem_data', 'solution_data',
                                  'solved_data', 'result'}

            expected = {'id_task': 0,
                        'task_params': {
                            'data_params': {'a': 1, 'b': 4, 'seed': 12},
                            'problem_params':
                                {'c': 1, 'd': 5, 'e': 'alpha', 'seed': 12},
                            'solver_params': {'f': 31, 'seed': 12}},
                        'source_data': None,
                        'problem_data': None,
                        'solved_data': None,
                        'solution_data': None,
                        'result': None}

            self.assertEqual(expected_dict_keys, set(task_data.keys()))
            self.assertEqual(expected, task_data)

            xp.run_task_by_id(0)
            task_data = xp.get_task_data_by_id(0)

            expected = {'id_task': 0,
                        'task_params': {
                            'data_params': {'a': 1, 'b': 4, 'seed': 12},
                            'problem_params':
                                {'c': 1, 'd': 5, 'e': 'alpha', 'seed': 12},
                            'solver_params': {'f': 31, 'seed': 12}},
                        'source_data': {'X': np.array([0, 1, 2, 3, 4])},
                        'problem_data': {'X': np.array([11, 12, 13, 14, 15])},
                        'solution_data': {'X': np.array([12, 13, 14, 15, 16])},
                        'solved_data': {'X': np.array([42, 43, 44, 45, 46])},
                        'result': {'snr': 0.2, 'error': 1.6}}

            self.assertEqual(expected_dict_keys, set(task_data.keys()))
            self.assertEqual(expected['id_task'], task_data['id_task'])
            self.assertEqual(expected['task_params'], task_data['task_params'])
            np.testing.assert_equal(expected['source_data']['X'],
                                    task_data['source_data']['X'])
            np.testing.assert_equal(expected['problem_data']['X'],
                                    task_data['problem_data']['X'])
            np.testing.assert_equal(expected['solution_data']['X'],
                                    task_data['solution_data']['X'])
            np.testing.assert_equal(expected['solved_data']['X'],
                                    task_data['solved_data']['X'])
            self.assertEqual(expected['result'], task_data['result'])

            error_msg = 'Unknown task: 100'
            with self.assertRaisesRegex(ValueError, error_msg):
                _ = xp.get_task_data_by_id(100)

    def test_get_task_data_by_params(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name)

            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()
            task_params = {'data_params': {'a': 1, 'b': 4, 'seed': 12},
                           'problem_params':
                               {'c': 1, 'd': 5, 'e': 'alpha', 'seed': 12},
                           'solver_params': {'f': 31, 'seed': 12}}
            task_data = xp.get_task_data_by_params(**task_params)

            expected_dict_keys = {'id_task', 'task_params', 'source_data',
                                  'problem_data', 'solution_data',
                                  'solved_data', 'result'}

            expected = {'id_task': 0,
                        'task_params': task_params,
                        'source_data': None,
                        'problem_data': None,
                        'solution_data': None,
                        'solved_data': None,
                        'result': None}

            self.assertEqual(expected_dict_keys, set(task_data.keys()))
            self.assertEqual(expected, task_data)

            xp.run_task_by_id(0)
            task_data = xp.get_task_data_by_params(**task_params)

            expected = {'id_task': 0,
                        'task_params': task_params,
                        'source_data': {'X': np.array([0, 1, 2, 3, 4])},
                        'problem_data': {'X': np.array([11, 12, 13, 14, 15])},
                        'solution_data': {'X': np.array([12, 13, 14, 15, 16])},
                        'solved_data': {'X': np.array([42, 43, 44, 45, 46])},
                        'result': {'snr': 0.2, 'error': 1.6}}

            self.assertEqual(expected_dict_keys, set(task_data.keys()))
            self.assertEqual(expected['id_task'], task_data['id_task'])
            self.assertEqual(expected['task_params'], task_data['task_params'])
            np.testing.assert_equal(expected['source_data']['X'],
                                    task_data['source_data']['X'])
            np.testing.assert_equal(expected['problem_data']['X'],
                                    task_data['problem_data']['X'])
            np.testing.assert_equal(expected['solution_data']['X'],
                                    task_data['solution_data']['X'])
            np.testing.assert_equal(expected['solved_data']['X'],
                                    task_data['solved_data']['X'])
            self.assertEqual(expected['result'], task_data['result'])

            task_params = {'data_params': {'a': 100, 'b': 4, 'seed': 12},
                           'problem_params':
                               {'c': 1, 'd': 5, 'e': 'alpha', 'seed': 12},
                           'solver_params': {'f': 31, 'seed': 12}}
            error_msg = 'No task exists for the given parameters.'
            with self.assertRaisesRegex(ValueError, error_msg):
                _ = xp.get_task_data_by_params(**task_params)

            error_msg = 'Parameters must be of type dict.'
            with self.assertRaisesRegex(TypeError, error_msg):
                _ = xp.get_task_data_by_params(12, {'c': 1}, {'f': 31})

            task_params = {'data_params': {'b': 4, 'seed': 12},
                           'problem_params':
                               {'c': 1, 'd': 5, 'e': 'alpha', 'seed': 12},
                           'solver_params': {'f': 31, 'seed': 12}}
            error_msg = ('Wrong keys in data_params, the expected keys are: '
                         'a, b, seed.')
            with self.assertRaisesRegex(ValueError, error_msg):
                _ = xp.get_task_data_by_params(**task_params)

            task_params = {'data_params': {'a': 100, 'b': 4, 'seed': 12},
                           'problem_params':
                               {'c': 1, 'd': 5, 'f': 'alpha', 'seed': 12},
                           'solver_params': {'f': 31, 'seed': 12}}
            error_msg = ('Wrong keys in problem_params, the expected keys are:'
                         ' c, d, e, seed.')
            with self.assertRaisesRegex(ValueError, error_msg):
                _ = xp.get_task_data_by_params(**task_params)

            task_params = {'data_params': {'a': 100, 'b': 4, 'seed': 12},
                           'problem_params':
                               {'c': 1, 'd': 5, 'e': 'alpha', 'seed': 12},
                           'solver_params': {'e': 31}}
            error_msg = ('Wrong keys in solver_params, the expected keys are:'
                         ' f, seed.')
            with self.assertRaisesRegex(ValueError, error_msg):
                _ = xp.get_task_data_by_params(**task_params)

    def test_track_tasks(self):
        with tempfile.TemporaryDirectory(dir=str(get_data_path())) as tmpdir:

            xp_name = Path(tmpdir).stem
            Path(tmpdir).rmdir()

            exp = Experiment(name=xp_name,
                             get_data=MockTrackData(),
                             get_problem=MockTrackProblem,
                             get_solver=MockTrackSolver,
                             measure=mock_track_measure)
            exp.reset()
            exp.add_tasks(data_params=MockTrackData.get_params(),
                          problem_params=MockTrackProblem.get_params(),
                          solver_params=MockTrackSolver.get_params())
            exp.generate_tasks()
            exp.launch_experiment()
            exp.collect_results()
            results, axes_labels, axes_values = \
                exp.load_results(array_type='numpy')

            check_tracked_results(results=results,
                                  axes_labels=axes_labels,
                                  axes_values=axes_values)

            exp.add_tasks(
                data_params=MockTrackData.get_additional_params(),
                problem_params=MockTrackProblem.get_additional_params(),
                solver_params=MockTrackSolver.get_additional_params())
            exp.generate_tasks()
            exp.launch_experiment()
            exp.collect_results()
            results, axes_labels, axes_values = \
                exp.load_results(array_type='numpy')
            check_tracked_results(results=results,
                                  axes_labels=axes_labels,
                                  axes_values=axes_values)

    def test_experiment_with_specified_data_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xp_name = Path(tmpdir).stem
            xp = MockExperiment(xp_name, data_path=tmpdir)

            xp.add_tasks(self.data_params, self.problem_params,
                         self.solver_params)
            xp.generate_tasks()
            xp.run_task_by_id(10)

            results = xp._get_all_items('result')
            self.assertTrue(10 in results, msg='the presence of a result')


# Test using a more realistic example based on the yafe tutorial

def get_sine_data(f0, signal_len=1000):
    return {'signal': np.sin(2*np.pi*f0*np.arange(signal_len))}


class SimpleDenoisingProblem:
    def __init__(self, snr_db):
        self.snr_db = snr_db

    def __call__(self, signal):
        random_state = np.random.RandomState(0)
        noise = random_state.randn(*signal.shape)
        observation = signal + 10 ** (-self.snr_db / 20) * noise \
            / np.linalg.norm(noise) * np.linalg.norm(signal)
        problem_data = {'observation': observation}
        solution_data = {'signal': signal}
        return problem_data, solution_data

    def __str__(self):
        return 'SimpleDenoisingProblem(snr_db={})'.format(self.snr_db)


class SmoothingSolver:
    def __init__(self, filter_len):
        self.filter_len = filter_len

    def __call__(self, observation):
        smoothing_filter = np.hamming(self.filter_len)
        smoothing_filter /= np.sum(smoothing_filter)
        return {'reconstruction': np.convolve(observation, smoothing_filter,
                                              mode='same')}

    def __str__(self):
        return 'SmoothingSolver(filter_len={})'.format(self.filter_len)


def measure_full_example(solution_data, solved_data, task_params=None,
                         source_data=None, problem_data=None):
    euclidian_distance = np.linalg.norm(solution_data['signal']
                                        - solved_data['reconstruction'])
    sdr = 20 * np.log10(np.linalg.norm(solution_data['signal'])
                        / euclidian_distance)
    inf_distance = np.linalg.norm(solution_data['signal']
                                  - solved_data['reconstruction'], ord=np.inf)
    return {'sdr': sdr,
            'euclidian_distance': euclidian_distance,
            'inf_distance': inf_distance}


class TestFullExample(unittest.TestCase):

    def setUp(self):
        self.temp_data_path = tempfile.mkdtemp(prefix='yafe_')

    def tearDown(self):
        rmtree(self.temp_data_path)

    def test_full_example(self):
        decimal_precision = 13
        data_params = {'f0': np.arange(0.01, 0.1, 0.01), 'signal_len': [1000]}
        problem_params = {'snr_db': [-10, 0, 30]}
        solver_params = {'filter_len': 2**np.arange(6, step=2)}
        exp = Experiment(name='full_experiment',
                         get_data=get_sine_data,
                         get_problem=SimpleDenoisingProblem,
                         get_solver=SmoothingSolver,
                         measure=measure_full_example,
                         data_path=self.temp_data_path,
                         log_to_file=False,
                         log_to_console=False)
        exp.add_tasks(data_params=data_params,
                      problem_params=problem_params,
                      solver_params=solver_params)
        status = ('Number of tasks: 0\n'
                  'Number of finished tasks: 0\n'
                  'Number of failed tasks: 0\n'
                  'Number of pending tasks: 0\n')
        self.assertEqual(status, exp._get_status(True))
        exp.generate_tasks()
        status = ('Number of tasks: 81\n'
                  'Number of finished tasks: 0\n'
                  'Number of failed tasks: 0\n'
                  'Number of pending tasks: 81\n')
        self.assertEqual(status, exp._get_status(True))
        exp.run_task_by_id(idt=5)
        status = ('Number of tasks: 81\n'
                  'Number of finished tasks: 1\n'
                  'Number of failed tasks: 0\n'
                  'Number of pending tasks: 80\n')
        self.assertEqual(status, exp._get_status(True))
        exp.launch_experiment()
        status = ('Number of tasks: 81\n'
                  'Number of finished tasks: 81\n'
                  'Number of failed tasks: 0\n'
                  'Number of pending tasks: 0\n')
        self.assertEqual(status, exp._get_status(True))
        exp.collect_results()
        results, axes_labels, axes_values = exp.load_results()
        self.assertEqual((9, 1, 3, 3, 3), results.shape)
        self.assertEqual(np.float64, results.dtype)
        self.assertAlmostEqual(10.100157318786618, np.mean(results),
                               places=decimal_precision)
        expected_labels = np.array(
            ['data_f0', 'data_signal_len', 'problem_snr_db',
             'solver_filter_len', 'measure'],
            dtype=object)
        np.testing.assert_array_equal(expected_labels, axes_labels)
        expected_values = np.array(
            [
             np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]),
             np.array([1000]),
             np.array([-10, 0, 30]),
             np.array([1,  4, 16]),
             np.array(['sdr', 'euclidian_distance', 'inf_distance'],
                      dtype=object)
            ],
            dtype=object)
        np.testing.assert_array_almost_equal(
            expected_values[0], axes_values[0], decimal=decimal_precision)
        for ind in range(1, 5):
            np.testing.assert_array_equal(expected_values[ind],
                                          axes_values[ind])
        additional_snr_db = [10, 20]
        additional_filter_len = 2**np.arange(6)
        exp.add_tasks(problem_params={'snr_db': additional_snr_db},
                      data_params=dict(),
                      solver_params={'filter_len': additional_filter_len})
        exp.generate_tasks()
        status = ('Number of tasks: 270\n'
                  'Number of finished tasks: 81\n'
                  'Number of failed tasks: 0\n'
                  'Number of pending tasks: 189\n')
        self.assertEqual(status, exp._get_status(True))
        exp.launch_experiment()
        status = ('Number of tasks: 270\n'
                  'Number of finished tasks: 270\n'
                  'Number of failed tasks: 0\n'
                  'Number of pending tasks: 0\n')
        self.assertEqual(status, exp._get_status(True))
        exp.collect_results()
        results, axes_labels, axes_values = exp.load_results()
        self.assertEqual((9, 1, 5, 6, 3), results.shape)
        self.assertEqual(np.float64, results.dtype)
        self.assertAlmostEqual(8.2145574032230524, np.mean(results),
                               places=decimal_precision)
        np.testing.assert_array_equal(expected_labels, axes_labels)
        expected_values = np.array(
            [
             np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]),
             np.array([1000]),
             np.array([-10, 0, 10, 20, 30]),
             np.array([1, 2, 4, 8, 16, 32]),
             np.array(['sdr', 'euclidian_distance', 'inf_distance'],
                      dtype=object)
            ],
            dtype=object)
        np.testing.assert_array_almost_equal(
            expected_values[0], axes_values[0], decimal=decimal_precision)
        for ind in range(1, 5):
            np.testing.assert_array_equal(expected_values[ind],
                                          axes_values[ind])
