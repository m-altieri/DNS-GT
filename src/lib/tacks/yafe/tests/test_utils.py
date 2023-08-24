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
"""Test of the module :module:`yafe.utils`

.. moduleauthor:: Ronan Hamon
.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Florent Jaillet
"""
from contextlib import redirect_stdout
from io import StringIO
import json
import logging
import os
from pathlib import Path
from shutil import rmtree
import runpy
import stat
import tempfile
import unittest

import numpy as np

from yafe.utils import get_logger, ConfigParser, generate_oar_script
from .test_base import create_temp_test_config, get_data_path, MockExperiment


class TestGetLogger(unittest.TestCase):

    def setUp(self):
        self._temp_dir = create_temp_test_config()

    def tearDown(self):
        rmtree(self._temp_dir)

    def test_get_logger(self):

        logger_path = ConfigParser().get_path('LOGGER', 'PATH', False)

        # no saving to file
        logger = get_logger(name='test', to_file=False, to_console=False)

        logger.info('test')
        self.assertEqual(logging.getLevelName(logger.level), 'DEBUG')
        self.assertEqual(len(logger.handlers), 0)

        # saving to file with path from config file
        self.assertFalse(logger_path.exists())

        logger = get_logger(name='test')

        self.assertTrue(logger_path.exists())

        self.assertEqual(logging.getLevelName(logger.level), 'DEBUG')
        self.assertEqual(len(logger.handlers), 2)

        self.assertEqual(
            logging.getLevelName(logger.handlers[0].level), 'DEBUG')
        self.assertEqual(logger.handlers[0].stream.name, str(logger_path))

        self.assertEqual(
            logging.getLevelName(logger.handlers[1].level), 'INFO')

        # saving to file with specified path
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        tmp_file.close()
        try:
            logger = get_logger(name='test',
                                logger_path=tmp_file.name)

            self.assertEqual(logging.getLevelName(logger.level), 'DEBUG')
            self.assertEqual(len(logger.handlers), 2)

            self.assertEqual(
                logging.getLevelName(logger.handlers[0].level), 'DEBUG')
            self.assertEqual(logger.handlers[0].stream.name, tmp_file.name)

            self.assertEqual(
                logging.getLevelName(logger.handlers[1].level), 'INFO')
        finally:
            os.remove(tmp_file.name)


class TestConfigParser(unittest.TestCase):

    def setUp(self):
        self._temp_dir = create_temp_test_config()

    def tearDown(self):
        rmtree(self._temp_dir)

    def test_generate_config_and_get_path(self):
        original_config_path = ConfigParser._config_path
        temp_dir = tempfile.mkdtemp(prefix='yafe_')
        try:
            ConfigParser._config_path = temp_dir / Path('yafe.log')
            ConfigParser().generate_config()

            # Check that the config file has been created
            self.assertTrue(ConfigParser._config_path.exists())

            config = ConfigParser()
            # Check that the USER section exists with an empty data_path
            user_data_path = config.get_path('USER', 'data_path')
            self.assertIs(user_data_path, None)
            # Check that the LOGGER section exists with an empty path
            logger_path = config.get_path('LOGGER', 'path')
            self.assertIs(logger_path, None)

            # Further check that get_path() works as expected
            config['TEST_SECTION'] = {'good_path': temp_dir,
                                      'bad_path': '/non_existing_path'}
            with ConfigParser._config_path.open('w') as configfile:
                config.write(configfile)
            good_path = config.get_path('TEST_SECTION', 'good_path')
            self.assertEqual(temp_dir, str(good_path))
            with self.assertRaises(IOError):
                bad_path = config.get_path('TEST_SECTION', 'bad_path')

        finally:
            ConfigParser._config_path = original_config_path
            rmtree(temp_dir)


class TestGenerateOarScript(unittest.TestCase):

    def tearDown(self):
        rmtree(tempfile.gettempdir() / Path('yafe_temp_test_data'))

    def test_generate_oar_script(self):
        script_file_path = (Path(__file__).parent
                            / 'script_for_generate_oar_script.py')
        xp_var_name = 'experiment'
        temp_dir = tempfile.gettempdir()
        xp_path = (temp_dir / Path('yafe_temp_test_data')
                   / 'test_generate_oar_script')

        runpy.run_path(str(script_file_path), run_name='__main__')

        tmp_stdout = StringIO()
        with redirect_stdout(tmp_stdout):
            generate_oar_script(script_file_path, xp_var_name)

        out = tmp_stdout.getvalue().strip()
        expected_out = ('oarsub -S {}/yafe_temp_test_data/test_generate_oar'
                        '_script/script_oar.sh'.format(temp_dir))
        self.assertEqual(out, expected_out)

        task_file_path = xp_path / "listoftasks.txt"
        self.assertTrue(task_file_path.exists())
        task_data = np.loadtxt(task_file_path, dtype=np.int64)
        np.testing.assert_array_equal(np.arange(81), task_data)

        tasks = [1, 3]
        generate_oar_script(script_file_path, xp_var_name, tasks)

        task_file_path = xp_path / "listoftasks.txt"
        self.assertTrue(task_file_path.exists())
        task_data = np.loadtxt(task_file_path, dtype=np.int64)
        np.testing.assert_array_equal(tasks, task_data)

        script_path = xp_path / "script_oar.sh"
        self.assertTrue(script_path.exists())
        # Check that the script is executable
        self.assertTrue(os.stat(script_path).st_mode & stat.S_IXUSR)
        # Check the content of the script
        with script_path.open() as script_file:
            script_content = script_file.readlines()
        self.assertEqual(len(script_content), 13)
        expected_content = [
            '#!/bin/sh\n',
            '#OAR -n test_generate_oar_script\n',
            '#OAR --array-param-file {}/yafe_temp_test_data/'
            'test_generate_oar_script/listoftasks.txt\n'.format(temp_dir),
            '#OAR -O {}/yafe_temp_test_data/'
            'test_generate_oar_script/%jobid%.out\n'.format(temp_dir),
            '#OAR -E {}/yafe_temp_test_data/'
            'test_generate_oar_script/%jobid%.err\n'.format(temp_dir),
            '#OAR -l walltime=02:00:00\n',
            '#OAR -p gpu IS NULL\n',
            'echo "OAR_JOB_ID: $OAR_JOB_ID"\n',
            'echo "OAR_ARRAY_ID: $OAR_ARRAY_ID"\n',
            'echo "OAR_ARRAY_INDEX: $OAR_ARRAY_INDEX"\n',
            'echo "Running experiment.launch_experiment(task_ids=\'$1\')"\n',
            'python -c "import sys; sys.path.append(\'{}\'); '
            'from script_for_generate_oar_script import experiment; '
            'experiment.launch_experiment(task_ids=\'$1\')"\n'.format(
                Path(__file__).parent),
            'exit $?']
        self.assertEqual(script_content, expected_content)

        activate_env_command = 'source activate test_env'
        generate_oar_script(script_file_path, xp_var_name,
                            activate_env_command=activate_env_command)
        self.assertTrue(script_path.exists())
        with script_path.open() as script_file:
            script_content = script_file.readlines()
        self.assertEqual(len(script_content), 14)
        expected_content.insert(10, '{}\n'.format(activate_env_command))
        self.assertEqual(script_content, expected_content)

        generate_oar_script(script_file_path, xp_var_name,
                            use_gpu=True)
        self.assertTrue(script_path.exists())
        with script_path.open() as script_file:
            script_content = script_file.readlines()
        self.assertEqual(len(script_content), 13)
        expected_content.pop(10)
        expected_content[6] = '#OAR -p gpu IS NOT NULL\n'
        self.assertEqual(script_content, expected_content)
