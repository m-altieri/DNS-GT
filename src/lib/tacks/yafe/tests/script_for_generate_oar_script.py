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
"""Example script to test yafe.utils.generate_oar_script()

.. moduleauthor:: Ronan Hamon
.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Florent Jaillet
"""
from pathlib import Path
import tempfile

import numpy as np

from yafe import Experiment


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


def measure(solution_data, solved_data, task_params=None, source_data=None,
            problem_data=None):
    euclidian_distance = np.linalg.norm(solution_data['signal']
                                        - solved_data['reconstruction'])
    sdr = 20 * np.log10(np.linalg.norm(solution_data['signal'])
                        / euclidian_distance)
    inf_distance = np.linalg.norm(solution_data['signal']
                                  - solved_data['reconstruction'], ord=np.inf)
    return {'sdr': sdr,
            'euclidian_distance': euclidian_distance,
            'inf_distance': inf_distance}


def create_tasks(exp):
    data_params = {'f0': np.arange(0.01, 0.1, 0.01), 'signal_len': [1000]}
    problem_params = {'snr_db': [-10, 0, 30]}
    solver_params = {'filter_len': 2**np.arange(6, step=2)}
    exp.add_tasks(data_params=data_params,
                  problem_params=problem_params,
                  solver_params=solver_params)
    exp.generate_tasks()


temp_data_path = tempfile.gettempdir() / Path('yafe_temp_test_data')
if not temp_data_path.exists():
    temp_data_path.mkdir()


experiment = Experiment(name='test_generate_oar_script',
                        get_data=get_sine_data,
                        get_problem=SimpleDenoisingProblem,
                        get_solver=SmoothingSolver,
                        measure=measure,
                        data_path=temp_data_path,
                        log_to_file=False,
                        log_to_console=False)


if __name__ == '__main__':
    create_tasks(experiment)
