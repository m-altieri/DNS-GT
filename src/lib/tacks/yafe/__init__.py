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
"""Experiments as combinations of data, problems and solvers.

.. moduleauthor:: Ronan Hamon
.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Florent Jaillet
"""
from .base import Experiment, InvalidTask

__all__ = ['Experiment', 'InvalidTask']
__version__ = "1.0.3"
