# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================
import sys

sys.path.append(".")
sys.path.append("./plots")
from .figure import figure_tensor, blittable_figure_tensor
from .create import create_figure, create_figures

# import plots

# Needs to be last line
__version__ = "1.0.2"
