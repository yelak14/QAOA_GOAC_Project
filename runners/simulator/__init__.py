"""Simulator runners for QAOA."""

from .standard_penalty import main as run_standard_penalty
from .dicke_xy import main as run_dicke_xy
from .dicke_xy_multistart import main as run_dicke_xy_multistart

__all__ = [
    'run_standard_penalty',
    'run_dicke_xy',
    'run_dicke_xy_multistart'
]
