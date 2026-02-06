"""Hardware runners for QAOA on IBM Quantum."""

from .standard_penalty import main as run_standard_penalty_hardware
from .dicke_xy import main as run_dicke_xy_hardware
from .dicke_xy_multistart import main as run_dicke_xy_multistart_hardware

__all__ = [
    'run_standard_penalty_hardware',
    'run_dicke_xy_hardware',
    'run_dicke_xy_multistart_hardware'
]
