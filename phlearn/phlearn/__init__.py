"""
phlearn is a package for simulating, controlling and learning
dynamical systems in general, and pseudo-Hamiltonian systems in particular.
phlearn contains three subpackages:

- :py:mod:`~.phsystems` with tools for simulation

- :py:mod:`~.phnns` with tools for constructing and training
  pseudo-Hamiltonian neural networks

- :py:mod:`~.control` for control functionality

- :py:mod:`~.utils` for convenience functions.

"""

from . import control
from . import phnns
from . import phsystems
from . import utils
