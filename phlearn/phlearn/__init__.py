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
try:
  from . import control
except ModuleNotFoundError:
  # from warnings import warn
  # warn("Loading phlearn without control module")
  pass 
from . import phnns
from . import phsystems
from . import utils
