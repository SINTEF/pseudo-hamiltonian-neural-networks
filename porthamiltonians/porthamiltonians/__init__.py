"""
porthamiltonians is a package for simulating, controlling and learning
dynamical systems in general, and port-Hamiltonian systems in particular.
porthamiltonians contains three subpackages:

- :py:mod:`~.phsystems` with tools for simulation

- :py:mod:`~.phnns` with tools for constructing and training
  port-Hamiltoninan neural networks

- :py:mod:`~.control` for control functionality

- :py:mod:`~.utils` for convenience functions.

"""

from . import control
from . import phnns
from . import phsystems
from . import utils
