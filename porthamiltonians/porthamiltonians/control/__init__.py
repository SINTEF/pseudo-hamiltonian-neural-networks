"""
The control subpackage implements PID and MPC controllers for
port-Hamiltonian systems and port-Hamiltonian neural networks.

Controller classes
---------------------------------------------------

    :py:class:`~.pid.PIDController`

    :py:class:`~.mpc.PortHamiltonianMPC`

Reference classes
-----------------
    :py:class:`~.pid.Reference`

    :py:class:`~.pid.ConstantReference`

    :py:class:`~.pid.StepReference`

    :py:class:`~.pid.PoissonStepReference`

    :py:class:`~.pid.FixedReference`

"""

from . import reference
from .reference import *
from . import pid
from .pid import *
from . import mpc
from .mpc import *

__all__ = reference.__all__.copy()
__all__ += mpc.__all__.copy()
__all__ += pid.__all__.copy()
