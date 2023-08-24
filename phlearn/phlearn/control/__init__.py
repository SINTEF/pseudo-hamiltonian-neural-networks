"""
The control subpackage implements PID and MPC controllers for
pseudo-Hamiltonian systems and pseudo-Hamiltonian neural networks.

Controller classes
---------------------------------------------------

    :py:class:`~.pid.PIDController`

    :py:class:`~.mpc.PseudoHamiltonianMPC`

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
from . import phcontroller
from .phcontroller import *
from . import casadiNN
from .casadiNN import *
from . import casadiPH
from .casadiPH import *

__all__ = mpc.__all__.copy()
__all__ += pid.__all__.copy()
__all__ += phcontroller.__all__.copy()
__all__ += casadiNN.__all__.copy()
__all__ += casadiPH.__all__.copy()