'''
The phsystems subpackage is divided into two further subpackages, one for ODEs and one for PDEs.
These implement pseudo-Hamiltonian systems and numerical integration of these to obtain data.
'''

from . import ode
from .ode import *
from . import pde
from .pde import *

__all__ = ode.__all__.copy()
__all__ += pde.__all__.copy()