"""
The phsystems subpackage implements pseudo-Hamiltonian systems for
simulation and control.

Classes present in phlearn.phsystems
-----------------------------------------------

    :py:class:`~.pseudo_hamiltonian_system.PseudoHamiltonianSystem`

    :py:class:`~.tank_system.TankSystem`

    :py:class:`~.msd_system.MassSpringDamperSystem`


Functions present in phlearn.phsystems
-----------------------------------------------

    :py:func:`~.pseudo_hamiltonian_system.zero_force`

    :py:func:`~.tank_system.init_tanksystem`

    :py:func:`~.tank_system.init_tanksystem_leaky`

    :py:func:`~.msd_system.init_msdsystem`

    :py:func:`~.msd_system.initial_condition_radial`

"""

from . import pseudo_Hamiltonian_system
from .pseudo_Hamiltonian_system import *
from . import conservative_dissipative_system
from .conservative_dissipative_system import *
from . import tank_system
from .tank_system import *
from . import msd_system
from .msd_system import *
from . import kdv_system
from .kdv_system import *
from . import burgers_system
from .burgers_system import *
from . import bbm_system
from .bbm_system import *
from . import perona_malik_system
from .perona_malik_system import *
from . import heat_system
from .heat_system import *
from . import phase_system
from .phase_system import *
from . import allen_cahn_system
from .allen_cahn_system import *
from . import cahn_hilliard_system
from .cahn_hilliard_system import *

__all__ = pseudo_Hamiltonian_system.__all__.copy()
__all__ += conservative_dissipative_system.__all__.copy()
__all__ += tank_system.__all__.copy()
__all__ += msd_system.__all__.copy()
__all__ += kdv_system.__all__.copy()
__all__ += burgers_system.__all__.copy()
__all__ += bbm_system.__all__.copy()
__all__ += perona_malik_system.__all__.copy()
__all__ += heat_system.__all__.copy()
__all__ += phase_system.__all__.copy()
__all__ += allen_cahn_system.__all__.copy()
__all__ += cahn_hilliard_system.__all__.copy()
