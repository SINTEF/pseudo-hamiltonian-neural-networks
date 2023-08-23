"""
The phsystems.pde subpackage implements pseudo-Hamiltonian PDE systems for
discretization in space and integration in time.

Classes present in phlearn.phsystems.pde
-----------------------------------------------

    :py:class:`~.pseudo_hamiltonian_pde_system.PseudoHamiltonianPDESystem`

    :py:class:`~.kdv_system.KdVSystem`

    :py:class:`~.bbm_system.BBMSystem`

    :py:class:`~.perona_malik_system.PeronaMalikSystem`

    :py:class:`~.cahn_hilliard_system.CahnHilliardSystem`

    :py:class:`~.heat_system.HeatSystem`

    :py:class:`~.allen_cahn_system.AllenCahnSystem`


Functions present in phlearn.phsystems.pde
-----------------------------------------------

    :py:func:`~.pseudo_hamiltonian_pde_system.zero_force`

    :py:func:`~.kdv_system.initial_condition_kdv`

    :py:func:`~.bbm_system.initial_condition_bbm`

    :py:func:`~.perona_malik_system.initial_condition_ac`

    :py:func:`~.cahn_hilliard_system.initial_condition_pm`

    :py:func:`~.heat_system.initial_condition_heat`
        
    :py:func:`~.allen_cahn_system.initial_condition_ac`

"""

from . import pseudo_hamiltonian_pde_system
from .pseudo_hamiltonian_pde_system import *
from . import kdv_system
from .kdv_system import *
from . import bbm_system
from .bbm_system import *
from . import perona_malik_system
from .perona_malik_system import *
from . import heat_system
from .heat_system import *
from . import allen_cahn_system
from .allen_cahn_system import *
from . import cahn_hilliard_system
from .cahn_hilliard_system import *

__all__ = pseudo_hamiltonian_pde_system.__all__.copy()
__all__ += kdv_system.__all__.copy()
__all__ += bbm_system.__all__.copy()
__all__ += perona_malik_system.__all__.copy()
__all__ += heat_system.__all__.copy()
__all__ += allen_cahn_system.__all__.copy()
__all__ += cahn_hilliard_system.__all__.copy()
