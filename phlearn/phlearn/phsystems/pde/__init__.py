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

from .pseudo_hamiltonian_pde_system import *
from .kdv_system import *
from .bbm_system import *
from .perona_malik_system import *
from .heat_system import *
from .allen_cahn_system import *
from .cahn_hilliard_system import *
