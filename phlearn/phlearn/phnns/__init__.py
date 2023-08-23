"""
The phnns subpackage implements neural networks and functionality for
learning dynamic systems and pseudo-Hamiltonian systems.

Classes present in phlearn.phnns
-----------------------------------------------

    :py:class:`~.dynamic_system_neural_network.DynamicSystemNN`

    :py:class:`~.pseudo_hamiltonian_neural_network.PseudoHamiltonianNN`

    :py:class:`~.pseudo_hamiltonian_pde_neural_network.PseudoHamiltonianPDENN`

    :py:class:`~.ode_models.BaseNN`

    :py:class:`~.ode_models.BaselineNN`

    :py:class:`~.ode_models.HamiltonianNN`

    :py:class:`~.ode_models.ExternalForcesNN`

    :py:class:`~.ode_models.BaselineSplitNN`

    :py:class:`~.ode_models.R_NN`

    :py:class:`~.ode_models.R_estimator`

    :py:class:`~.pde_models.CentralPadding`

    :py:class:`~.pde_models.ForwardPadding`

    :py:class:`~.pde_models.Summation`

    :py:class:`~.pde_models.PDEBaseNN`

    :py:class:`~.pde_models.PDEBaseLineNN`

    :py:class:`~.pde_models.PDEIntegralNN`

    :py:class:`~.pde_models.PDEExternalForcesNN`

    :py:class:`~.pde_models.PDEBaselineSplitNN`

    :py:class:`~.pde_models.A_estimator`

    :py:class:`~.pde_models.S_estimator`

    :py:class:`~.train_utils.EarlyStopping`



Functions present in phlearn.phnns
-----------------------------------------------

    :py:func:`~.pseudo_hamiltonian_neural_network.load_phnn_model`

    :py:func:`~.pseudo_hamiltonian_neural_network.store_phnn_model`

    :py:func:`~.pseudo_hamiltonian_pde_neural_network.load_cdnn_model`

    :py:func:`~.pseudo_hamiltonian_pde_neural_network.store_cdnn_model`

    :py:func:`~.dynamic_system_neural_network.load_baseline_model`

    :py:func:`~.dynamic_system_neural_network.store_baseline_model`

    :py:func:`~.train_utils.generate_dataset`

    :py:func:`~.train_utils.train`

    :py:func:`~.train_utils.compute_validation_loss`

    :py:func:`~.train_utils.batch_data`

    :py:func:`~.train_utils.train_one_epoch`

    :py:func:`~.train_utils.l1_loss_pHnn`

    :py:func:`~.train_utils.npoints_to_ntrajectories_tsample`

    :py:func:`~.train_utils.load_dynamic_system_model`

    :py:func:`~.train_utils.store_dynamic_system_model`

"""

from .dynamic_system_neural_network import *
from .pseudo_hamiltonian_neural_network import *
from .pseudo_hamiltonian_pde_neural_network import *
from .ode_models import *
from .pde_models import *
from .train_utils import *

__all__ = dynamic_system_neural_network.__all__.copy()
__all__ += pseudo_hamiltonian_neural_network.__all__.copy()
__all__ += pseudo_hamiltonian_pde_neural_network.__all__.copy()
__all__ += ode_models.__all__.copy()
__all__ += pde_models.__all__.copy()
__all__ += train_utils.__all__.copy()