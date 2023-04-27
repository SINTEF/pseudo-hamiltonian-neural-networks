"""
The phnns subpackage implements neural networks and functionality for
learning dynamic systems and pseudo-Hamiltonian systems.

Classes present in phlearn.phsystems
-----------------------------------------------

    :py:class:`~.dynamic_system_neural_network.DynamicSystemNN`

    :py:class:`~.pseudo_hamiltonian_neural_network.PseudoHamiltonianNN`

    :py:class:`~.models.BaseNN`

    :py:class:`~.models.BaselineNN`

    :py:class:`~.models.HamiltonianNN`

    :py:class:`~.models.ExternalForcesNN`

    :py:class:`~.models.BaselineSplitNN`

    :py:class:`~.models.R_NN`

    :py:class:`~.models.R_estimator`

    :py:class:`~.train_utils.EarlyStopping`



Functions present in phlearn.phsystems
-----------------------------------------------

    :py:func:`~.pseudo_hamiltonian_neural_network.load_phnn_model`

    :py:func:`~.pseudo_hamiltonian_neural_network.store_phnn_model`

    :py:func:`~.models.load_baseline_model`

    :py:func:`~.models.store_baseline_model`

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

from . import dynamic_system_neural_network
from .dynamic_system_neural_network import *
from . import pseudo_hamiltonian_neural_network
from .pseudo_hamiltonian_neural_network import *
from . import conservative_dissipative_neural_network
from .conservative_dissipative_neural_network import *
from . import models
from .models import *
from . import train_utils
from .train_utils import *

__all__ = dynamic_system_neural_network.__all__.copy()
__all__ += pseudo_hamiltonian_neural_network.__all__.copy()
__all__ += conservative_dissipative_neural_network.__all__.copy()
__all__ += models.__all__.copy()
__all__ += train_utils.__all__.copy()
