import numpy as np
from ..dynamic_system_neural_network import DynamicSystemNN
from ..ode_models import BaseNN


def test_initial_condition_sampler():
    n, m = 5, 10
    NN = DynamicSystemNN(n)
    init_cond = NN._initial_condition_sampler(m)
    assert tuple(init_cond.size()) == (m, n)

nstates=3
noutputs=2
hidden_dim=2
bnn = BaseNN(
    nstates=nstates,
    noutputs=noutputs,
    hidden_dim=hidden_dim,
    timedependent=True,
    statedependent=True,
)


# When developing tests, from the top level dir phlearn/ run
# python3 -m phlearn.phnns.tests.test_phnns
# to run this as a module, allowing use of relative imports
