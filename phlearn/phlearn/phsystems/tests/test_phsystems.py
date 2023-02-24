# TODO: Test case when len(S.shape) == 3 and len(R.shape) == 3
# TODO: Have ignored controller for now. I.e., assumed controller = None throughout

# When developing tests, from the top level dir (i.e., phlearn) run
# python3 -m phlearn.phnns.tests.test_phsystems
# to run this as a module, allowing use of relative imports

import numpy as np
from ..pseudo_Hamiltonian_system import PseudoHamiltonianSystem
import torch


N_STATES = 10
N_TIMESTEPS = 100

X_RAND = np.random.rand(N_STATES)
T_AXIS = np.linspace(0,10, N_TIMESTEPS)
EXTERNAL_FORCE = np.random.rand(N_STATES)
DISSIPATION_MATRIX = np.random.rand(N_STATES, N_STATES)
DISSIPATION_MATRIX = DISSIPATION_MATRIX + DISSIPATION_MATRIX.T
STRUCTURE_MATRIX = np.random.rand(N_STATES, N_STATES)
STRUCTURE_MATRIX = STRUCTURE_MATRIX - STRUCTURE_MATRIX.T


def H(x):
    x = x.flatten() if torch.is_tensor(x) else x
    return sum(x[i] ** 2 for i in range(N_STATES))


def dH(x):
    return np.array([2 * x[i] for i in range(N_STATES)])


psh_kwargs = dict(
    nstates=N_STATES,
    hamiltonian=H,
)


def test_R_is_zero_matrix_when_dissipation_matrix_is_None():
    phs = PseudoHamiltonianSystem(**psh_kwargs)
    assert not phs.R(
        np.random.rand(psh_kwargs["nstates"])
    ).any(), "Dissipation_matrix not defaulting to zero matrix"


def test_S_is_canonical_when_structure_matrix_is_None():
    phs = PseudoHamiltonianSystem(**psh_kwargs)
    m = int(N_STATES / 2)
    O, I = np.zeros([m, m]), np.eye(m)
    S = np.block([[O, I], [-I, O]])
    assert np.array_equal(
        S, phs.S(X_RAND)
    ), "strucutre_matrix not defulating to canonical form"


def test_dissipation_matrix_is_returned():
    phs = PseudoHamiltonianSystem(dissipation_matrix=DISSIPATION_MATRIX, **psh_kwargs)
    assert np.array_equal(
        DISSIPATION_MATRIX, phs.R(X_RAND)
    ), "dissipation_matrix not returned"


def test_structure_matrix_is_returned():
    psh_kwargs_loc = psh_kwargs
    n = psh_kwargs["nstates"]
    phs = PseudoHamiltonianSystem(structure_matrix=STRUCTURE_MATRIX, **psh_kwargs_loc)
    assert np.array_equal(
        STRUCTURE_MATRIX, phs.S(X_RAND)
    ), "structure_matrix not returned"


def test_dH_calculated_correctly():
    phs = PseudoHamiltonianSystem(**psh_kwargs)
    assert np.allclose(phs.dH(X_RAND), 2 * X_RAND), "grad_H computed incorrectly"


def test_x_dot():
    phs = PseudoHamiltonianSystem(
        dissipation_matrix=lambda x: DISSIPATION_MATRIX,
        structure_matrix=lambda x: STRUCTURE_MATRIX,
        external_forces=lambda x, t: EXTERNAL_FORCE,
        **psh_kwargs
    )
    x_dot = dH(X_RAND) @ (STRUCTURE_MATRIX.T - DISSIPATION_MATRIX.T) + EXTERNAL_FORCE
    assert np.allclose(phs.x_dot(X_RAND, T_AXIS), x_dot), "x_dot() returns incorrect ODE"


def test_sample_trajectory_on_const_ode():
    phs = PseudoHamiltonianSystem(structure_matrix=0 * STRUCTURE_MATRIX, **psh_kwargs)
    x, _, _, _ = phs.sample_trajectory(t=[0, 1], x0=X_RAND, noise_std=0)
    assert np.allclose(
        x[-1], X_RAND
    ), "Solution of a constant ODE does not remain constant"
