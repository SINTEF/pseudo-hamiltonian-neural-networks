import numpy as np

from .pseudo_hamiltonian_system import PseudoHamiltonianSystem

__all__ = ['MassSpringDamperSystem', 'init_msdsystem',
           'initial_condition_radial']

class MassSpringDamperSystem(PseudoHamiltonianSystem):
    """
    Implements a general forced and damped mass-spring system as a
    pseudo-Hamiltonian formulation::

           .
        |  q |     |  0     1 |                   |      0     |
        |  . |  =  |          | * grad[H(q, p)] + |            |
        |  p |     | -1    -c |                   | f(q, p, t) |

    where q is the position, p the momentum and c the damping coefficient.

    Parameters
    ----------
    mass : number, default 1.0
        Scalar mass
    spring_constant : number, default 1.0
        Scalar spring coefficient
    damping : number, default 0.3
        Scalar damping coefficient. Corresponds to c.
    kwargs  : any, optional
        Keyword arguments that are passed to PseudoHamiltonianSystem constructor.

    """

    def __init__(self, mass=1.0, spring_constant=1.0, damping=0.3, **kwargs):
        R = np.diag([0, damping])
        M = np.diag([spring_constant / 2, 1 / (2 * mass)])

        def hamiltonian(x):
            return x.T @ M @ x

        def hamiltonian_grad(x):
            return 2 * M @ x

        super().__init__(
            nstates=2,
            hamiltonian=hamiltonian,
            grad_hamiltonian=hamiltonian_grad,
            dissipation_matrix=R,
            **kwargs
        )


def init_msdsystem():
    """
    Initialize a standard example of a damped mass-spring system affected by a
    sine force.

    Returns
    -------
    MassSpringDamperSystem

    """
    f0 = 1.0
    omega = 3

    def F(x, t):
        return (f0 * np.sin(omega * t)).reshape(x[..., 1:].shape) * np.array([0, 1])

    return MassSpringDamperSystem(
        external_forces=F, init_sampler=initial_condition_radial(1, 4.5)
    )


def initial_condition_radial(r_min, r_max):
    """
    Creates an initial condition sampler that draws samples uniformly
    from the disk r_min <= x^Tx < r_max.

    Returns
    -------
    callable
        Function taking a numpy random generator and returning an
        initial state of size 2.

    """

    def sampler(rng):
        r = (r_max - r_min) * np.sqrt(rng.uniform(size=1)) + r_min
        theta = 2.0 * np.pi * rng.uniform(size=1)
        q = r * np.cos(theta)
        p = r * np.sin(theta)
        return np.array([q, p]).flatten()

    return sampler
