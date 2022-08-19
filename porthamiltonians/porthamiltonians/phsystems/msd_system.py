
import numpy as np

from .port_Hamiltonian_system import PortHamiltonianSystem

__all__ = ['MassSpringDamperSystem', 'init_msdsystem',
           'initial_condition_radial']


class MassSpringDamperSystem(PortHamiltonianSystem):
    """
    Implements a general forced damped mass spring damper as a
    port-Hamiltonian formulation::

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
        Keyword arguments that are passed to PortHamiltonianSystem constructor.

    """

    def __init__(self, mass=1.0, spring_constant=1.0, damping=0.3, **kwargs):
        R = np.array([[0, 0], [0, damping]])

        def ham(x):
            return np.dot(x**2, np.array([spring_constant / 2, 1/(2*mass)]))

        def ham_grad(x):
            return np.matmul(x, np.diag([spring_constant, 1/mass]))

        super().__init__(nstates=2, hamiltonian=ham, grad_hamiltonian=ham_grad,
                         dissipation_matrix=R, **kwargs)


def init_msdsystem():
    """
    Initialize a standard mass spring damper system affected by a
    sine force.

    Returns
    -------
    MassSpringDamperSystem

    """
    f0 = 1.0
    omega = 3

    def F(x, t):
        return (f0*np.sin(omega*t)).reshape(x[..., 1:].shape)*np.array([0, 1])

    return MassSpringDamperSystem(
        external_port=F, init_sampler=initial_condition_radial(1, 4.5))


def initial_condition_radial(r_min, r_max):
    """
    Create an initial consition sampler that draws samples uniformly
    from the disk r_min <= x^Tx < r_max.

    Returns
    -------
    callable
        Function taking a numpy random generator and returning an
        initial state of size 2.

    """
    def sampler(rng):
        r = (r_max - r_min) * np.sqrt(rng.uniform(size=1)) + r_min
        theta = 2.*np.pi * rng.uniform(size=1)
        q = r * np.cos(theta)
        p = r * np.sin(theta)
        return np.array([q, p]).flatten()

    return sampler
