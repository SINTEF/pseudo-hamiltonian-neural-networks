import numpy as np
from scipy.sparse import spdiags

from .pseudo_hamiltonian_pde_system import PseudoHamiltonianPDESystem

__all__ = ['CahnHilliardSystem', 'initial_condition_ch']

class CahnHilliardSystem(PseudoHamiltonianPDESystem):
    """
    Implements a discretization of the Cahn-Hilliard equation  with an
    optional external force, as described by::

        u_t - (nu u + alpha u^3 + mu u_xx)_xx = f(u,t,x)

    on the pseudo-Hamiltonian formulation::

        du/dt = d^2/dx^2 grad[V(u)] + F(u,t,x)

    where u is a vector of the system states at the spatial points given by
    x, and t is time.

    The system is by default integrated in time using the implicit midpoint method. 
    The line 'self.sample_trajectory = self.sample_trajectory_midpoint' can be 
    commented out to instead use the RK45 method of scipy.

    Parameters
    ----------
    x : nparray, default np.linspace(0, 1.0 - 1/100, 100)
        The spatial discretization points.
    nu : number, default -1.0
        The parameter nu in the Cahn-Hilliard equation.
    alpha : number, default 1.0
        The parameter alpha in the Cahn-Hilliard equation.
    mu : number, default -0.001
        The parameter mu in the Cahn-Hilliard equation.
    init_sampler : callable, default None
        Function for sampling initial conditions. Callable taking
        a numpy random generator as input and returning an ndarray
        of shape same as x with initial conditions for the system.
        This sampler is used when calling
        CahnHilliardSystem.sample_trajectory if no initial
        condition is provided.
    kwargs : any, optional
        Keyword arguments that are passed to PseudoHamiltonianPDESystem
        constructor.

    """

    def __init__(
        self,
        x=np.linspace(0, 1.0 - 1 / 100, 100),
        nu=-1.0,
        alpha=1.0,
        mu=-0.001,
        init_sampler=None,
        **kwargs
    ):
        M = x.size
        dx = x[-1] / (M - 1)
        e = np.ones(M)
        # Forward difference matrix:
        Dp = (
            1 / dx * spdiags([e, -e, e], np.array([-M + 1, 0, 1]), M, M).toarray()
        )  
        # Central difference matrix:
        D1 = (
            0.5
            / dx
            * spdiags([e, -e, e, -e], np.array([-M + 1, -1, 1, M - 1]), M, M).toarray()
        )  
        # 2nd order central difference matrix:
        D2 = (
            1
            / dx**2
            * spdiags(
                [e, e, -2 * e, e, e], np.array([-M + 1, -1, 0, 1, M - 1]), M, M
            ).toarray()
        ) 
        I = np.eye(M)
        skewsymmetric_matrix = D1
        dissipation_matrix = -D2
        self.x = x
        self.M = M
        self.D2 = D2
        self.sample_trajectory = self.sample_trajectory_midpoint

        def dissintegral(u):
            return (
                1
                / 2
                * np.sum(
                    nu * u**2
                    + 1 / 2 * alpha * u**4
                    - mu * (np.matmul(Dp, u.T) ** 2).T,
                    axis=1,
                )
            )

        def dissintegral_grad(u):
            return nu * u + alpha * u**3 + mu * u @ D2

        def ham_hessian(u):
            return np.zeros_like(D1)

        def dissintegral_hessian(u):
            return nu * I + 3 * alpha * np.diag(u**2) + mu * D2

        if init_sampler is None:
            init_sampler = initial_condition_ch(x)

        super().__init__(
            nstates=M,
            skewsymmetric_matrix=skewsymmetric_matrix,
            dissipation_matrix=dissipation_matrix,
            dissintegral=dissintegral,
            grad_dissintegral=dissintegral_grad,
            hess_hamiltonian=ham_hessian,
            hess_dissintegral=dissintegral_hessian,
            init_sampler=init_sampler,
            **kwargs
        )

        self.skewsymmetric_matrix_flat = 0.5 / dx * np.array([[[-1, 0, 1]]])
        self.dissipation_matrix_flat = -1 / dx**2 * np.array([[[1, -2, 1]]])


def initial_condition_ch(x=np.linspace(0, 1.0 - 1 / 100, 100)):
    """
    Creates an initial condition sampler for the Cahn-Hilliard eqation.

    Parameters
    ----------
    x : numpy.ndarray, optional
        Spatial grid on which to create the initial conditions. The default is 
        an equidistant grid between 0 and .99 with step size 0.01.

    Returns
    -------
    callable
        A function that takes a numpy random generator as input and returns an
        initial state on the spatial grid x.
    """

    M = x.size
    P = (x[-1] - x[0]) * M / (M - 1)

    def sampler(rng):
        a1, a2 = rng.uniform(0.0, 0.05, 2)
        a3, a4 = rng.uniform(0.0, 0.2, 2)
        k1, k2, k3, k4 = rng.integers(1, 6, 4)
        u0 = 0
        u0 += a1 * np.cos(2 * k1 * np.pi / P * x)
        u0 += a2 * np.cos(2 * k2 * np.pi / P * x)
        u0 += a3 * np.sin(2 * k3 * np.pi / P * x)
        u0 += a4 * np.sin(2 * k4 * np.pi / P * x)
        return u0

    return sampler
