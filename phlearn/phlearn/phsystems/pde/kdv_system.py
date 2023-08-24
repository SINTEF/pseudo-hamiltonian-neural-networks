import numpy as np
from scipy.sparse import spdiags

from .pseudo_hamiltonian_pde_system import PseudoHamiltonianPDESystem

__all__ = ['KdVSystem', 'initial_condition_kdv']

class KdVSystem(PseudoHamiltonianPDESystem):
    """
    Implements the a discretization of the KdV-Burgers equation with an
    optional external force::

        u_t + eta u u_x - nu u_xx - gamma^2 u_xxx = f(u,x,t)

    on the pseudo-Hamiltonian formulation::

        u/dt = d/dx(grad[H(u)]) - grad[V(u)] + F(u, t, x)

    where u is a vector of the system states at the spatial points given by
    x, and t is time.

    The system is by default integrated in time using the implicit midpoint
    method. The line 'self.sample_trajectory = self.sample_trajectory_midpoint'
    can be commented out to instead use the RK45 method of scipy.

    Parameters
    ----------
    x : nparray, default np.linspace(0, 20.0 - 0.2, 100)
        The spatial discretization points
    eta : number, default 6.0
        The parameter eta in the KdV equation
    gamma : number, default 1.0
        The parameter gamma in the KdV equation
    nu : number, default 0.0
        The damping coefficient
    init_sampler : callable, default None
        Function for sampling initial conditions. Callabale taking
        a numpy random generator as input and returning an ndarray
        of shape same as x with inital conditions for the system.
        This sampler is used when calling
        KdVSystem.sample_trajectory if no initial
        condition is provided.
    kwargs : any, optional
        Keyword arguments that are passed to PseudoHamiltonianPDESystem
        constructor.

    """

    def __init__(
        self,
        x=np.linspace(0, 20.0 - 0.2, 100),
        eta=6.0,
        gamma=1.0,
        nu=0.0,
        init_sampler=None,
        **kwargs
    ):
        M = x.size
        dx = x[-1] / (M - 1)
        e = np.ones(M)
        # Forward difference matrix:
        Dp = 1 / dx * spdiags([e, -e, e], np.array([-M + 1, 0, 1]), M, M).toarray()
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
        skewsymmetric_matrix = D1
        self.x = x
        self.M = M
        self.eta = eta
        self.gamma = gamma
        self.nu = nu
        self.D2 = D2
        self.sample_trajectory = self.sample_trajectory_midpoint

        def ham(u):
            return np.sum(
                -1 / 6 * eta * u**3
                + (0.5 * gamma**2 * (np.matmul(Dp, u.T)) ** 2).T,
                axis=-1,
            )

        def dissintegral(u):
            return np.sum(0.5 * nu * (np.matmul(Dp, u.T) ** 2).T, axis=-1)

        def ham_grad(u):
            return -0.5 * eta * u**2 - (gamma**2 * u @ D2)

        def dissintegral_grad(u):
            return -nu * u @ D2

        def ham_hessian(u):
            return -eta * np.diag(u) - gamma**2 * D2

        def dissintegral_hessian(u):
            return -nu * D2

        if init_sampler is None:
            init_sampler = initial_condition_kdv(x, eta)

        super().__init__(
            nstates=M,
            skewsymmetric_matrix=skewsymmetric_matrix,
            hamiltonian=ham,
            dissintegral=dissintegral,
            grad_hamiltonian=ham_grad,
            grad_dissintegral=dissintegral_grad,
            hess_hamiltonian=ham_hessian,
            hess_dissintegral=dissintegral_hessian,
            init_sampler=init_sampler,
            **kwargs
        )

        self.skewsymmetric_matrix_flat = 0.5 / dx * np.array([[[-1, 0, 1]]])


def initial_condition_kdv(x=np.linspace(0, 20.0 - 0.2, 100), eta=6.0):
    """
    Creates an initial condition sampler for the KdV-Burgers equation.

    Parameters
    ----------
    x : numpy.ndarray, optional
        Spatial grid on which to create the initial conditions. The default is 
        an equidistant grid between 0 and 19.8 with step size 0.2.
    eta : float, optional
        The parameter eta in the KdV-Burgers equation. The default is 6.0.

    Returns
    -------
    callable
        A function that takes a numpy random generator as input and returns an
        initial state on the spatial grid x.
    """

    M = x.size
    P = (x[-1] - x[0]) * M / (M - 1)

    def sech(a):
        return 1 / np.cosh(a)

    def sampler(rng):
        k1, k2 = rng.uniform(0.5, 2.0, 2)
        d1, d2 = rng.uniform(0.0, 1.0, 1), rng.uniform(0.0, 1.0, 1)
        u0 = 0
        u0 += (
            (-6.0 / -eta)
            * 2
            * k1**2
            * sech(k1 * ((x + P / 2 - P * d1) % P - P / 2)) ** 2
        )
        u0 += (
            (-6.0 / -eta)
            * 2
            * k2**2
            * sech(k2 * ((x + P / 2 - P * d2) % P - P / 2)) ** 2
        )
        u0 = np.concatenate([u0[M:], u0[:M]], axis=-1)
        return u0

    return sampler
