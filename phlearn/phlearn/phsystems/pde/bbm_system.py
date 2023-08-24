import numpy as np
from scipy.sparse import spdiags

from .pseudo_hamiltonian_pde_system import PseudoHamiltonianPDESystem

__all__ = ['BBMSystem', 'initial_condition_bbm']

class BBMSystem(PseudoHamiltonianPDESystem):
    """
    BBMSystem class, representing a discretization of the Benjamin-Bona-Mahony
    (BBM) equation.

    Implements a discretization of the BBM equation with an optional
    viscosity term and external forces::

        u_t - u_xxt + u_x + u u_x - nu u_xx = f(u,x,t)

    on the pseudo-Hamiltonian formulation::

        (1-d^2/dx^2)(du/dt) = d/dx(grad[H(u)]) - grad[V(u)] + F(u, t, x)

    where u is a vector of the system states at the spatial points given by
    x, and t is time.

    The system is by default integrated in time using the implicit midpoint
    method. The line 'self.sample_trajectory = self.sample_trajectory_midpoint'
    can be commented out to instead use the RK45 method of scipy.

    Parameters
    ----------
    x : nparray, default np.linspace(0, 20.0 - 0.2, 100)
        The spatial discretization points
    nu : number, default 0.0
        The damping coefficient
    init_sampler : callable, default None
        Function for sampling initial conditions. Callable taking
        a numpy random generator as input and returning an ndarray
        of shape same as x with initial conditions for the system.
        This sampler is used when calling BBMSystem.sample_trajectory if no
        initial condition is provided.
    kwargs : any, optional
        Keyword arguments that are passed to PseudoHamiltonianPDESystem
        constructor.

    """

    def __init__(
        self, x=np.linspace(0, 20.0 - 0.2, 100), nu=0.0, init_sampler=None, **kwargs
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
        I = np.eye(M)
        lhs_matrix = I - D2
        self.lhs_matrix_flat = np.array([[[0, 1, 0]]]) - 1.0 / dx**2 * np.array(
            [[[1, -2, 1]]]
        )
        self.lhs_matrix = lhs_matrix
        skewsymmetric_matrix = D1
        self.x = x
        self.M = M
        self.nu = nu
        self.D2 = D2
        self.sample_trajectory = self.sample_trajectory_midpoint

        def ham(u):
            return np.sum(-1 / 2 * u**2 - 1 / 6 * u**3, axis=1)

        def dissintegral(u):
            return np.sum(0.5 * nu * (np.matmul(Dp, u.T) ** 2).T, axis=1)

        def ham_grad(u):
            return -u - 0.5 * u**2

        def dissintegral_grad(u):
            return -nu * u @ D2

        def ham_hessian(u):
            return -I - np.diag(u)

        def dissintegral_hessian(u):
            return -nu * D2

        if init_sampler is None:
            init_sampler = initial_condition_bbm(x)

        super().__init__(
            nstates=M,
            lhs_matrix=lhs_matrix,
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
        self.dissipation_matrix_flat = np.array([[[0, 1, 0]]])


def initial_condition_bbm(x=np.linspace(0, 50.0 - 0.5, 100)):
    """
    Creates an initial condition sampler for the Benjamin-Bona-Mahony equation.

    Parameters
    ----------
    x : numpy.ndarray, optional
        Spatial grid on which to create the initial conditions. The default is 
        an equidistant grid between 0 and 49.5 with step size 0.5.

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
        c1, c2 = rng.uniform(1.0, 4.0, 2)
        d1, d2 = rng.uniform(0.0, 1.0, 2)
        u0 = 0
        u0 += (
            3
            * (c1 - 1)
            * sech(1 / 2 * np.sqrt(1 - 1 / c1) * ((x + P / 2 - P * d1) % P - P / 2))
            ** 2
        )
        u0 += (
            3
            * (c2 - 1)
            * sech(1 / 2 * np.sqrt(1 - 1 / c2) * ((x + P / 2 - P * d2) % P - P / 2))
            ** 2
        )
        u0 = np.concatenate([u0[M:], u0[:M]], axis=-1)
        return u0

    return sampler
