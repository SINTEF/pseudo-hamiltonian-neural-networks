import numpy as np
from scipy.sparse import spdiags

from .pseudo_hamiltonian_pde_system import PseudoHamiltonianPDESystem


class AllenCahnSystem(PseudoHamiltonianPDESystem):
    """
    Implements a discretization of the Allen-Cahn equation with an
    optional external force, as described by

    u_t - u_xx - u + u^3 = f(u,t,x)

    on the pseudo-Hamiltonian formulation

    du/dt = -grad[V(u)] + F(u,t,x)

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
        x=np.linspace(0, 6.0 - 6 / 300, 300),
        nu=0.001,
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
        I = np.eye(M)
        skewsymmetric_matrix = D1
        self.x = x
        self.M = M
        self.D2 = D2

        def ham(u):
            return np.sum(np.zeros_like(u), axis=1)

        def dissintegral(u):
            return (
                1
                / 2
                * np.sum(
                    (nu * np.matmul(Dp, u.T) ** 2).T - u**2 + 1 / 2 * u**4, axis=1
                )
            )

        def ham_grad(u):
            return np.zeros_like(u)

        def dissintegral_grad(u):
            return -nu * u @ D2 - u + u**3

        def ham_hessian(u):
            return np.zeros_like(D1)

        def dissintegral_hessian(u):
            return -nu * D2 - I + 3 * np.diag(u**2)

        if init_sampler is None:
            init_sampler = initial_condition_ac(x)

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


def initial_condition_ac(x=np.linspace(0, 6.0 - 6 / 300, 300)):
    """
    Creates an initial condition sampler for the Allen-Cahn eqation.

    Parameters
    ----------
    x : numpy.ndarray, optional
        Spatial grid on which to create the initial conditions. The default is 
        an equidistant grid between 0 and 5.98 with step size 0.02.

    Returns
    -------
    callable
        A function that takes a numpy random generator as input and returns an
        initial state on the spatial grid x.
    """

    M = x.size
    P = (x[-1] - x[0]) * M / (M - 1)

    def sampler(rng):
        d1, d2 = rng.uniform(0.0, 1.0, 2)
        k1, k2 = rng.uniform(0.5, 1.0, 2)
        u0 = 0
        u0 += k1 * np.cos(2 * np.pi / P * (x - d1 * P))
        u0 += k2 * np.cos(2 * np.pi / P * (x - d2 * P))
        return u0

    return sampler
