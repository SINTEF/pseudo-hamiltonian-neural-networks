import numpy as np
from scipy.sparse import spdiags

from .pseudo_hamiltonian_pde_system import PseudoHamiltonianPDESystem


class HeatEquationSystem(PseudoHamiltonianPDESystem):
    """
    Implements a discretization of the heat equation with an
    optional external force,

    u_t - u_xx = f(u,t,x),

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
        nu=1.0,
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
        skewsymmetric_matrix = D1
        self.x = x
        self.M = M
        self.D2 = D2

        def ham(u):
            return np.sum(np.zeros_like(u), axis=1)

        def dissintegral(u):
            return np.sum(0.5 * nu * (np.matmul(Dp, u.T) ** 2).T, axis=1)

        def ham_grad(u):
            return np.zeros_like(u)

        def dissintegral_grad(u):
            return -nu * u @ D2

        def ham_hessian(u):
            return np.zeros_like(D1)

        def dissintegral_hessian(u):
            return -nu * D2

        if init_sampler is None:
            init_sampler = initial_condition_heat(x)

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


def initial_condition_heat(x=np.linspace(0, 6.0 - 6 / 300, 300)):
    """
    Creates an initial condition sampler for the heat eqation.

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
        d1, d2 = rng.uniform(0.3, 3, 2)
        c1, c2 = rng.uniform(0.5, 1.5, 2)
        k1 = rng.uniform(0.5, 3.0, 1)
        k2 = rng.uniform(10.0, 20.0, 1)
        n1 = rng.uniform(20.0, 40.0, 1)
        n2 = rng.uniform(0.05, 0.15, 1)
        u0 = 0
        u0 += (
            rng.uniform(-5.0, 5.0, 1)
            - c1 * np.tanh(n1 * (x - d1))
            + c1 * np.tanh(n1 * (x - P + d1))
        )
        u0 += -c2 * np.tanh(n1 * (x - d2)) + c2 * np.tanh(n1 * (x - P + d2))
        u0 += n2 * np.sin(k1 * np.pi * x) ** 2 * np.sin(k2 * np.pi * x)
        return u0

    return sampler