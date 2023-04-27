import numpy as np

# import autograd.numpy as np
# from autograd import jacobian
# import numpy.linalg as la
from scipy.sparse import spdiags
import numpy.linalg as la
from .fvm import Burgers, solve



from .conservative_dissipative_system import (
    ConservativeDissipativeSystem,
)  # Should be changed to preservation-dissipation system

__all__ = ["BurgersSystem", "init_burgers", "initial_condition_burgers"]


class BurgersSystem(ConservativeDissipativeSystem):
    """
    Add description (see ODE examples)

    """

    def __init__(
        self,
        x=np.linspace(0, 20.0 - 0.2, 100),
        eta=6.0,
        gamma=1.0,
        nu=0.0,
        force=None,
        force_jac=None,
        **kwargs
    ):
        M = x.size
        dx = x[-1] / (M - 1)
        e = np.ones(M)
        deltafx = 1 / dx * spdiags([e, -e, e], np.array([-M + 1, 0, 1]), M, M)
        delta2cx = (
            1
            / dx**2
            * spdiags([e, e, -2 * e, e, e], np.array([-M + 1, -1, 0, 1, M - 1]), M, M)
        )
        Dp = deltafx.toarray()  # Fix this so it doesn't go via spdiags
        D2 = delta2cx.toarray()
        deltacx = (
            0.5 / dx * spdiags([e, -e, e, -e], np.array([-M + 1, -1, 1, M - 1]), M, M)
        )
        D1 = deltacx.toarray()
        structure_matrix = D1
        self.structure_matrix_flat = 0.5 / dx * np.array([[[-1, 0, 1]]])
        self.x = x
        self.M = M
        self.eta = eta
        self.gamma = gamma
        self.D2 = D2

        def ham(u):
            return np.sum(
                1 / 6 * eta * u**3 - (0.5 * gamma**2 * (np.matmul(Dp, u.T)) ** 2).T,
                axis=1,
            )

        def ham_grad(u):
            return 0.5 * eta * u**2 + (gamma**2 * np.matmul(D2, u.T)).T

        super().__init__(
            nstates=M,
            skewsymmetric_matrix=structure_matrix,
            hamiltonian=ham,
            grad_hamiltonian=ham_grad,
            **kwargs
        )

    def sample_trajectory(self, t, x0=None, noise_std=0, reference=None):
        """
        Samples a trajectory of the system at times *t*.

        Parameters
        ----------
        t : (T, 1) ndarray
            Times at which the trajectory is sampled.
        x0 : (N,) ndarray, default None
            Initial condition.
        noise_std : number, default 0.
            Standard deviation of Gaussian white noise added to the
            samples of the trajectory.
        reference : porthamiltonian.control.Reference, default None
            If the system has a controller a reference object may be
            passed.

        Returns
        -------
        x : (T, N) ndarray
        dxdt : (T, N) ndarray
        t : (T, 1) ndarray
        us : (T, N) ndarray

        """

        if x0 is None:
            x0 = self._initial_condition_sampler(self.rng)

        x = np.zeros([t.shape[0], x0.shape[-1]])
        dxdt = np.zeros_like(x)
        us = np.zeros([t.shape[0] - 1, x0.shape[-1]])
        x[0, :] = x0
        _, u, all_u, du_dts = solve(
            x0, t[-1], x.shape[1], x.shape[0], Burgers(), x_end=self.x.amax()
        )
        assert noise_std == 0
        return all_u, du_dts, t, us


def init_burgers():
    """
    Initialize a standard Burgers system

    Returns
    -------
    BurgersSystem

    """
    x_end = 20
    x_points = 100
    dx = x_end / x_points
    x = np.linspace(0, x_end - dx, x_points)
    eta = 6.0
    gamma = 1.0

    return BurgersSystem(
        x=x, eta=eta, gamma=gamma, init_sampler=initial_condition_burgers(x, eta)
    )


def initial_condition_burgers(x=np.linspace(0, 20.0 - 0.2, 100), eta=6.0):
    """
    Add description (see ODE examples)

    """
    M = x.size
    P = (x[-1] - x[0]) * M / (M - 1)
    sech = lambda a: 1 / np.cosh(a)

    def sampler(rng):
        k1, k2 = rng.uniform(0.5, 2.0, 2)
        d1 = rng.uniform(0.2, 0.3, 1)
        d2 = d1 + rng.uniform(0.2, 0.5, 1)
        u0 = 0
        u0 += (-6.0 / -eta) * 2 * k1**2 * sech(k1 * (x - P * d1)) ** 2
        u0 += (-6.0 / -eta) * 2 * k2**2 * sech(k2 * (x - P * d2)) ** 2
        u0 = np.concatenate([u0[M:], u0[:M]], axis=-1)
        # Diversifying between having the smaller wave to the left or right of the bigger one:
        if np.random.randint(0, 2) == 1:
            u0 = u0[::-1].copy()
        return u0

    return sampler
