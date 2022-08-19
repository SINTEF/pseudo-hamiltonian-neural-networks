
import numpy as np
from scipy.integrate import solve_ivp
import torch

from ..utils.derivatives import time_derivative

__all__ = ['PortHamiltonianSystem', 'zero_force']


class PortHamiltonianSystem():
    """
    Implements a port-Hamiltonian system of the form::

        dx/dt = (S(x) - R(x))*grad[H(x)] + F(x, t)

    where x is the system state, S is the interconection matrix,
    R is the dissipation-matrix, H is the Hamiltonian of the system,
    F is the external port.

    Parameters
    ----------
        nstates : int
            Number of system states N.

        structure_matrix : (N, N) ndarray or callable, default None
            Corresponds to the S matrix. Must either be an
            ndarray, or callable taking an ndarray
            input of shape (nsamples, nstates) and returning an ndarray
            of shape (nsamples, nstates, nstates). If None,
            the system is assumed to be canonical, and the
            S matrix is set ot the skew-symmetric matrix
            [[0, I_n], [-I_n, 0]].

        dissipation_matrix : (N, N) ndarray or callable, default None
            Corresponds to the R matrix. Must either be an
            ndarray, or callable taking an ndarray input of shape
            (nsamples, nstates) and returning an ndarray of shape
            (nsamples, nstates, nstates).

        hamiltonian : callable, default None
            The Hamiltonian H of the system. Callable taking a
            torch tensor input of shape (nsamples, nstates) and
            returning a torch tensor of shape (nsamples, 1).
            If the gradient of the Hamiltonian is not provided,
            the gradient of this function will be computed by torch and
            used instead. If this is not provided, the grad_hamiltonian
            must be provided.

        grad_hamiltonian : callable, default None
            The gradient of the Hamiltonian H of the system. Callable
            taking an ndarray input of shape (nsamples, nstates) and
            returning a torch tensor of shape (nsamples, nstates).
            If this is not provided, the hamiltonian must be provided.

        external_port : callable, default None
            The external ports affecting system. Callable taking two
            ndarrays as input, x and t, of shape (nsamples, nstates),
            (nsamples, 1), respectively and returning an ndarray of
            shape (nasamples, nstates).

        controller : porthamiltonians.control.PortHamiltonianController,
        default None
            Additional external ports set by a controller. Callable
            taking an ndarray x of shape (nstates,) and a scalar t as
            input and returning an ndarray of shape (nstates,). Note
            that this function should not take batch inputs, and that
            when calling PortHamiltonianSystem.sample_trajectory when a
            controller is provided, the Runge-Kutta 4 method will be
            used for integration in favor of Scipy's solve_ivp.

        init_sampler : callable, default None
            Function for sampling initial conditions. Callabale taking
            a numpy random generator as input and returning an ndarray
            of shape (nstates,) with inital conditions for the system.
            This sampler is used when calling
            PortHamiltonianSystem.sample_trajectory if no initial
            condition is provided.

    """

    def __init__(self, nstates, structure_matrix=None,
                 dissipation_matrix=None, hamiltonian=None,
                 grad_hamiltonian=None, external_port=None,
                 controller=None, init_sampler=None):

        self.nstates = nstates

        if structure_matrix is None:
            npos = nstates // 2
            structure_matrix = np.block(
                [[np.zeros([npos, npos]), np.eye(npos)],
                 [-np.eye(npos), np.zeros([npos, npos])]])

        if not callable(structure_matrix):
            self.structure_matrix = structure_matrix
            self.S = lambda x: structure_matrix
        else:
            self.structure_matrix = None
            self.S = structure_matrix

        if dissipation_matrix is None:
            dissipation_matrix = np.zeros((self.nstates, self.nstates))

        if not callable(dissipation_matrix):
            if len(dissipation_matrix.shape) == 1:
                dissipation_matrix = np.diag(dissipation_matrix)
            self.dissipation_matrix = dissipation_matrix
            self.R = lambda x: dissipation_matrix
        else:
            self.dissipation_matrix = None
            self.R = dissipation_matrix

        self.H = hamiltonian
        self.dH = grad_hamiltonian
        if grad_hamiltonian is None:
            self.dH = self._dH

        self.controller = controller

        self.external_port = external_port
        if external_port is None:
            self.external_port = zero_force

        if init_sampler is not None:
            self._initial_condition_sampler = init_sampler

        self.seed(None)

    def seed(self, seed):
        """
        Set the internal random state.

        Parameters
        ----------
        seed : int

        """

        self.rng = np.random.default_rng(seed)

    def time_derivative(self, integrator, *args, **kwargs):
        """
        See :py:meth:~`utils.derivatives.time_derivative`
        """
        return time_derivative(integrator, self.x_dot, *args, **kwargs)

    def x_dot(self, x, t, u=None):
        """
        Computes the time derivative, the right hand side of the port-
        Hamiltonian equation.

        Parameters
        ----------
        x : (..., N) ndarray
        t : (..., 1) ndarray
        u : (..., N) ndarray or None, default None

        Returns
        -------
        (..., N) ndarray

        """

        S = self.S(x)
        R = self.R(x)
        dH = self.dH(x)
        if (len(S.shape) == 3) or (len(R.shape) == 3):
            dynamics = (np.matmul(S - R, np.atleast_3d(dH)).reshape(x.shape)
                        + self.external_port(x, t))
        else:
            dynamics = dH@(S.T - R.T) + self.external_port(x, t)
        if u is not None:
            dynamics += u
        return dynamics

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

        if self.controller is None:
            x_dot = lambda t, x: self.x_dot(x.reshape(1, x.shape[-1]),
                                            np.array(t).reshape((1, 1)))
            out_ivp = solve_ivp(fun=x_dot, t_span=(t[0], t[-1]), y0=x0,
                                t_eval=t, rtol=1e-10)
            x, t = out_ivp['y'].T, out_ivp['t'].T
            dxdt = self.x_dot(x, t)
            us = None
        else:
            # Use RK4 integrator instead of solve_ivp when controller
            # is provided
            self.controller.reset()
            if reference is not None:
                self.controller.set_reference(reference)
            x = np.zeros([t.shape[0], x0.shape[-1]])
            dxdt = np.zeros_like(x)
            us = np.zeros([t.shape[0] - 1, x0.shape[-1]])
            x[0, :] = x0
            for i, t_step in enumerate(t[:-1]):
                dt = t[i + 1] - t[i]
                us[i, :] = self.controller(x[i, :], t_step)
                dxdt[i, :] = self.time_derivative('rk4', x[i:i+1, :],
                                                  x[i:i+1, :],
                                                  np.array([t_step]),
                                                  np.array([t_step]),
                                                  dt, u=us[i:i+1, :])
                x[i + 1, :] = x[i, :] + dt*dxdt[i, :]

        # Add noise:
        x += self.rng.normal(size=x.shape)*noise_std
        dxdt += self.rng.normal(size=dxdt.shape)*noise_std

        return x, dxdt, t, us

    def set_controller(self, controller):
        """
        Set system controller.
        """
        self.controller = controller

    def _dH(self, x):
        x = torch.tensor(x, requires_grad=True)
        return torch.autograd.grad(self.H(x).sum(), x, retain_graph=False,
                                   create_graph=False)[0].detach().numpy()

    def _initial_condition_sampler(self, rng=None):
        if rng is None:
            assert self.rng is not None
            rng = self.rng
        return rng.uniform(low=-1., high=1.0, size=self.nstates)


def zero_force(x, t=None):
    """
    A force term that is always zero.

    Parameters
    ----------
    x : (..., N) ndarray
    t : (..., 1) ndarray, default None

    Returns
    -------
    (..., N) ndarray
        All zero ndarray

    """

    return np.zeros_like(x)
