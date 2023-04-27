
import autograd.numpy as np
import autograd
from scipy.integrate import solve_ivp

from ..utils.derivatives import time_derivative
from ..utils.utils import midpoint_method

__all__ = ['ConservativeDissipativeSystem', 'zero_force']


class ConservativeDissipativeSystem():
    """
    Implements a conservative-dissipative system of the form::

        dx/dt = S(x)*grad[H(x)] - R(x)*grad[V(x)] + F(x, t)

    where x is the system state, S is the interconection matrix,
    H is the Hamiltonian of the system, V is the dissipating integral,
    F is the external forces.

    Parameters
    ----------
        nstates : int
            Number of system states N.

        skewsymmetric_matrix : (N, N) ndarray or callable, default None
            Corresponds to the S matrix. Must either be an
            ndarray, or callable taking an ndarray
            input of shape (nsamples, nstates) and returning an ndarray
            of shape (nsamples, nstates, nstates). If None,
            the system is assumed to be canonical, and the
            S matrix is set ot the skew-symmetric matrix
            [[0, I_n], [-I_n, 0]].

        hamiltonian : callable, default None
            The Hamiltonian H of the system. Callable taking a
            torch tensor input of shape (nsamples, nstates) and
            returning a torch tensor of shape (nsamples, 1).
            If the gradient of the Hamiltonian is not provided,
            the gradient of this function will be computed by torch and
            used instead. If this is not provided, the grad_hamiltonian
            must be provided.

        dissintegral : callable, default None
            The dissipating integral V of the system. Callable taking a
            torch tensor input of shape (nsamples, nstates) and
            returning a torch tensor of shape (nsamples, 1).
            If the gradient of the dissipating integral is not provided,
            the gradient of this function will be computed by torch and
            used instead. If this is not provided, the grad_dissintegral
            must be provided.

        grad_hamiltonian : callable, default None
            The gradient of the Hamiltonian H of the system. Callable
            taking an ndarray input of shape (nsamples, nstates) and
            returning a torch tensor of shape (nsamples, nstates).
            If this is not provided, the hamiltonian must be provided.

        grad_dissintegral : callable, default None
            The gradient of the dissipating integral V of the system. Callable
            taking an ndarray input of shape (nsamples, nstates) and
            returning a torch tensor of shape (nsamples, nstates).
            If this is not provided, the hamiltonian must be provided.

        external_forces : callable, default None
            The external forces affecting system. Callable taking two
            ndarrays as input, x and t, of shape (nsamples, nstates),
            (nsamples, 1), respectively and returning an ndarray of
            shape (nsamples, nstates).

        controller : phlearn.control.PseudoHamiltonianController,
        default None
            Additional external forces set by a controller. Callable
            taking an ndarray x of shape (nstates,) and a scalar t as
            input and returning an ndarray of shape (nstates,). Note
            that this function should not take batch inputs, and that
            when calling PseudoHamiltonianSystem.sample_trajectory when a
            controller is provided, the Runge-Kutta 4 method will be
            used for integration in favor of Scipy's solve_ivp.

        init_sampler : callable, default None
            Function for sampling initial conditions. Callabale taking
            a numpy random generator as input and returning an ndarray
            of shape (nstates,) with inital conditions for the system.
            This sampler is used when calling
            PseudoHamiltonianSystem.sample_trajectory if no initial
            condition is provided.

    """

    def __init__(self, nstates, lhs_matrix=None,
                 skewsymmetric_matrix=None,
                 dissipation_matrix=None,
                 hamiltonian=None, dissintegral=None,
                 grad_hamiltonian=None, grad_dissintegral=None,
                 hess_hamiltonian=None, hess_dissintegral=None,
                 external_forces=None, jac_external_forces=None,
                 controller=None,
                 init_sampler=None):

        self.nstates = nstates

        self.lhs_matrix_provided = True
        if lhs_matrix is None:
            lhs_matrix = np.eye(nstates)
            self.lhs_matrix_provided = False

        self.lhs_matrix = lhs_matrix
        self.A = lambda x: lhs_matrix # check if this is necessary
        
        if skewsymmetric_matrix is None:
            npos = nstates // 2
            skewsymmetric_matrix = np.block(
                [[np.zeros([npos, npos]), np.eye(npos)],
                 [-np.eye(npos), np.zeros([npos, npos])]])

        if not callable(skewsymmetric_matrix):
            self.skewsymmetric_matrix = skewsymmetric_matrix
            self.S = lambda x: skewsymmetric_matrix
        else:
            self.skewsymmetric_matrix = None
            self.S = skewsymmetric_matrix

        if dissipation_matrix is None:
            dissipation_matrix = np.eye(nstates)

        self.dissipation_matrix = dissipation_matrix
        self.R = lambda x: dissipation_matrix # check if this is necessary

        self.H = hamiltonian
        self.dH = grad_hamiltonian
        self.ddH = hess_hamiltonian
        if grad_hamiltonian is None:
            self.dH = self._dH
        if hess_hamiltonian is None:
            self.ddH = self._ddH

        self.V = dissintegral
        self.dV = grad_dissintegral
        self.ddV = hess_dissintegral
        if grad_dissintegral is None:
            self.dV = self._dV
        if hess_dissintegral is None:
            self.ddV = self._ddV

        self.controller = controller

        self.external_forces = external_forces
        self.external_forces_jacobian = jac_external_forces
        if external_forces is None:
            self.external_forces = zero_force
            self.external_forces_jacobian = None
        elif jac_external_forces is None:
            self.external_forces_jacobian = self._jacforce

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
        Computes the time derivative, the right hand side of the pseudo-
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
        if self.H is not None:
            dH = self.dH(x)
        else:
            dH = np.zeros_like(x)
        if self.V is not None:
            dV = self.dV(x)
        else:
            dV = np.zeros_like(x)
        if (len(S.shape) == 3):
            dynamics = ((np.matmul(S, np.atleast_3d(dH))
                        - np.matmul(R,np.atleast_3d(dV))).reshape(x.shape)
                        + self.external_forces(x, t))
        else:
            dynamics = dH@(S.T) - dV@R + self.external_forces(x, t)
        if u is not None:
            dynamics += u
        return dynamics
    
    def x_dot_jacobian(self, x, t, u=None):
        """
        Computes the Jacobian of the right hand side of the pseudo-
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
        jacobian = np.zeros_like(S)
        if self.H is not None:
            ddH = self.ddH(x)
            jacobian += np.matmul(S,ddH)
        if self.V is not None:
            ddV = self.ddV(x)
            jacobian -= np.matmul(R,ddV)
        if self.external_forces_jacobian is not None:
            jacobian += self.external_forces_jacobian(x, t)
        return jacobian

    def sample_trajectory(self, t, x0=None, noise_std=0, reference=None):
        """
        Samples a trajectory of the system at times *t*, found by using the
        solve_ivp solver for temporal integration, starting from the initial
        state x0.

        Parameters
        ----------
        t : (T, 1) ndarray
            Times at which the trajectory is sampled.
        x0 : (N,) ndarray, default None
            Initial condition.
        noise_std : number, default 0.
            Standard deviation of Gaussian white noise added to the
            samples of the trajectory.
        reference : phlearn.control.Reference, default None
            Not in use for now.

        Returns
        -------
        x : (T, N) ndarray
        dxdt : (T, N) ndarray
        t : (T, 1) ndarray
        us : (T, N) ndarray

        """

        if x0 is None:
            x0 = self._initial_condition_sampler(self.rng)

        if self.lhs_matrix_provided:
            lhs_matrix_inv = np.linalg.inv(self.lhs_matrix)
            x_dot = lambda t, x: np.matmul(lhs_matrix_inv,
                                           self.x_dot(x.reshape(1, x.shape[-1]),
                                           np.array(t).reshape((1, 1))).T).T
        else:
            x_dot = lambda t, x: self.x_dot(x.reshape(1, x.shape[-1]),
                                            np.array(t).reshape((1, 1)))
        out_ivp = solve_ivp(fun=x_dot, t_span=(t[0], t[-1]), y0=x0,
                            t_eval=t, rtol=1e-10)
        x, t = out_ivp['y'].T, out_ivp['t'].T
        dxdt = self.x_dot(x, t)
        us = None

        # Add noise:
        x += self.rng.normal(size=x.shape)*noise_std
        dxdt += self.rng.normal(size=dxdt.shape)*noise_std

        return x, dxdt, t, us
    
    def sample_trajectory_midpoint(self, t, x0=None, noise_std=0, reference=None):
        """
        Samples a trajectory of the system at times *t*, found by using the
        implicit midpoint method for temporal integration, starting from the
        initial state x0. Newton's method is used for solving the system of
        nonlinear equations at each integration step.

        Parameters
        ----------
        t : (T, 1) ndarray
            Times at which the trajectory is sampled.
        x0 : (N,) ndarray, default None
            Initial condition.
        noise_std : number, default 0.
            Standard deviation of Gaussian white noise added to the
            samples of the trajectory.
        reference : phlearn.control.Reference, default None
            Not in use for now.

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

        M = x0.shape[-1]
        if self.lhs_matrix_provided:
            lhs_matrix_inv = np.linalg.inv(self.lhs_matrix)
            f = lambda u, t: np.linalg.solve(self.lhs_matrix, self.x_dot(u,t))
            Df = lambda u, t: np.linalg.solve(self.lhs_matrix, self.x_dot_jacobian(u,t))
        else:
            f = lambda u, t: self.x_dot(u,t)
            Df = lambda u, t: self.x_dot_jacobian(u,t)
        for i, t_step in enumerate(t[:-1]):
            dt = t[i + 1] - t[i]
            dxdt[i, :] = f(x[i,:], t[i])
            x[i+1,:] = midpoint_method(x[i,:],x[i,:],t[i],f,Df,dt,M,1e-12,5)

        # Add noise:
        x += self.rng.normal(size=x.shape)*noise_std
        dxdt += self.rng.normal(size=dxdt.shape)*noise_std

        return x, dxdt, t, us

    def _dH(self, x):
        H = lambda x: self.H(x).sum()
        return autograd.grad(H)(x)

    def _dV(self, x):
        V = lambda x: self.V(x).sum()
        return autograd.grad(V)(x)
    
    def _ddH(self, x):
        H = lambda x: self.H(x).sum()
        return autograd.hessian(H)(x)

    def _ddV(self, x):
        H = lambda x: self.H(x).sum()
        return autograd.hessian(H)(x)
    
    def _jacforce(self, x, t):
        external_forces_x = lambda x: self.external_forces(x, t)
        return autograd.jacobian(external_forces_x)(x)

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
