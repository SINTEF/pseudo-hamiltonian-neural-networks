
import numpy as np
from scipy.integrate import solve_ivp
import torch

from ..utils.derivatives import time_derivative
from ..utils.utils import to_tensor

__all__ = ['DynamicSystemNN']


class DynamicSystemNN(torch.nn.Module):
    """
    Base class for implementing neural networks estimating the right
    hand side of equations of the form::

        dx/dt = f(x, t) + u

    where x is the system state, t is time and u is optional control
    inputs.

    Parameters
    ----------
    nstates : int
        Number of system states.

    rhs_model : callable, default None
        Model estimating the right hand side of the above equation.
        Should take inputs x and t, where x is a tensor of shape
        (nsamples, nstates) and t is a tensor of shape (nsamples, 1),
        and return a tensor of shape (nsamples, nstates), estimating the
        time derivative of each state in each sample.

    controller : callable, default None
        Additional external ports set by a controller. Callable taking a
        tensor x of shape (nstates,) and a scalar t as input and
        returning a tensor of shape (nstates,). Note that this function
        should not take batch inputs, and that when calling
        PortHamiltonianNN.sample_trajectory when a controller is
        provided, the Runge-Kutta 4 method will be used for integration
        in favor of Scipy's solve_ivp.

    init_sampler : callable, default None
        Function for sampling initial conditions. Callabale taking a
        number specifying the number of inital conditions to sample, M,
        as input and returning a tensor of shape (M, nstates) with
        inital conditions for the system. This sampler is used when
        calling :py:meth:`~DynamicSystemNN.simulate_trajectory` and
        :py:meth:`~DynamicSystemNN.simulate_trajectories` if no initial
        condition is provided. If not provided, initial conditions are
        uniformly sampled from (0, 1).

    ttype : torch type, default torch.float32

    """

    def __init__(self,
                 nstates,
                 rhs_model=None,
                 init_sampler=None,
                 controller=None,
                 ttype=torch.float32):
        super().__init__()
        self.ttype = ttype
        self.nstates = nstates
        self.controller = controller
        self.model = rhs_model
        if init_sampler is not None:
            self._initial_condition_sampler = init_sampler
        self.rhs_model = self._x_dot

    def seed(self, seed):
        """
        Set the torch seed.

        Parameters
        ----------
        seed : int

        """

        torch.manual_seed(seed)

    def time_derivative(self, integrator, *args, **kwargs):
        """
        See :py:meth:~`utils.derivatives.time_derivative`
        """

        return time_derivative(integrator, self.rhs_model, *args, **kwargs)

    def simulate_trajectory(self, integrator, t_sample, x0=None,
                            noise_std=0., reference=None):
        """
        Simulate a trajectory using the rhs_model and sample at times
        *t_sample*.

        Parameters
        ----------
        integrator : str or False
            Specifies which solver to use during simulation. If False,
            the problem is left to scipy's solve_ivp. If 'euler',
            'midpoint', 'rk4' or 'srk4' the system is simulated with
            the forward euler method, the implicit midpoint method,
            the explicit Runge-Kutta 4 method or a symmetric fourth
            order Runge-Kutta method, respectively.
        t_sample : (T, 1) tensor or ndarray
            Times at which the trajectory is sampled.
        x0 : (N,) tensor or ndarray, default None
            Initial condition. If None, an initial condition is sampled
            with the internal sampler.
        noise_std : number, default 0.
            Standard deviation of Gaussian white noise added to the
            samples of the trajectory.
        reference : porthamiltonian.control.Reference, default None
            If the system has a controller a reference object may be
            passed.

        Returns
        -------
        xs : (T, N) tensor
        us : (T, N) tensor

        """

        x0 = to_tensor(x0)
        if x0 is None:
            x0 = self._initial_condition_sampler(1)

        if not integrator and self.controller is None:
            x_dot = lambda t, x: self.rhs_model(
                        torch.tensor(x.reshape(1, x.shape[-1]),
                                     dtype=self.ttype),
                        torch.tensor(np.array(t).reshape((1, 1)),
                                     dtype=self.ttype)
                        ).detach().numpy().flatten()
            out_ivp = solve_ivp(fun=x_dot, t_span=(t_sample[0], t_sample[-1]),
                                y0=x0.detach().numpy().flatten(),
                                t_eval=t_sample, rtol=1e-10)
            xs = out_ivp['y'].T
            us = None
        else:
            t_sample = to_tensor(t_sample, self.ttype)
            if not integrator and self.controller is not None:
                integrator = 'rk4'
                print('Warning: Since the system contains a controller, '
                      'the rk4 integrator is used to simulate the trajectory '
                      'instead of solve_ivp.')
            elif integrator.lower() not in ['euler', 'rk4']:
                print('Warning: Only explicit integrators euler and rk4 or no '
                      'integrator (False) allowed for inference. Ignoring '
                      f'integrator {integrator} and using rk4.')
                integrator = 'rk4'

            if self.controller is not None:
                self.controller.reset()
                if reference is not None:
                    self.controller.set_reference(reference)
            nsteps = t_sample.shape[0]
            x0 = x0.reshape(1, x0.shape[-1])
            xs = torch.zeros([nsteps, x0.shape[-1]])
            xs[0, :] = x0

            u = None
            us = torch.zeros([nsteps - 1, x0.shape[-1]])

            for i, t_step in enumerate(t_sample[:-1]):
                t_step = torch.squeeze(t_step).reshape(-1, 1)
                if self.controller is not None:
                    u = to_tensor(self.controller(xs[i, :], t_step),
                                  self.ttype)
                    us[i, :] = u
                dt = t_sample[i + 1] - t_step
                xs[i + 1, :] = xs[i, :] + dt*self.time_derivative(
                    integrator, xs[i:i+1, :], xs[i:i+1, :],
                    t_step, t_step, dt, u)
            xs = xs.detach().numpy()
            if self.controller is not None:
                us = us.detach().numpy()
            else:
                us = None

        return xs, us

    def simulate_trajectories(self, ntrajectories, integrator, t_sample,
                              x0=None, noise_std=0, references=None):
        """
        Calls :py:meth:`~DynamicSystemNN.simulate_trajectory`
        *ntrajectories* times.

        Parameters
        ----------
        integrator : str or False
            Specifies which solver to use during simulation. If False,
            the problem is left to scipy's solve_ivp. If 'euler',
            'midpoint', 'rk4' or 'srk4' the system is simulated with
            the forward euler method, the implicit midpoint method,
            the explicit Runge-Kutta 4 method or a symmetric fourth
            order Runge-Kutta method, respectively.
        t_sample : (T, 1) or (ntrajectories, T, 1) tensor or ndarray
            Times at which the trajectory is sampled.
        x0 : (ntrajectories, N) tensor or ndarray, default None
            Initial condition. If None, an initial condition is sampled
            with the internal sampler.
        noise_std : number, default 0.
            Standard deviation of Gaussian white noise added to the
            samples of the trajectory.
        references : list of porthamiltonian.control.Reference, default
        None
            If the system has a controller a list of ntrajectories
            reference objects may be passed.

        Returns
        -------
        xs : (ntrajectories, T, N) tensor
        t_sample : (ntrajectories, T, 1) tensor
        us : (ntrajectories, T, N) tensor or None

        """

        if integrator in ('euler', 'rk4') and self.controller is None:
            if x0 is None:
                x0 = self._initial_condition_sampler(ntrajectories, self.rng)
            x0 = to_tensor(x0, self.ttype)
            t_sample = to_tensor(t_sample, self.ttype)

            if len(t_sample.shape) == 1:
                t_sample = np.tile(t_sample, (ntrajectories, 1))

            dt = t_sample[0, 1] - t_sample[0, 0]
            nsteps = t_sample.shape[-1]
            x0 = x0.reshape(ntrajectories, self.nstates)
            t_sample = t_sample.reshape(ntrajectories, nsteps, 1)
            xs = torch.zeros([ntrajectories, nsteps, self.nstates])
            xs[:, 0, :] = x0
            for i in range(nsteps - 1):
                xs[:, i + 1, :] = xs[:, i] + dt*self.time_derivative(
                    integrator, xs[i:i+1, :], xs[i:i+1, :],
                    t_sample, t_sample, dt)

            xs = xs.detach().numpy()
            t_sample = t_sample.detach().numpy()
            us = None
        else:
            t_sample = np.atleast_2d(t_sample)
            if t_sample.shape[0] == 1:
                t_sample = np.repeat(t_sample, ntrajectories, 0)
            else:
                assert t_sample.shape[0] == ntrajectories
            nsteps = t_sample.shape[-1]
            xs = torch.zeros([ntrajectories, nsteps, self.nstates])
            us = torch.zeros((ntrajectories, nsteps - 1, self.nstates))
            if references is None:
                references = [None] * ntrajectories

            for i in range(ntrajectories):
                xs[i], us[i] = self.simulate_trajectory(
                    integrator=integrator, t_sample=t_sample[i],
                    x0=x0[i], noise_std=noise_std, reference=references[i])

            if self.controller is None:
                us = None

            if len(t_sample.shape) == 1:
                t_sample = torch.tile(t_sample, (ntrajectories, 1))
            t_sample = t_sample.reshape(ntrajectories, nsteps, 1)

        return xs, t_sample, us

    def set_controller(self, controller):
        """
        Set controller.
        """

        self.controller = controller

    def _x_dot(self, x, t, u=None):
        x = to_tensor(x, self.ttype)
        t = to_tensor(t, self.ttype)
        u = to_tensor(u, self.ttype)

        dynamics = self.model(x, t)
        if u is not None:
            dynamics += u
        return dynamics

    def _initial_condition_sampler(self, nsamples=1):
        return 2*torch.rand((nsamples, self.nstates), dtype=self.ttype) - 1
