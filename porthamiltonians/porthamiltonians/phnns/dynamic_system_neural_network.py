
import numpy as np
from scipy.integrate import solve_ivp
import torch

from ..utils.derivatives import time_derivative


class DynamicSystemNN(torch.nn.Module):
    """
    Base class for implementing neural networks estimating the right hand side
    of equations of the form
        dx/dt = f(x, t) + u
    where x is the system state, t is time and u is optional control inputs.


    parameters
    ----------
        nstates            : Number of system states.

        rhs_model          : Model estimating the right hand side of the above equation. Should
                             take inputs x and t, where x is a tensor of shape (nsamples, nstates)
                             and t is a tensor of shape (nsamples, 1), and return a tensor of shape
                             (nsamples, nstates), estimating the time derivative of each state in each
                             sample.

        controller         : Additional external ports set by a controller. Callable taking a tensor x
                             of shape (nstates,) and a scalar t as input and returning
                             a tensor of shape (nstates,). Note that this function should not take batch inputs,
                             and that when calling PortHamiltonianNN.sample_trajectory when a controller
                             is provided, the Runge-Kutta 4 method will be used for integration in favor of
                             Scipy's solve_ivp.

        init_sampler       : Function for sampling initial conditions. Callabale taking a torch random generator
                             as input and returning a tensor of shape (nstates,) with inital conditions for
                             the system. This sampler is used when calling DynamicSystemNN.sample_trajectory
                             if no initial condition is provided.

        ttype              : Torch type.
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
        self.rhs_model = rhs_model
        if init_sampler is not None:
            self._initial_condition_sampler = init_sampler

    def seed(self, seed):
        torch.manual_seed(seed)

    def time_derivative(self, integrator, *args):
        return time_derivative(integrator, self.rhs_model, *args)

    def simulate_trajectory(self, integrator, t_sample, x0=None, noise_std=0, reference=None):
        if x0 is None:
            x0 = self._initial_condition_sampler(1)

        if not integrator and self.controller is None:
            x_dot = lambda t, x: self.rhs_model(
                        torch.tensor(x.reshape(1, x.shape[-1]), dtype=self.ttype),
                        torch.tensor(np.array(t).reshape((1, 1)), dtype=self.ttype)).detach().numpy().flatten()
            out_ivp = solve_ivp(fun=x_dot, t_span=(t_sample[0], t_sample[-1]), y0=x0.detach().numpy().flatten(),
                                t_eval=t_sample, rtol=1e-10)
            xs = out_ivp['y'].T
            us = None
        else:
            if not integrator and self.controller is not None:
                integrator = 'rk4'
                print('Warning: Since the system contains a controller, the RK4 integrator is used to simulate the trajectory instead of solve_ivp')
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
                    u = torch.tensor(self.controller(xs[i, :], t_step), dtype=self.ttype)
                    us[i, :] = u
                dt = t_sample[i + 1] - t_step
                xs[i + 1, :] = xs[i, :] + dt*self.time_derivative(integrator, xs[i:i+1, :], xs[i:i+1, :],
                                                                  t_step, t_step, dt, u)
            xs = xs.detach().numpy()
            if self.controller is not None:
                us = us.detach().numpy()
            else:
                us = None

        return xs, us

    def simulate_trajectories(self, ntrajectories, integrator, t_sample, x0=None, noise_std=0, references=None):
        if integrator in ('euler', 'rk4') and self.controller is None:
            if x0 is None:
                x0 = self._initial_condition_sampler(ntrajectories, self.rng)

            if len(t_sample.shape) == 1:
                t_sample = np.tile(t_sample, (ntrajectories, 1))

            dt = t_sample[0, 1] - t_sample[0, 0]
            nsteps = t_sample.shape[-1]
            x0 = torch.tensor(x0.reshape(ntrajectories, self.nstates), dtype=self.ttype)
            t_sample = torch.tensor(t_sample.reshape(ntrajectories, nsteps, 1), dtype=self.ttype)
            dt = torch.tensor(dt, dtype=self.ttype)
            xs = torch.zeros([ntrajectories, nsteps, self.nstates])
            xs[:, 0, :] = x0
            for i in range(nsteps - 1):
                xs[:, i + 1, :] = xs[:, i] + dt*self.time_derivative(integrator, xs[i:i+1, :], xs[i:i+1, :],
                                                                     t_sample, t_sample, dt)

            xs, t_sample, us = xs.detach().numpy(), t_sample.detach().numpy(), None
        else:
            t_sample = np.atleast_2d(t_sample)
            if t_sample.shape[0] == 1:
                t_sample = np.repeat(t_sample, ntrajectories, 0)
            else:
                assert t_sample.shape[0] == ntrajectories
            nsteps = t_sample.shape[-1]
            xs = np.zeros([ntrajectories, nsteps, self.nstates])
            us = np.zeros((ntrajectories, nsteps - 1, self.nstates))
            if references is None:
                references = [None] * ntrajectories

            for i in range(ntrajectories):
                xs[i], us[i] = self.simulate_trajectory(integrator=integrator, t_sample=t_sample[i], x0=x0[i], noise_std=noise_std, reference=references[i])

            if self.controller is None:
                us = None

            if len(t_sample.shape) == 1:
                t_sample = np.tile(t_sample, (ntrajectories, 1))
            t_sample = t_sample.reshape(ntrajectories, nsteps, 1)

        return xs, t_sample, us

    def _initial_condition_sampler(self, nsamples=1):
        return 2*torch.rand((nsamples, self.nstates), dtype=self.ttype) - 1
