import numpy as np
from scipy.integrate import solve_ivp
import torch

from ..utils.derivatives import time_derivative
from ..utils.utils import to_tensor
from .ode_models import BaselineNN, BaselineSplitNN
from .pde_models import PDEBaselineNN, PDEBaselineSplitNN

__all__ = ['DynamicSystemNN', 'load_baseline_model', 'store_baseline_model']

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
        PseudoHamiltonianNN.sample_trajectory when a controller is
        provided, the Runge-Kutta-4 method will be used for integration
        in favor of SciPy's solve_ivp.

    init_sampler : callable, default None
        Function for sampling initial conditions. Callable taking a
        number specifying the number of inital conditions to sample, M,
        as input and returning a tensor of shape (M, nstates) with
        inital conditions for the system. This sampler is used when
        calling :py:meth:`~DynamicSystemNN.simulate_trajectory` and
        :py:meth:`~DynamicSystemNN.simulate_trajectories` if no initial
        condition is provided. If not provided, initial conditions are
        uniformly sampled from (0, 1).

    ttype : torch type, default torch.float32

    """

    def __init__(
        self,
        nstates,
        rhs_model=None,
        init_sampler=None,
        controller=None,
        ttype=torch.float32,
    ):
        super().__init__()
        self.ttype = ttype
        self.nstates = nstates
        self.controller = controller
        self.rhs_model = rhs_model
        if init_sampler is not None:
            self._initial_condition_sampler = init_sampler
        self.x_dot = self._x_dot

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

        return time_derivative(integrator, self.x_dot, *args, **kwargs)

    def simulate_trajectory(
        self,
        integrator,
        t_sample,
        x0=None,
        xspatial=None,
        noise_std=0.0,
        reference=None,
    ):
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
        reference : phlearn.control.Reference, default None
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
            if xspatial is not None:
                x_dot = (
                    lambda t, x: self._x_dot(
                        torch.tensor(x.reshape(1, x.shape[-1]), dtype=self.ttype),
                        torch.tensor(np.array(t).reshape((1, 1)), dtype=self.ttype),
                        xspatial=torch.tensor(
                            np.array(xspatial).reshape(1, xspatial.shape[-1]),
                            dtype=self.ttype,
                        ),
                    )
                    .detach()
                    .numpy()
                    .flatten()
                )
            else:
                x_dot = (
                    lambda t, x: self._x_dot(
                        torch.tensor(x.reshape(1, x.shape[-1]), dtype=self.ttype),
                        torch.tensor(np.array(t).reshape((1, 1)), dtype=self.ttype),
                    )
                    .detach()
                    .numpy()
                    .flatten()
                )
            out_ivp = solve_ivp(
                fun=x_dot,
                t_span=(t_sample[0], t_sample[-1]),
                y0=x0.detach().numpy().flatten(),
                t_eval=t_sample,
                rtol=1e-10,
            )
            xs = out_ivp["y"].T
            us = None
        else:
            t_sample = to_tensor(t_sample, self.ttype)
            if not integrator and self.controller is not None:
                integrator = "rk4"
                print(
                    "Warning: Since the system contains a controller, "
                    "the rk4 integrator is used to simulate the trajectory "
                    "instead of solve_ivp."
                )
            elif integrator.lower() not in ["euler", "rk4"]:
                print(
                    "Warning: Only explicit integrators euler and rk4 or no "
                    "integrator (False) allowed for inference. Ignoring "
                    f"integrator {integrator} and using rk4."
                )
                integrator = "rk4"

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

            if xspatial is not None:
                for i, t_step in enumerate(t_sample[:-1]):
                    t_step = torch.squeeze(t_step).reshape(-1, 1)
                    if self.controller is not None:
                        u = to_tensor(self.controller(xs[i, :], t_step), self.ttype)
                        us[i, :] = u
                    dt = t_sample[i + 1] - t_step
                    xs[i + 1, :] = xs[i, :] + dt * self.time_derivative(
                        integrator,
                        xs[i : i + 1, :],
                        xs[i : i + 1, :],
                        t_step,
                        t_step,
                        dt,
                        u,
                        xspatial=torch.tensor(
                            np.array(xspatial).reshape(1, xspatial.shape[-1]),
                            dtype=self.ttype,
                        ),
                    )
            else:
                for i, t_step in enumerate(t_sample[:-1]):
                    t_step = torch.squeeze(t_step).reshape(-1, 1)
                    if self.controller is not None:
                        u = to_tensor(self.controller(xs[i, :], t_step), self.ttype)
                        us[i, :] = u
                    dt = t_sample[i + 1] - t_step
                    xs[i + 1, :] = xs[i, :] + dt * self.time_derivative(
                        integrator,
                        xs[i : i + 1, :],
                        xs[i : i + 1, :],
                        t_step,
                        t_step,
                        dt,
                        u,
                    )
            xs = xs.detach().numpy()
            if self.controller is not None:
                us = us.detach().numpy()
            else:
                us = None

        return xs, us

    def simulate_trajectories(
        self, ntrajectories, integrator, t_sample, x0=None, noise_std=0, references=None
    ):
        """
        Calls :py:meth:`~DynamicSystemNN.simulate_trajectory`
        *ntrajectories* times.

        Parameters
        ----------
        integrator : str or False
            Specifies which solver to use during simulation. If False,
            the problem is left to scipy's solve_ivp. If 'euler', the system
            is simulated with the forward Euler method.
            If 'midpoint', 'rk4' or 'srk4' the system is simulated with
            the classic Runge-Kutta-4 method.
        t_sample : (T, 1) or (ntrajectories, T, 1) tensor or ndarray
            Times at which the trajectory is sampled.
        x0 : (ntrajectories, N) tensor or ndarray, default None
            Initial condition. If None, an initial condition is sampled
            with the internal sampler.
        noise_std : number, default 0.
            Standard deviation of Gaussian white noise added to the
            samples of the trajectory.
        references : list of phlearn.control.Reference, default
        None
            If the system has a controller a list of ntrajectories
            reference objects may be passed.

        Returns
        -------
        xs : (ntrajectories, T, N) tensor
        t_sample : (ntrajectories, T, 1) tensor
        us : (ntrajectories, T, N) tensor or None

        """

        if integrator in ("euler", "rk4") and self.controller is None:
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
                xs[:, i + 1, :] = xs[:, i] + dt * self.time_derivative(
                    integrator,
                    xs[i : i + 1, :],
                    xs[i : i + 1, :],
                    t_sample,
                    t_sample,
                    dt,
                )

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
                    integrator=integrator,
                    t_sample=t_sample[i],
                    x0=x0[i],
                    noise_std=noise_std,
                    reference=references[i],
                )

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

    def lhs(self, dxdt):
        return dxdt

    def _x_dot(self, x, t, u=None, xspatial=None):
        x = to_tensor(x, self.ttype)
        t = to_tensor(t, self.ttype)
        u = to_tensor(u, self.ttype)

        if xspatial is not None:
            xspatial = to_tensor(xspatial, self.ttype)
            dynamics = self.rhs_model(x, t, xspatial)
        else:
            dynamics = self.rhs_model(x, t)
        if u is not None:
            dynamics += u
        return dynamics

    def _initial_condition_sampler(self, nsamples=1):
        return 2 * torch.rand((nsamples, self.nstates), dtype=self.ttype) - 1


def load_baseline_model(modelpath):
    """
    Loads a :py:class:`BaslineNN` or a :py:class:`BaselineSplitNN`
    that has been stored using the :py:meth:`store_baseline_model`.

    Parameters
    ----------
    modelpath : str

    Returns
    -------
    model : BaslineNN, BaselineSplitNN
    optimizer : torch.optim.Adam
    metadict : dict
        Contains information about the model and training details.

    """

    metadict = torch.load(modelpath)

    nstates = metadict["nstates"]
    init_sampler = metadict["init_sampler"]
    controller = metadict["controller"]
    ttype = metadict["ttype"]

    if "external_forces_filter_x" in metadict["rhs_model"].keys():
        hidden_dim = metadict["rhs_model"]["hidden_dim"]
        noutputs_x = metadict["rhs_model"]["noutputs_x"]
        noutputs_t = metadict["rhs_model"]["noutputs_t"]
        external_forces_filter_x = metadict["rhs_model"]["external_forces_filter_x"]
        external_forces_filter_t = metadict["rhs_model"]["external_forces_filter_t"]
        rhs_model = BaselineSplitNN(
            nstates,
            hidden_dim,
            noutputs_x=noutputs_x,
            noutputs_t=noutputs_t,
            external_forces_filter_x=external_forces_filter_x,
            external_forces_filter_t=external_forces_filter_t,
            ttype=ttype,
        )
    elif "split" in metadict["rhs_model"].keys():
        hidden_dim = metadict["rhs_model"]["hidden_dim"]
        timedependent = metadict["rhs_model"]["timedependent"]
        statedependent = metadict["rhs_model"]["statedependent"]
        spacedependent = metadict["rhs_model"]["spacedependent"]
        period = metadict["rhs_model"]["period"]
        number_of_intermediate_outputs = metadict["rhs_model"][
            "number_of_intermediate_outputs"
        ]
        rhs_model = PDEBaselineSplitNN(
            nstates,
            hidden_dim,
            timedependent,
            statedependent,
            spacedependent,
            period,
            number_of_intermediate_outputs,
        )
    elif "spacedependent" in metadict["rhs_model"].keys():
        hidden_dim = metadict["rhs_model"]["hidden_dim"]
        timedependent = metadict["rhs_model"]["timedependent"]
        statedependent = metadict["rhs_model"]["statedependent"]
        spacedependent = metadict["rhs_model"]["spacedependent"]
        period = metadict["rhs_model"]["period"]
        number_of_intermediate_outputs = metadict["rhs_model"][
            "number_of_intermediate_outputs"
        ]
        rhs_model = PDEBaselineNN(
            nstates,
            hidden_dim,
            timedependent,
            statedependent,
            spacedependent,
            period,
            number_of_intermediate_outputs,
        )
    else:
        hidden_dim = metadict["rhs_model"]["hidden_dim"]
        timedependent = metadict["rhs_model"]["timedependent"]
        statedependent = metadict["rhs_model"]["statedependent"]
        rhs_model = BaselineNN(nstates, hidden_dim, timedependent, statedependent)
    rhs_model.load_state_dict(metadict["rhs_model"]["state_dict"])

    model = DynamicSystemNN(
        nstates,
        rhs_model=rhs_model,
        init_sampler=init_sampler,
        controller=controller,
        ttype=ttype,
    )

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(metadict["traininginfo"]["optimizer_state_dict"])

    return model, optimizer, metadict


def store_baseline_model(storepath, model, optimizer, **kwargs):
    """
    Stores a :py:class:`BaslineNN` or a :py:class:`BaselineSplitNN`
    with additional information to disc. The stored model can be
    read into memory again with :py:meth:`load_baseline_model`.

    Parameters
    ----------
    storepath : str
    model : BaselineNN, BaselineSplitNN, PDEBaselineNN
    optimizer : torch optimizer
    * * kwargs : dict
        Contains additional information about for instance training
        hyperparameters and loss values.

    """

    metadict = {}

    metadict["nstates"] = model.nstates
    metadict["init_sampler"] = model._initial_condition_sampler
    metadict["controller"] = model.controller
    metadict["ttype"] = model.ttype

    if isinstance(model.rhs_model, BaselineNN):
        metadict["rhs_model"] = {}
        metadict["rhs_model"]["hidden_dim"] = model.rhs_model.hidden_dim
        metadict["rhs_model"]["timedependent"] = model.rhs_model.timedependent
        metadict["rhs_model"]["statedependent"] = model.rhs_model.statedependent
        metadict["rhs_model"]["state_dict"] = model.rhs_model.state_dict()

        metadict["traininginfo"] = {}
        metadict["traininginfo"]["optimizer_state_dict"] = optimizer.state_dict()
        for key, value in kwargs.items():
            metadict["traininginfo"][key] = value

    elif isinstance(model.rhs_model, BaselineSplitNN):
        metadict["rhs_model"] = {}
        metadict["rhs_model"]["hidden_dim"] = model.rhs_model.hidden_dim
        metadict["rhs_model"]["noutputs_x"] = model.rhs_model.noutputs_x
        metadict["rhs_model"]["noutputs_t"] = model.rhs_model.noutputs_t
        metadict["rhs_model"][
            "external_forces_filter_x"
        ] = model.rhs_model.network_x.external_forces_filter.T
        metadict["rhs_model"][
            "external_forces_filter_t"
        ] = model.rhs_model.network_t.external_forces_filter.T
        metadict["rhs_model"]["state_dict"] = model.rhs_model.state_dict()

    elif isinstance(model.rhs_model, PDEBaselineNN) or isinstance(
        model.rhs_model, PDEBaselineSplitNN
    ):
        metadict["rhs_model"] = {}
        metadict["rhs_model"]["hidden_dim"] = model.rhs_model.hidden_dim
        metadict["rhs_model"]["timedependent"] = model.rhs_model.timedependent
        metadict["rhs_model"]["statedependent"] = model.rhs_model.statedependent
        metadict["rhs_model"]["spacedependent"] = model.rhs_model.spacedependent
        metadict["rhs_model"]["period"] = model.rhs_model.period
        metadict["rhs_model"][
            "number_of_intermediate_outputs"
        ] = model.rhs_model.number_of_intermediate_outputs
        metadict["rhs_model"]["state_dict"] = model.rhs_model.state_dict()
        if isinstance(model.rhs_model, PDEBaselineSplitNN):
            metadict["rhs_model"]["split"] = model.rhs_model.split

    metadict["traininginfo"] = {}
    metadict["traininginfo"]["optimizer_state_dict"] = optimizer.state_dict()
    for key, value in kwargs.items():
        metadict["traininginfo"][key] = value

    torch.save(metadict, storepath)
