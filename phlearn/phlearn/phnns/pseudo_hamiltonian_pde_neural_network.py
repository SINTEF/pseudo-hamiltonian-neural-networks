import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
import torch

from .dynamic_system_neural_network import DynamicSystemNN
from .pde_models import PDEIntegralNN, A_estimator, S_estimator
from ..utils.utils import to_tensor

__all__ = ['PseudoHamiltonianPDENN']

class PseudoHamiltonianPDENN(DynamicSystemNN):
    """
    Implements a pseudo-Hamiltonian neural network abiding to the
    spatially discretized pseudo-Hamiltonian PDE formulation::

        A*dx/dt = S*grad[H(x)] - R*grad[V(x)] + F(x, t, xspatial)

    where x is the system state, A is a symmetric matrix, S is a skew-symmetric
    matrix, R is the symmetric dissipation matrix, H and V are discretized
    integrals of the system, F is the external force depending on state, time
    and space.

    What is x here is usually u in the literature, and xspatial is x. We use x
    for the state to be consistent with the ODE case.

    It is possible to provide function estimators like neural networks to model
    H, V and F, as well as A, S and R. All estimators must subclass
    torch.nn.Module, such that gradients can be recorded with PyTorch.

    If either of A, R, S, H, V or F are known, they can be provided and used in
    favor of estimators. Note that R, H and F must be functions both taking as
    input and returning tensors, and that the gradients of H(x) must be
    available through autograd unless the true gradient is provided.

    Parameters
    ----------
        nstates : int
            Number of system states M after discretization, i.e. the number of
            spatial discretization points, including only one of the boundaries
            of the periodic domain.

        kernel_sizes : List or ndarray of four integers, default [1, 3, 1, 0]
            All the integers should be 0 or odd, the first three should be
            maximum M (nstates), and the last should be 0 or 1.

        skewsymmetric_matrix : (1, 1, kernel_size[1]) ndarray or callable,
            default None
            Corresponds to band of width kernel_size[1] of the S matrix.
            If None, S is assumed to be given by S_estimator with kernel size
            kernel_size[1], i.e. a trainable skew-symmetric Toeplitz matrix
            with a diagonal band of kernel_size[1], and periodicity imposed.

        dissipation_matrix : (1, 1, kernel_size[2]) ndarray or callable,
            default None
            Corresponds to band of width kernel_size[2] of the R matrix.
            If None, R is assumed to be given by A_estimator with kernel size
            kernel_size[2], i.e. a trainable symmetric Toeplitz matrix with a
            diagonal band of kernel_size[2], and periodicity imposed.

        lhs_matrix : (1, 1, kernel_size[0]) ndarray or callable,
            default None
            Corresponds to band of width kernel_size[0] of the A matrix. If
            None, A is assumed to be given by A_estimator with kernel size
            kernel_size[0], i.e. a trainable symmetric Toeplitz matrix with a
            diagonal band of kernel_size[0], and periodicity imposed.

        hamiltonian_true : callable, default None
            The known Hamiltonian H of the system. Callable taking a
            torch tensor input of shape (nsamples, nstates) and
            returning a torch tensor of shape (nsamples, 1). If the
            gradient of the Hamiltonian is not provided, the gradient of
            this function will be computed by torch and used instead.
            If hamiltonian_true is provided, hamiltonian_est will be
            ignored.

        grad_hamiltonian_true : callable, default None
            The known gradient of the Hamiltonian. Callable taking a
            tensor input of shape (nsamples, nstates) and returning a
            tensor of shape (nsamples, nstates).

        dissintegral_true : callable, default None
            The known dissiaption integral V of the system. Callable taking a
            torch tensor input of shape (nsamples, nstates) and
            returning a torch tensor of shape (nsamples, 1). If the
            gradient of the integral is not provided, the gradient of
            this function will be computed by torch and used instead.
            If dissintegral_true is provided, dissintegral_est will be
            ignored.

        grad_dissintegral_true : callable, default None
            The known gradient of the dissipation integral. Callable taking a
            tensor input of shape (nsamples, nstates) and returning a
            tensor of shape (nsamples, nstates).

        external_forces_true : callable, default None
            The external ports affecting the system. Callable taking two or
            three tensors as input, x, t and optionally xspatial, of shape
            (nsamples, nstates), (nsamples, 1), (nsamples, nstates), resp.
            Returning a tensor of shape (nsamples, nstates). If
            external_forces_true is provided, external_forces_est will be
            ignored.

        hamiltonian_est : callable, default None
            Estimator for the Hamiltonian. Takes a tensor of shape
            (nsamples, nstates) as input, returning a tensor of shape
            (nsamples, 1).

        dissintegral_est : callable, default None
            Estimator for the dissipation integral. Takes a tensor of shape
            (nsamples, nstates) as input, returning a tensor of shape
            (nsamples, 1).

        external_forces_est : callable, default None
            Estimator for the external ports. Takes as input from zero to three
            tensors of shape (nsamples, nstates), (nsamples, 1),
            (nsamples, nstates), resp. Returns a tensor of shape
            (nsamples, nstates).

    """

    def __init__(
        self,
        nstates,
        kernel_sizes=[1, 3, 1, 0],
        skewsymmetric_matrix=None,
        dissipation_matrix=None,
        lhs_matrix=None,
        hamiltonian_true=None,
        grad_hamiltonian_true=None,
        dissintegral_true=None,
        grad_dissintegral_true=None,
        external_forces_true=None,
        hamiltonian_est=None,
        dissintegral_est=None,
        external_forces_est=None,
        **kwargs,
    ):
        super().__init__(nstates, **kwargs)
        self.hamiltonian = None
        self.dissintegral = None
        self.external_forces = None
        self.hamiltonian_provided = False
        self.grad_hamiltonian_provided = False
        self.dissintegral_provided = False
        self.grad_dissintegral_provided = False
        self.external_forces_provided = False
        self.spacedependent = False
        self.nstates = nstates
        self.kernel_sizes = kernel_sizes
        self.skewsymmetric_matrix = skewsymmetric_matrix
        self.dissipation_matrix = dissipation_matrix
        self.lhs_matrix = lhs_matrix
        self.hamiltonian_true = hamiltonian_true
        self.grad_hamiltonian_true = grad_hamiltonian_true
        self.dissintegral_true = dissintegral_true
        self.grad_dissintegral_true = grad_dissintegral_true
        self.external_forces_true = external_forces_true
        self.external_forces_est = external_forces_est

        if skewsymmetric_matrix is None:
            self.S = S_estimator(kernel_size=self.kernel_sizes[1])
        elif not callable(skewsymmetric_matrix):
            self.S = self._skewsymmetric_matrix
        else:
            self.S = skewsymmetric_matrix

        if dissipation_matrix is None:
            self.R = A_estimator(kernel_size=self.kernel_sizes[2])
        elif not callable(dissipation_matrix):
            self.R = self._dissipation_matrix
        else:
            self.R = dissipation_matrix

        if lhs_matrix is None:
            self.A = A_estimator(kernel_size=self.kernel_sizes[0])
        elif not callable(lhs_matrix):
            self.A = self._lhs_matrix
        else:
            self.A = lhs_matrix

        if hamiltonian_true is not None:
            if grad_hamiltonian_true is None:
                self.hamiltonian = hamiltonian_true
                self.dH = self._dH_hamiltonian_true
            else:
                self.hamiltonian = self._hamiltonian_true
                self.dH = self._grad_hamiltonian_true
                self.grad_hamiltonian_provided = True
            self.hamiltonian_provided = True
        elif grad_hamiltonian_true is not None:
            self.dH = self._grad_hamiltonian_true
            self.grad_hamiltonian_provided = True
        else:
            if hamiltonian_est is not None:
                self.hamiltonian = hamiltonian_est
            else:
                self.hamiltonian = PDEIntegralNN(nstates=self.nstates)
            self.dH = self._dH_hamiltonian_est

        if dissintegral_true is not None:
            if grad_dissintegral_true is None:
                self.dissintegral = dissintegral_true
                self.dV = self._dV_dissintegral_true
            else:
                self.dissintegral = self._dissintegral_true
                self.dV = self._grad_dissintegral_true
                self.grad_dissintegral_provided = True
            self.dissintegral_provided = True
        elif grad_dissintegral_true is not None:
            self.dV = self._grad_dissintegral_true
            self.grad_dissintegral_provided = True
        else:
            if dissintegral_est is not None:
                self.dissintegral = dissintegral_est
            else:
                self.dissintegral = PDEIntegralNN(nstates=self.nstates)
            self.dV = self._dV_dissintegral_est

        if external_forces_true is not None:
            self.external_forces = self._external_forces_true
            self.external_forces_provided = True
        elif external_forces_est is not None:
            self.external_forces = self._external_forces_est
            self.spacedependent = external_forces_est.spacedependent

    def _lhs_matrix(self, x=None):
        return to_tensor(self.lhs_matrix, self.ttype)

    def _dissipation_matrix(self, x=None):
        return to_tensor(self.dissipation_matrix, self.ttype)

    def _skewsymmetric_matrix(self, x=None):
        return to_tensor(self.skewsymmetric_matrix, self.ttype)

    def _hamiltonian_true(self, x):
        return self.hamiltonian_true(x).detach()

    def _grad_hamiltonian_true(self, x):
        return self.grad_hamiltonian_true(x).detach()

    def _dissintegral_true(self, x):
        return self.dissintegral_true(x).detach()

    def _grad_dissintegral_true(self, x):
        return self.grad_dissintegral_true(x).detach()

    def _external_forces_true(self, x, t, xspatial=None):
        return self.external_forces_true(x, t, xspatial).detach()

    def _external_forces_est(self, x, t, xspatial=None):
        return self.external_forces_est(x, t, xspatial)

    def _external_forces_corrected(self, x, t, xspatial=None):
        return self.external_forces_original(x, t, xspatial) + (
            -self.R().sum() * self.dV_original(torch.tensor(((0,),), dtype=self.ttype))
        )

    def _dH_hamiltonian_est(self, x):
        x = x.detach().requires_grad_()
        return torch.autograd.grad(
            self.hamiltonian(x).sum(),
            x,
            retain_graph=self.training,
            create_graph=self.training,
        )[0]

    def _dH_hamiltonian_true(self, x):
        x = x.detach().requires_grad_()
        return torch.autograd.grad(
            self.hamiltonian(x).sum(), x, retain_graph=False, create_graph=False
        )[0].detach()

    def _dV_dissintegral_est(self, x):
        x = x.detach().requires_grad_()
        return torch.autograd.grad(
            self.dissintegral(x).sum(),
            x,
            retain_graph=self.training,
            create_graph=self.training,
        )[0]

    def _dV_dissintegral_true(self, x):
        x = x.detach().requires_grad_()
        return torch.autograd.grad(
            self.dissintegral(x).sum(), x, retain_graph=False, create_graph=False
        )[0].detach()

    def _dV_corrected(self, x):
        return self.dV_original(x) - self.dV_original(torch.tensor(((0,),), dtype=self.ttype))

    def _x_dot(self, x, t, u=None, xspatial=None):
        x = to_tensor(x, self.ttype)
        t = to_tensor(t, self.ttype)
        u = to_tensor(u, self.ttype)

        dynamics = torch.zeros_like(x)

        if self.kernel_sizes[1] > 0:
            S = self.S()
            if self.hamiltonian is not None:
                dH = self.dH(x)
            else:
                dH = torch.zeros_like(x)
            d = int((self.kernel_sizes[1] - 1) / 2)
            dH_padded = torch.cat([dH[..., self.nstates - d :], dH, dH[..., :d]], dim=-1)
            dynamics += torch.nn.functional.conv1d(dH_padded, S)
        if self.kernel_sizes[2] > 0:
            R = self.R()
            if self.dissintegral is not None:
                dV = self.dV(x)
            else:
                dV = torch.zeros_like(x)
            d = int((self.kernel_sizes[2] - 1) / 2)
            dV_padded = torch.cat([dV[..., self.nstates - d :], dV, dV[..., :d]], dim=-1)
            dynamics -= torch.nn.functional.conv1d(dV_padded, R)
        if self.kernel_sizes[3] > 0:
            if self.external_forces is not None:
                dynamics += self.kernel_sizes[3] * self.external_forces(x, t, xspatial)
        return dynamics

    def dV_correction(self):
        """
        Amends the grad of the dissipation integral so that it is zero for the
        zero solution.
        """

        self.dV_original = self.dV
        self.dV = self._dV_corrected

    def external_forces_correction(self):
        """
        Amends the external force term to correpond with the dissipation term
        being zero for the zero solution but the full PHNN model is unchanged.
        """

        self.external_forces_original = self.external_forces
        self.external_forces = self._external_forces_corrected

    def lhs(self, dxdt):
        """
        Applies the A matrix to the left-hand side of the PHNN formulation.

        Parameter
        ----------
        dxdt : torch tensor of dimension (nsamples, 1, nstates)
        dx/dt in the PHNN formulation

        Returns:
        ----------
        Rdxdt : torch tensor of dimension (nsamples, 1, nstates)
        A*dx/dt in the PHNN formulation
        """

        if self.kernel_sizes[0] == 1:
            return dxdt
        else:
            A = self.A()
            d = int((self.kernel_sizes[0] - 1) / 2)
            dxdt_padded = torch.cat([dxdt[..., self.nstates - d :], dxdt, dxdt[..., :d]], dim=-1)
            Adxdt = torch.nn.functional.conv1d(dxdt_padded, A)
            return Adxdt

    def simulate_trajectory(self, integrator, t_sample, x0=None, xspatial=None):
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
        t_sample : (T,) tensor or ndarray
            Times at which the trajectory is sampled.
        x0 : (M,) tensor or ndarray, default None
            Initial condition. If None, an initial condition is sampled
            with the internal sampler.
        xspatial : (M,) tensor or ndarray, default None
            The spatial points. Required if rhs_model depends on these.

        Returns
        -------
        xs : (T, M) tensor
            The solutions at the time steps given by t_sample
        us : None
            Placeholder for control output
        """

        x0 = to_tensor(x0)
        M = x0.shape[0]
        if x0 is None:
            x0 = self._initial_condition_sampler(1)
        if not integrator:
            if self.kernel_sizes[0] == 1:
                if xspatial is not None:

                    def x_dot(t, x):
                        return (
                            self._x_dot(
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

                    def x_dot(t, x):
                        return (
                            self._x_dot(
                                torch.tensor(x.reshape(1, x.shape[-1]), dtype=self.ttype),
                                torch.tensor(np.array(t).reshape((1, 1)), dtype=self.ttype),
                            )
                            .detach()
                            .numpy()
                            .flatten()
                        )

            else:
                d = int((self.kernel_sizes[0] - 1) / 2)
                A = self.A().detach().numpy()
                diagonals = np.concatenate(
                    [A[0, :, (d + 1) :], A[0], A[0, :, : -(d + 1)]],
                    axis=1,
                ).T.repeat(M, axis=1)
                offsets = np.concatenate(
                    [
                        np.arange(-M + 1, -M + 1 + d),
                        np.arange(-d, d + 1),
                        np.arange(M - d, M),
                    ]
                )
                D = diags(diagonals, offsets, (M, M)).toarray()
                if xspatial is not None:

                    def x_dot(t, x):
                        return np.linalg.solve(
                            D,
                            self._x_dot(
                                torch.tensor(x.reshape(1, x.shape[-1]), dtype=self.ttype),
                                torch.tensor(np.array(t).reshape((1, 1)), dtype=self.ttype),
                                xspatial=torch.tensor(
                                    np.array(xspatial).reshape(1, xspatial.shape[-1]),
                                    dtype=self.ttype,
                                ),
                            )
                            .detach()
                            .numpy()
                            .flatten(),
                        )

                else:

                    def x_dot(t, x):
                        return np.linalg.solve(
                            D,
                            self._x_dot(
                                torch.tensor(x.reshape(1, x.shape[-1]), dtype=self.ttype),
                                torch.tensor(np.array(t).reshape((1, 1)), dtype=self.ttype),
                            )
                            .detach()
                            .numpy()
                            .flatten(),
                        )

            out_ivp = solve_ivp(
                fun=x_dot,
                t_span=(t_sample[0], t_sample[-1]),
                y0=x0.detach().numpy().flatten(),
                t_eval=t_sample,
                rtol=1e-10,
            )
            xs = out_ivp["y"].T
        else:
            t_sample = to_tensor(t_sample, self.ttype)
            if not integrator and self.controller is not None:
                integrator = "rk4"
                print(
                    "Warning: Since the system contains a controller, "
                    "the RK4 integrator is used to simulate the trajectory "
                    "instead of solve_ivp."
                )
            elif integrator.lower() not in ["euler", "rk4"]:
                print(
                    "Warning: Only explicit integrators Euler and RK4 or no "
                    "integrator (False) allowed for integration. Ignoring "
                    f"integrator {integrator} and using RK4."
                )
                integrator = "rk4"

            nsteps = t_sample.shape[0]
            x0 = x0.reshape(1, x0.shape[-1])
            xs = torch.zeros([nsteps, x0.shape[-1]])
            xs[0, :] = x0

            if self.kernel_sizes[0] == 1:
                if xspatial is not None:
                    for i, t_step in enumerate(t_sample[:-1]):
                        t_step = torch.squeeze(t_step).reshape(-1, 1)
                        dt = t_sample[i + 1] - t_step
                        I = np.eye(M)
                        xs[i + 1, :] = xs[i, :] + dt * self.time_derivative(
                            integrator,
                            xs[i : i + 1, :],
                            xs[i : i + 1, :],
                            t_step,
                            t_step,
                            dt,
                            xspatial=torch.tensor(
                                np.array(xspatial).reshape(1, xspatial.shape[-1]),
                                dtype=self.ttype,
                            ),
                        )
                else:
                    for i, t_step in enumerate(t_sample[:-1]):
                        t_step = torch.squeeze(t_step).reshape(-1, 1)
                        dt = t_sample[i + 1] - t_step
                        I = np.eye(M)
                        xs[i + 1, :] = xs[i, :] + dt * self.time_derivative(
                            integrator,
                            xs[i : i + 1, :],
                            xs[i : i + 1, :],
                            t_step,
                            t_step,
                            dt,
                        )
            else:
                d = int((self.kernel_sizes[0] - 1) / 2)
                A = self.A().detach().numpy()
                diagonals = np.concatenate(
                    [A[0, :, (d + 1) :], A[0], A[0, :, : -(d + 1)]],
                    axis=1,
                ).T.repeat(M, axis=1)
                offsets = np.concatenate(
                    [
                        np.arange(-M + 1, -M + 1 + d),
                        np.arange(-d, d + 1),
                        np.arange(M - d, M),
                    ]
                )
                D = diags(diagonals, offsets, (M, M)).toarray()
                if xspatial is not None:
                    for i, t_step in enumerate(t_sample[:-1]):
                        t_step = torch.squeeze(t_step).reshape(-1, 1)
                        dt = t_sample[i + 1] - t_step
                        xs[i + 1, :] = (
                            xs[i, :]
                            + dt
                            * np.linalg.solve(
                                D,
                                self.time_derivative(
                                    integrator,
                                    xs[i : i + 1, :],
                                    xs[i : i + 1, :],
                                    t_step,
                                    t_step,
                                    dt,
                                    xspatial=torch.tensor(
                                        np.array(xspatial).reshape(1, xspatial.shape[-1]),
                                        dtype=self.ttype,
                                    ),
                                )
                                .detach()
                                .T,
                            ).T
                        )
                else:
                    for i, t_step in enumerate(t_sample[:-1]):
                        t_step = torch.squeeze(t_step).reshape(-1, 1)
                        dt = t_sample[i + 1] - t_step
                        xs[i + 1, :] = (
                            xs[i, :]
                            + dt
                            * np.linalg.solve(
                                D,
                                self.time_derivative(
                                    integrator,
                                    xs[i : i + 1, :],
                                    xs[i : i + 1, :],
                                    t_step,
                                    t_step,
                                    dt,
                                )
                                .detach()
                                .T,
                            ).T
                        )
            xs = xs.detach().numpy()

        return xs, None


def load_cdnn_model(modelpath):
    """
    Loads a :py:class:`PseudoHamiltonianPDENN` that has been stored using the
    :py:func:`store_cdnn_model`. Assumes that the hamiltonian function and the
    dissipative integral has either been provided or has been modeled with a
    :py:class:`~.models.PDEIntegralNN`, that the external forces
    has either been provided or modelled with a
    :py:class:`~.models.PDEExternalForcesNN`.

    Parameters
    ----------
    modelpath : str

    Returns
    -------
    model : PseudoHamiltonianPDENN
    optimizer : torch.optim.Adam
    metadict : dict
        Contains information about the model and training details.

    """

    metadict = torch.load(modelpath)

    model = metadict["model"]

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(metadict["traininginfo"]["optimizer_state_dict"])

    return model, optimizer, metadict


def store_cdnn_model(storepath, model, optimizer, **kwargs):
    """
    Stores a :py:class:`PseudoHamiltonianPDENN` with additional information to disc.
    The stored model can be read into memory again with :py:func:`load_cdnn_model`.

    Parameters
    ----------
    storepath : str
    model : PseudoHamiltonianPDENN
    optimizer : torch optimizer
    * * kwargs : dict
        Contains additional information about for instance training
        hyperparameters and loss values.

    """

    metadict = {}
    metadict["model"] = model
    # metadict['nstates'] = model.nstates
    # metadict['kernel_sizes'] = model.kernel_sizes
    # metadict['skewsymmetric_matrix'] = model.S().detach() # model.skewsymmetric_matrix
    # metadict['dissipation_matrix'] = model.R().detach() # model.dissipation_matrix
    # metadict['lhs_matrix'] = model.A().detach() # model.lhs_matrix
    metadict["lhs_matrix"] = None
    # #metadict['skewsymmetric_matrix'] = model.skewsymmetric_matrix
    # #metadict['dissipation_matrix'] = model.dissipation_matrix
    # #metadict['lhs_matrix'] = model.lhs_matrix
    # metadict['hamiltonian_provided'] = model.hamiltonian_provided
    # metadict['grad_hamiltonian_provided'] = model.hamiltonian_provided
    # metadict['dissintegral_provided'] = model.dissintegral_provided
    # metadict['grad_dissintegral_provided'] = model.grad_dissintegral_provided
    # metadict['external_forces_provided'] = model.external_forces_provided
    # metadict['init_sampler'] = model._initial_condition_sampler
    # metadict['ttype'] = model.ttype

    metadict["traininginfo"] = {}
    metadict["traininginfo"]["optimizer_state_dict"] = optimizer.state_dict()
    for key, value in kwargs.items():
        metadict["traininginfo"][key] = value

    # metadict['hamiltonian'] = {}
    # metadict['grad_hamiltonian'] = {}
    # metadict['dissintegral'] = {}
    # metadict['grad_dissintegral'] = {}
    # metadict['external_forces'] = {}

    # if model.hamiltonian_provided:
    #     metadict['hamiltonian']['true'] = model.hamiltonian_true
    #     metadict['hamiltonian']['hidden_dim'] = None
    #     metadict['hamiltonian']['state_dict'] = None
    # else:
    #     metadict['hamiltonian']['true'] = None
    #     metadict['hamiltonian']['hidden_dim'] = model.hamiltonian.hidden_dim
    #     metadict['hamiltonian']['state_dict'] = model.hamiltonian.state_dict()

    # if model.grad_hamiltonian_provided:
    #     metadict['grad_hamiltonian']['true'] = model.grad_hamiltonian_true
    # else:
    #     metadict['grad_hamiltonian']['true'] = None

    # if model.dissintegral_provided:
    #     metadict['dissintegral']['true'] = model.dissintegral_true
    #     metadict['dissintegral']['hidden_dim'] = None
    #     metadict['dissintegral']['state_dict'] = None
    # else:
    #     metadict['dissintegral']['true'] = None
    #     metadict['dissintegral']['hidden_dim'] = model.dissintegral.hidden_dim
    #     metadict['dissintegral']['state_dict'] = model.dissintegral.state_dict()

    # if model.grad_dissintegral_provided:
    #     metadict['grad_dissintegral']['true'] = model.grad_dissintegral_true
    # else:
    #     metadict['grad_dissintegral']['true'] = None

    # if model.external_forces_provided:
    #     metadict['external_forces']['true'] = model.external_forces_true
    #     metadict['external_forces']['hidden_dim'] = None
    #     metadict['external_forces']['timedependent'] = None
    #     metadict['external_forces']['spacedependent'] = None
    #     metadict['external_forces']['statedependent'] = None
    #     metadict['external_forces']['period'] = None
    #     metadict['external_forces']['ttype'] = None
    #     metadict['external_forces']['state_dict'] = None
    # else:
    #     metadict['external_forces']['true'] = None
    #     metadict['external_forces']['hidden_dim'] = model.external_forces.hidden_dim
    #     metadict['external_forces']['timedependent'] = model.external_forces.timedependent
    #     metadict['external_forces']['statedependent'] = model.external_forces.statedependent
    #     metadict['external_forces']['spacedependent'] = model.external_forces.spacedependent
    #     metadict['external_forces']['period'] = model.external_forces.period
    #     metadict['external_forces']['ttype'] = model.external_forces.ttype
    #     metadict['external_forces']['state_dict'] = model.external_forces.state_dict()

    torch.save(metadict, storepath)
