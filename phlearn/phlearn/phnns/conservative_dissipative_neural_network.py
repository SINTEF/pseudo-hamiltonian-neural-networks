import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
import torch

from .dynamic_system_neural_network import DynamicSystemNN
from .models import PDEIntegralNN, A_estimator, S_estimator
from ..utils.utils import to_tensor

__all__ = ['ConservativeDissipativeNN', 'load_cdnn_model', 'store_cdnn_model']


class ConservativeDissipativeNN(DynamicSystemNN):
    """
    Description to be added

    """

    def __init__(self,
                 nstates,
                 kernel_sizes=[1,3,1,0],
                 structure_matrix=None,
                 dissipation_matrix=None,
                 symmetric_matrix=None,
                 hamiltonian_true=None,
                 grad_hamiltonian_true=None,
                 dissintegral_true=None,
                 grad_dissintegral_true=None,
                 external_forces_true=None,
                 hamiltonian_est=None,
                 dissintegral_est=None,
                 external_forces_est=None,
                 **kwargs):
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
        self.structure_matrix = structure_matrix
        self.dissipation_matrix = dissipation_matrix
        self.symmetric_matrix = symmetric_matrix
        self.hamiltonian_true = hamiltonian_true
        self.grad_hamiltonian_true = grad_hamiltonian_true
        self.dissintegral_true = dissintegral_true
        self.grad_dissintegral_true = grad_dissintegral_true
        self.external_forces_true = external_forces_true
        self.external_forces_est = external_forces_est

        if structure_matrix is None:
            self.S = S_estimator(kernel_size=self.kernel_sizes[1])
        elif not callable(structure_matrix):
            self.S = self._structure_matrix
        else:
            self.S = structure_matrix

        if dissipation_matrix is None:
            self.P = A_estimator(kernel_size=self.kernel_sizes[2])
        elif not callable(dissipation_matrix):
            self.P = self._dissipation_matrix
        else:
            self.P = dissipation_matrix
            
        if symmetric_matrix is None:
            self.D_flat = A_estimator(kernel_size=self.kernel_sizes[0])
        elif not callable(symmetric_matrix):
            self.D_flat = self._symmetric_matrix
        else:
            self.D_flat = symmetric_matrix

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

    def _symmetric_matrix(self, x=None):
        return to_tensor(self.symmetric_matrix, self.ttype)
        
    def _dissipation_matrix(self, x=None):
        return to_tensor(self.dissipation_matrix, self.ttype)
        
    def _structure_matrix(self, x=None):
        return to_tensor(self.structure_matrix, self.ttype)
    
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
        return (self.external_forces_original(x, t, xspatial) +
                (-self.P().sum()*self.dV_original(torch.tensor(((0,),),dtype=self.ttype))))
    
    def external_forces_correction(self):
        self.external_forces_original = self.external_forces
        self.external_forces = self._external_forces_corrected

    def _dH_hamiltonian_est(self, x):
        x = x.detach().requires_grad_()
        return torch.autograd.grad(self.hamiltonian(x).sum(), x,
                                   retain_graph=self.training,
                                   create_graph=self.training)[0]

    def _dH_hamiltonian_true(self, x):
        x = x.detach().requires_grad_()
        return torch.autograd.grad(self.hamiltonian(x).sum(), x,
                                   retain_graph=False,
                                   create_graph=False)[0].detach()

    def _dV_dissintegral_est(self, x):
        x = x.detach().requires_grad_()
        return torch.autograd.grad(self.dissintegral(x).sum(), x,
                                   retain_graph=self.training,
                                   create_graph=self.training)[0]

    def _dV_dissintegral_true(self, x):
        x = x.detach().requires_grad_()
        return torch.autograd.grad(self.dissintegral(x).sum(), x,
                                   retain_graph=False,
                                   create_graph=False)[0].detach()

    def _dV_corrected(self, x):
        return (self.dV_original(x) - 
                self.dV_original(torch.tensor(((0,),),dtype=self.ttype)))

    def dV_correction(self):
        self.dV_original = self.dV
        self.dV = self._dV_corrected
        
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
            d = int((self.kernel_sizes[1]-1)/2)
            dH_padded = torch.cat([dH[..., self.nstates-d:], dH, dH[..., :d]], dim=-1)
            dynamics += torch.nn.functional.conv1d(dH_padded, S)
        if self.kernel_sizes[2] > 0:
            P = self.P()
            if self.dissintegral is not None:
                dV = self.dV(x)
            else:
                dV = torch.zeros_like(x)
            d = int((self.kernel_sizes[2]-1)/2)
            dV_padded = torch.cat([dV[..., self.nstates-d:], dV, dV[..., :d]], dim=-1)
            dynamics -= torch.nn.functional.conv1d(dV_padded, P)
        if self.kernel_sizes[3] > 0:
            if self.external_forces is not None:
                dynamics += self.kernel_sizes[3]*self.external_forces(x, t, xspatial)
        return dynamics

    def lhs(self, dxdt):
        dxdt = to_tensor(dxdt, self.ttype)
        if self.kernel_sizes[0] == 1:
            return dxdt
        else:
            R = self.D_flat()
            d = int((self.kernel_sizes[0]-1)/2)
            dxdt_padded = torch.cat([dxdt[..., self.nstates-d:], dxdt, dxdt[..., :d]], dim=-1)
            Rdxdt = torch.nn.functional.conv1d(dxdt_padded, R)
            return Rdxdt

    def simulate_trajectory(self, integrator, t_sample, x0=None,
                            xspatial=None, noise_std=0., reference=None):
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
        M = x0.shape[0]
        if x0 is None:
            x0 = self._initial_condition_sampler(1)
        if not integrator:
            if self.kernel_sizes[0] == 1:
                if xspatial is not None:
                    x_dot = lambda t, x: self._x_dot(
                                torch.tensor(x.reshape(1, x.shape[-1]),
                                            dtype=self.ttype),
                                torch.tensor(np.array(t).reshape((1, 1)),
                                            dtype=self.ttype),
                                xspatial=torch.tensor(np.array(xspatial).reshape(1, xspatial.shape[-1]),
                                            dtype=self.ttype)
                                ).detach().numpy().flatten()
                else:
                    x_dot = lambda t, x: self._x_dot(
                                torch.tensor(x.reshape(1, x.shape[-1]),
                                            dtype=self.ttype),
                                torch.tensor(np.array(t).reshape((1, 1)),
                                            dtype=self.ttype)
                                ).detach().numpy().flatten()
            else:
                d = int((self.kernel_sizes[0]-1)/2)
                D_flat = self.D_flat().detach().numpy()
                diagonals = np.concatenate([D_flat[0,:,(d+1):], D_flat[0], D_flat[0,:,:-(d+1)]], axis=1).T.repeat(M, axis=1)
                offsets = np.concatenate([np.arange(-M+1,-M+1+d),np.arange(-d,d+1),np.arange(M-d,M)])
                D = diags(diagonals, offsets, (M,M)).toarray()
                if xspatial is not None:
                    x_dot = lambda t, x: np.linalg.solve(
                                D,
                                self._x_dot(
                                torch.tensor(x.reshape(1, x.shape[-1]),
                                            dtype=self.ttype),
                                torch.tensor(np.array(t).reshape((1, 1)),
                                            dtype=self.ttype),
                                xspatial=torch.tensor(np.array(xspatial).reshape(1, xspatial.shape[-1]),
                                            dtype=self.ttype)
                                ).detach().numpy().flatten())
                else:
                    x_dot = lambda t, x: np.linalg.solve(
                                D,
                                self._x_dot(
                                torch.tensor(x.reshape(1, x.shape[-1]),
                                            dtype=self.ttype),
                                torch.tensor(np.array(t).reshape((1, 1)),
                                            dtype=self.ttype)
                                ).detach().numpy().flatten())
            out_ivp = solve_ivp(fun=x_dot, t_span=(t_sample[0], t_sample[-1]),
                                y0=x0.detach().numpy().flatten(),
                                t_eval=t_sample, rtol=1e-10)
            xs = out_ivp['y'].T
        else:
            t_sample = to_tensor(t_sample, self.ttype)
            if not integrator and self.controller is not None:
                integrator = 'rk4'
                print('Warning: Since the system contains a controller, '
                      'the RK4 integrator is used to simulate the trajectory '
                      'instead of solve_ivp.')
            elif integrator.lower() not in ['euler', 'rk4']:
                print('Warning: Only explicit integrators Euler and RK4 or no '
                      'integrator (False) allowed for integration. Ignoring '
                      f'integrator {integrator} and using RK4.')
                integrator = 'rk4'

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
                        xs[i + 1, :] = xs[i, :] + dt*self.time_derivative(
                            integrator, xs[i:i+1, :], xs[i:i+1, :],
                            t_step, t_step, dt, xspatial=torch.tensor(np.array(xspatial).reshape(1,xspatial.shape[-1]),
                                                                      dtype=self.ttype))
                else:
                    for i, t_step in enumerate(t_sample[:-1]):
                        t_step = torch.squeeze(t_step).reshape(-1, 1)
                        dt = t_sample[i + 1] - t_step
                        I = np.eye(M)
                        xs[i + 1, :] = xs[i, :] + dt*self.time_derivative(
                            integrator, xs[i:i+1, :], xs[i:i+1, :],
                            t_step, t_step, dt)
            else:
                d = int((self.kernel_sizes[0]-1)/2)
                D_flat = self.D_flat().detach().numpy()
                diagonals = np.concatenate([D_flat[0,:,(d+1):], D_flat[0], D_flat[0,:,:-(d+1)]], axis=1).T.repeat(M, axis=1)
                offsets = np.concatenate([np.arange(-M+1,-M+1+d),np.arange(-d,d+1),np.arange(M-d,M)])
                D = diags(diagonals, offsets, (M,M)).toarray()
                if xspatial is not None:
                    for i, t_step in enumerate(t_sample[:-1]):
                        t_step = torch.squeeze(t_step).reshape(-1, 1)
                        dt = t_sample[i + 1] - t_step
                        xs[i + 1, :] = xs[i, :] + dt*np.linalg.solve(
                            D,
                            self.time_derivative(
                            integrator, xs[i:i+1, :], xs[i:i+1, :],
                            t_step, t_step, dt, xspatial=torch.tensor(np.array(xspatial).reshape(1, xspatial.shape[-1]),
                                                                      dtype=self.ttype)).detach().T).T
                else:
                    for i, t_step in enumerate(t_sample[:-1]):
                        t_step = torch.squeeze(t_step).reshape(-1, 1)
                        dt = t_sample[i + 1] - t_step
                        xs[i + 1, :] = xs[i, :] + dt*np.linalg.solve(
                            D,
                            self.time_derivative(
                            integrator, xs[i:i+1, :], xs[i:i+1, :],
                            t_step, t_step, dt).detach().T).T
            xs = xs.detach().numpy()

        return xs, None


def load_cdnn_model(modelpath):
    """
    Loads a :py:class:`PseudoHamiltonianNN` that has been stored using the
    :py:func:`store_phnn_model`. Assumes that the hamiltonian function
    has either been provided or has been modeled with a
    :py:class:`~.models.hamiltonianNN`, that the external forces
    has either been provided or modelled with a
    :py:class:`~.models.ExternalForcesNN`, and that the dissipation has
    either been provided or been modelled with a
    :py:class:`~.models.R_estimator` or a :py:class:`~.models.R_NN`.

    Parameters
    ----------
    modelpath : str

    Returns
    -------
    model : PseudoHamiltonianNN
    optimizer : torch.optim.Adam
    metadict : dict
        Contains information about the model and training details.

    """

    metadict = torch.load(modelpath)

    model = metadict['model']

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(metadict['traininginfo']['optimizer_state_dict'])

    return model, optimizer, metadict


def store_cdnn_model(storepath, model, optimizer, **kwargs):
    """
    Stores a :py:class:`ConservativeDissipativeNNNN` with additional information
    to disc. The stored model can be read into memory again with
    :py:func:`load_cdnn_model`.

    Parameters
    ----------
    storepath : str
    model : ConservativeDissipativeNN
    optimizer : torch optimizer
    * * kwargs : dict
        Contains additional information about for instance training
        hyperparameters and loss values.

    """

    metadict = {}
    metadict['model'] = model
    # metadict['nstates'] = model.nstates
    # metadict['kernel_sizes'] = model.kernel_sizes
    # metadict['structure_matrix'] = model.S().detach() # model.structure_matrix
    # metadict['dissipation_matrix'] = model.P().detach() # model.dissipation_matrix
    # metadict['symmetric_matrix'] = model.D_flat().detach() # model.symmetric_matrix
    metadict['symmetric_matrix'] = None
    # #metadict['structure_matrix'] = model.structure_matrix
    # #metadict['dissipation_matrix'] = model.dissipation_matrix
    # #metadict['symmetric_matrix'] = model.symmetric_matrix
    # metadict['hamiltonian_provided'] = model.hamiltonian_provided
    # metadict['grad_hamiltonian_provided'] = model.hamiltonian_provided
    # metadict['dissintegral_provided'] = model.dissintegral_provided
    # metadict['grad_dissintegral_provided'] = model.grad_dissintegral_provided
    # metadict['external_forces_provided'] = model.external_forces_provided
    # metadict['init_sampler'] = model._initial_condition_sampler
    # metadict['ttype'] = model.ttype

    metadict['traininginfo'] = {}
    metadict['traininginfo']['optimizer_state_dict'] = optimizer.state_dict()
    for key, value in kwargs.items():
        metadict['traininginfo'][key] = value

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