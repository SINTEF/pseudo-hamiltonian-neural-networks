import torch
import torch.nn as nn
import numpy as np
from .dynamic_system_neural_network import DynamicSystemNN


__all__ = ['BaseNN', 'BaselineNN', 'BaselineSplitNN', 'HamiltonianNN',
           'ExternalForcesNN', 'R_NN', 'R_estimator',
           'PDEBaseNN', 'PDEBaselineNN', 'PDEIntegralNN', 'PDEExternalForcesNN',
           'PDEBaselineSplitNN', 'A_estimator', 'S_estimator',
           'load_baseline_model', 'store_baseline_model']


class BaseNN(torch.nn.Module):
    """
    Neural network with three hidden layers, where the first has
    Tanh-activation, the second has ReLU-activation and the third has
    linear activation. The network can take either system states or
    time or both as input. If it is expected to take neither states nor
    time as input, the network is replaced by trainable parameters.
    Independently of whether the network uses state and/or time or neither,
    it can be called with both state and time::

        pred = network(x=x, t=t)


    Parameters
    ----------
    nstates : int
        Number of states in a potential state input.
    noutputs : int
        Number of outputs from the last linear layer.
    hidden_dim : int
        Dimension of hidden layers.
    timedependent : bool
        If True, time input is expected.
    statedependent : bool
        If True, state input is expected.

    """

    def __init__(self, nstates, noutputs, hidden_dim,
                 timedependent, statedependent, ttype=torch.float32):
        super().__init__()
        self.nstates = nstates
        self.noutputs = noutputs
        self.hidden_dim = hidden_dim
        self.timedependent = timedependent
        self.statedependent = statedependent
        if not statedependent and not timedependent:
            self.model = nn.Parameter(torch.zeros(noutputs, dtype=ttype))
        else:
            input_dim = int(statedependent)*nstates + int(timedependent)
            linear1 = nn.Linear(input_dim, hidden_dim)
            linear2 = nn.Linear(hidden_dim, hidden_dim)
            linear3 = nn.Linear(hidden_dim, noutputs)

            for lin in [linear1, linear2, linear3]:
                nn.init.orthogonal_(lin.weight)

            self.model = nn.Sequential(
                linear1,
                nn.Tanh(),
                linear2,
                nn.ReLU(),
                linear3,
            )

        if timedependent and not statedependent:
            self.forward = self._forward_without_state
        elif statedependent and not timedependent:
            self.forward = self._forward_without_time
        elif not statedependent and not timedependent:
            self.forward = self._forward_without_state_or_time
        else:
            self.forward = self._forward_with_state_and_time

    def _forward_with_state_and_time(self, x=None, t=None):
        return self.model(torch.cat([x, t], dim=-1))

    def _forward_without_time(self, x=None, t=None):
        return self.model(x)

    def _forward_without_state(self, x=None, t=None):
        return self.model(t)

    def _forward_without_state_or_time(self, x=None, t=None):
        return self.model


class BaselineNN(BaseNN):
    """
    Neural network for estimating the right hand side of a set of
    dynamic system equations with three hidden layers, where the first
    has Tanh-activation, the second has ReLU-activation and the third has
    linear activation. The network can take either system states or
    time or both as input. If it is expected to take neither states nor
    time as input, the network is replaced by trainable parameters.
    Independently of whether the network uses state and/or time or neither,
    it can be called with both state and time::

        pred = network(x=x, t=t)

    The output dimension of the network is always *nstates*.

    Parameters
    ----------
    nstates : int
        Number of states in a potential state input.
    hidden_dim : int
        Dimension of hidden layers.
    timedependent : bool
        If True, time input is expected.
    statedependent : bool
        If True, state input is expected.

    """
    def __init__(self, nstates, hidden_dim, timedependent=True,
                 statedependent=True):
        super().__init__(nstates, nstates, hidden_dim,
                         timedependent, statedependent)


class HamiltonianNN(BaseNN):
    """
    Neural network for estimating a scalar function H(x),
    with three hidden layers, where the first has
    Tanh-activation, the second has ReLU-activation and the third has
    linear activation. The network takes system states as input, but
    can be called with both state and time::

        pred = network(x=x, t=t)

    The output dimension of the network is always 1.

    Parameters
    ----------
    nstates : int
        Number of states in a potential state input.
    hidden_dim : int
        Dimension of hidden layers.

    """
    def __init__(self, nstates, hidden_dim=100):
        super().__init__(nstates, 1, hidden_dim, False, True)


class ExternalForcesNN(BaseNN):
    """
    Neural network for estimating external forces of a pseudo-Hamiltonian
    system with three hidden layers, where the first has
    Tanh-activation, the second has ReLU-activation and the third has
    linear activation. The network can take either system states or
    time or both as input. Independently of whether the network uses
    state and/or time, it can be called with both state and time::

        pred = network(x=x, t=t)

    If neither time or state input is to be expected, the neural
    network is replaced by trainable parameters.

    Parameters
    ----------
    nstates : int
        Number of states in a potential state input.
    noutputs : int
        Number of external forces to estimate.
    timedependent : bool
        If True, time input is expected.
    statedependent : bool
        If True, state input is expected.
    external_forces_filter : listlike of ints or None, default None
        If None, *noutputs* == *nstates* must be true. In this case,
        one external force is estimated for each state. If
        *noutputs* != *nstates*, *external_forces_filter* must decribe
        which states external forces should be estimated for. Either,
        *external_forces_filter* must be a 1d liststructure of length
        nstates filled with 0 and 1, where 1 indicates that an external
        force should be estimated for state corresponding to that index.
        Alternatively, *external_forces_filter* can be an array of shape
        (nstates, noutputs) of 0s and 1s, such that when it is
        multiplied with the network outout of shape (noutputs,), the
        right output is applied to the correct state.
    ttype : torch type, default torch.float32

    """

    def __init__(self, nstates, noutputs, hidden_dim, timedependent,
                 statedependent, external_forces_filter=None,
                 ttype=torch.float32):
        super().__init__(nstates, noutputs, hidden_dim,
                         timedependent, statedependent)
        self.nstates = nstates
        self.noutputs = noutputs
        self.ttype = ttype

        self.external_forces_filter = self._format_external_forces_filter(
            external_forces_filter)

    def _forward_with_state_and_time(self, x=None, t=None):
        return self.model(torch.cat([x, t], dim=-1))@self.external_forces_filter

    def _forward_without_time(self, x=None, t=None):
        return self.model(x)@self.external_forces_filter

    def _forward_without_state(self, x=None, t=None):
        return self.model(t)@self.external_forces_filter

    def _forward_without_state_or_time(self, x=None, t=None):
        return self.model@self.external_forces_filter

    def _format_external_forces_filter(self, external_forces_filter):
        if external_forces_filter is None:
            assert self.noutputs == self.nstates, (
                f'noutputs ({self.noutputs}) != nstates ({self.nstates}) is '
                'not allowed when external_forces_filter is not provided.')
            return torch.eye(self.noutputs, dtype=self.ttype)

        if isinstance(external_forces_filter, (list, tuple)):
            external_forces_filter = np.array(external_forces_filter)
        
        if not isinstance(external_forces_filter, torch.Tensor):
            external_forces_filter = torch.tensor(external_forces_filter > 0)
        external_forces_filter = external_forces_filter.int()

        if ((len(external_forces_filter.shape) == 1) or
                (external_forces_filter.shape[-1] == 1)):

            external_forces_filter = external_forces_filter.flatten()
            assert external_forces_filter.shape[-1] == self.nstates, (
                'external_forces_filter is a vector of '
                f'length {external_forces_filter.shape[-1]} != nstates, but '
                f'({self.nstates}). external_forces_filter must be a '
                'vector of length nstates or a matrix of shape'
                '(nstates x noutputs).')
            expanded = torch.zeros((self.nstates, external_forces_filter.sum()),
                                   dtype=self.ttype)
            c = 0
            for i, e in enumerate(external_forces_filter):
                if e > 0:
                    expanded[i, c] = 1
                    c += 1
            return expanded.T

        assert external_forces_filter.shape == (self.nstates, self.noutputs), (
            f'external_forces_filter.shape == {external_forces_filter.shape}, but '
            'external_forces_filter must be a vector of length nstates or '
            'a matrix of shape (naffected_states x noutputs).')
        return external_forces_filter.clone().type(self.ttype).detach().T


class BaselineSplitNN(torch.nn.Module):
    """
    Composition of two neural networks for estimating the right hand
    side of a set of dynamic system equations. The networks have
    three hidden layers, where the first has Tanh-activation, the
    second has ReLU-activation and the third has linear activation.
    One network takes system states and the other takes time as input.
    The output of the composition is the sum of the outputs of the
    two networks::

        pred = network(x, t) = network_x(x) + network_t(t)

    Both networks are instantiated from the
    :py:class:`~.models.ExternalForcesNN` class, allowing adjustment
    of the number and location of non-zero contributions from each
    network.
    The output dimension of a BaselineSplitNN is always *nstates*.

    Parameters
    ----------
    nstates : int
        Number of states in a potential state input.
    hidden_dim : int
        Dimension of hidden layers. Equal for both networks.
    noutputs_x : int or None, default None
        Number of non-zero outputs to estimate with network_x(x)
    noutputs_t : int or None, default None
        Number of non-zero outputs to estimate with network_t(t)
    external_forces_filter_x : listlike of ints or None, default None
        If provided, this decides to which states the output of
        network_x is contributing. See :py:class:`~.models.ExternalForcesNN`
        for fruther description.
    external_forces_filter_t : listlike of ints or None, default None
        If provided, this decides to which states the output of
        network_t is contributing. See :py:class:`~.models.ExternalForcesNN`
        for fruther description.
    ttype : torch type, default torch.float32

    """
    def __init__(self, nstates, hidden_dim, noutputs_x=None,
                 noutputs_t=None, external_forces_filter_x=None,
                 external_forces_filter_t=None,
                 ttype=torch.float32):
        super().__init__()
        self.nstates = nstates
        self.hidden_dim = hidden_dim
        self.noutputs_x = nstates if noutputs_x is None else noutputs_x
        self.noutputs_t = nstates if noutputs_t is None else noutputs_t
        self.network_x = ExternalForcesNN(
            nstates, self.noutputs_x, hidden_dim, False,
            True, external_forces_filter_x, ttype)
        self.network_t = ExternalForcesNN(
            nstates, self.noutputs_t, hidden_dim, True,
            False, external_forces_filter_t, ttype)

    def forward(self, x, t):
        return self.network_x(x, t) + self.network_t(x, t)


class R_NN(BaseNN):
    '''
    Neural network for estimating the parameters of a damping matrix.
    with three hidden layers, where the first has Tanh-activation, the
    second has ReLU-activation and the third has linear activation.
    The network takes system states as input.

    When called with a batch input, the network returns a batch of
    matrices of size (*nstates*, *nstates*). All damping parameters are
    assumed to be positive.

    Parameters
    ----------
    nstates : int
        Number of states in a potential state input.
    hidden_dim : int
        Dimension of hidden layers.
    diagonal   : bool
        If True, only damping coefficients on the diagonal
        are estimated. If False, all nstates**2 entries in the
        R matrix are estimated.

    '''

    def __init__(self, nstates, hidden_dim, diagonal=False):
        if diagonal:
            noutputs = nstates
            self.forward = self._forward_diag
        else:
            noutputs = nstates**2
            self.forward = self._forward
        super().__init__(nstates, noutputs, hidden_dim, False, True)

        self.nstates = nstates

    def _forward_diag(self, x):
        return torch.diag_embed(self.model(x)**2).reshape(
            x.shape[0], self.nstates, self.nstates)

    def _forward(self, x):
        return (self.model(x)**2).reshape(
            x.shape[0], self.nstates, self.nstates)


class R_estimator(torch.nn.Module):
    '''
    Creates an estimator of a diagonal damping matrix of shape
    (nstates, nstates), where only a chosen set of states are damped.

    Parameters
    ----------
    state_is_damped : listlike of bools
        Listlike of boolean values of length nstates. If
        state_is_damped[i] is True, a learnable damping parameter is
        created for state i. If not, the damping of state i is set to
        zero.
    ttype : torch type, default torch.float32

    '''
    def __init__(self, state_is_damped, ttype=torch.float32):
        super().__init__()

        if not isinstance(state_is_damped, torch.Tensor):
            state_is_damped = torch.tensor(state_is_damped)

        self.state_is_damped = state_is_damped.bool()
        self.ttype = ttype

        nstates = self.state_is_damped.shape[0]

        self.rs = nn.Parameter(torch.zeros(
            torch.sum(self.state_is_damped), dtype=ttype))

        self.pick_rs = torch.zeros((nstates, torch.sum(self.state_is_damped)))
        c = 0
        for i in range(nstates):
            if self.state_is_damped[i]:
                self.pick_rs[i, c] = 1
                c += 1

    def forward(self, x=None):
        """
        Returns
        -------
        (N, N) tensor
            Damping matrix
        """
        return torch.diag(torch.abs(self.pick_rs)@(self.rs))

    def get_parameters(self):
        """
        Returns
        -------
        ndarray
            Damping parameters
        """
        return self.rs.detach().numpy()


class PeriodicPadding(nn.Module):
    def __init__(self, d):
        super(PeriodicPadding, self).__init__()
        self.d = d
    def forward(self, x):
        return torch.cat([x[..., -self.d:], x, x[..., :self.d]], dim=-1)
    

class Summation(nn.Module):
    def __init__(self):
        super(Summation, self).__init__()
    def forward(self, x):
        axis = tuple(range(1, np.ndim(x)))
        return x.sum(axis=axis, keepdims=True)


class PDEBaseNN(torch.nn.Module):
    """
    Description to be added
    
    """

    def __init__(self, nstates, noutputs, hidden_dim,
                 timedependent, statedependent, spacedependent=False, ttype=torch.float32):
        super().__init__()
        self.nstates = nstates
        self.noutputs = noutputs
        self.hidden_dim = hidden_dim
        self.timedependent = timedependent
        self.statedependent = statedependent
        self.spacedependent = spacedependent
        if not statedependent and not timedependent:
            input_dim = 1
            linear1 = nn.Linear(input_dim, hidden_dim)
            linear2 = nn.Linear(hidden_dim, hidden_dim)
            linear3 = nn.Linear(hidden_dim, noutputs)

            self.model = nn.Sequential(
                linear1,
                nn.Tanh(),
                linear2,
                nn.Tanh(),
                linear3,
            )
        else:
            input_dim = 1
            pad = PeriodicPadding(d=1)
            conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3)
            conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=None)
            summation = Summation()

            self.model = nn.Sequential(
                pad,
                conv1,
                nn.Tanh(),
                conv2,
                nn.Tanh(),
                conv3,
                summation,
            )

        if timedependent and spacedependent and statedependent:
            self.forward = self._forward_with_state_and_time_and_space
        elif timedependent and spacedependent and not statedependent:
            self.forward = self._forward_with_time_and_space
        elif timedependent and not spacedependent and statedependent:
            self.forward = self._forward_with_state_and_time
        elif not timedependent and spacedependent and statedependent:
            self.forward = self._forward_with_state_and_space
        elif timedependent and not spacedependent and not statedependent:
            self.forward = self._forward_with_time
        elif spacedependent and not timedependent and not statedependent:
            self.forward = self._forward_with_space
        elif statedependent and not timedependent and not spacedependent:
            self.forward = self._forward_with_state
        else:
            self.forward = self._forward_without_state_or_time_nor_space

    def _forward_with_state_and_time_and_space(self, x=None, t=None, xspatial=None):
        xsbasis = torch.cat([torch.sin(2*torch.pi/self.period*xspatial),
                            torch.cos(2*torch.pi/self.period*xspatial)], axis=-2)
        ts = t.repeat_interleave(x.shape[-1], dim=-1)
        return self.model(torch.cat([x, xsbasis, ts], dim=-2))
        
    def _forward_with_time_and_space(self, x=None, t=None, xspatial=None):
        xsbasis = torch.cat([torch.sin(2*torch.pi/self.period*xspatial),
                            torch.cos(2*torch.pi/self.period*xspatial)], axis=-2)
        ts = t.repeat_interleave(x.shape[-1], dim=-1)
        return self.model(torch.cat([xsbasis, ts], dim=-2))
        
    def _forward_with_state_and_time(self, x=None, t=None, xspatial=None):
        ts = t.repeat_interleave(x.shape[-1], dim=-1)
        return self.model(torch.cat([x, ts], dim=-2))
    
    def _forward_with_state_and_space(self, x=None, t=None, xspatial=None):
        xsbasis = torch.cat([torch.sin(2*torch.pi/self.period*xspatial),
                            torch.cos(2*torch.pi/self.period*xspatial)], axis=-2)
        return self.model(torch.cat([x, xsbasis], dim=-2))

    def _forward_with_time(self, x=None, t=None, xspatial=None):
        return self.model(t)

    def _forward_with_space(self, x=None, t=None, xspatial=None):
        xsbasis = torch.cat([torch.sin(2*torch.pi/self.period*xspatial),
                            torch.cos(2*torch.pi/self.period*xspatial)], axis = -2)
        return self.model(xsbasis)

    def _forward_with_state(self, x=None, t=None, xspatial=None):
        return self.model(x)

    def _forward_without_state_or_time_nor_space(self, x=None, t=None, xspatial=None):
        return self.model


class PDEBaselineNN(PDEBaseNN):
    """
    Description to be added

    """

    def __init__(self, nstates, hidden_dim=100, timedependent=False,
                 statedependent=True, spacedependent=False,
                 period=20, number_of_intermediate_outputs=4):
        noutputs = 1
        super().__init__(nstates, noutputs, hidden_dim, 
                         timedependent, statedependent, spacedependent=spacedependent)
        self.period = period
        self.number_of_intermediate_outputs = number_of_intermediate_outputs
        input_dim = int(statedependent) + int(timedependent) + 2*int(spacedependent)
        pad = PeriodicPadding(d=2)
        hidden_dim_pre = 20
        dnn_pre = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim_pre, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim_pre, hidden_dim_pre, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim_pre, hidden_dim_pre, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim_pre, hidden_dim_pre, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim_pre, number_of_intermediate_outputs*input_dim, kernel_size=1),
        )

        conv1 = nn.Conv1d(number_of_intermediate_outputs*input_dim, hidden_dim, kernel_size=5)
        conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        conv3 = nn.Conv1d(hidden_dim, noutputs, kernel_size=1, bias=None)

        self.model = nn.Sequential(
            pad,
            dnn_pre,
            conv1,
            nn.Tanh(),
            conv2,
            nn.Tanh(),
            conv3,
        )


class PDEIntegralNN(PDEBaseNN):
    """
    Neural network for ...

    """
    def __init__(self, nstates, hidden_dim=100):
        super().__init__(nstates, 1, hidden_dim, False, True, False)


class PDEExternalForcesNN(PDEBaseNN):
    """
    Neural network for ...

    """

    def __init__(self, nstates, hidden_dim=100, timedependent=False, spacedependent=True,
                 statedependent=False, period=20, ttype=torch.float32):
        noutputs = 1
        super().__init__(nstates, noutputs, hidden_dim, 
                         timedependent, statedependent, spacedependent=spacedependent)
        self.nstates = nstates
        self.noutputs = noutputs
        self.hidden_dim = hidden_dim
        self.spacedependent = spacedependent
        self.timedependent = timedependent
        self.statedependent = statedependent
        self.period = period
        self.ttype = ttype

        if not statedependent and not timedependent and not spacedependent:
            self.model = nn.Parameter(torch.tensor([0.], dtype=ttype))
        else:
            input_dim = int(statedependent) + int(timedependent) + 2*int(spacedependent)
            conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
            conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            conv3 = nn.Conv1d(hidden_dim, noutputs, kernel_size=1)

            self.model = nn.Sequential(
                conv1,
                nn.Tanh(),
                conv2,
                nn.Tanh(),
                conv3,
            )

            self.input_dim = input_dim


class PDEBaselineSplitNN(torch.nn.Module):
    """
    A model composed of a PDEBaselineNN model only depending on the
    state variables and an PDEExternalForcesNN model that can depend
    on space and/or time variables.

    """
    def __init__(self, nstates, hidden_dim=100, timedependent=False,
                 statedependent=True, spacedependent=False,
                 period=20, number_of_intermediate_outputs=4):
        super().__init__()
        self.nstates = nstates
        self.hidden_dim = hidden_dim
        self.timedependent = timedependent
        self.statedependent = statedependent
        self.spacedependent = spacedependent
        self.period = period
        self.number_of_intermediate_outputs = number_of_intermediate_outputs
        self.split = True
        self.baseline_nn = PDEBaselineNN(
            nstates, hidden_dim, False, True, False,
            number_of_intermediate_outputs=number_of_intermediate_outputs)
        self.external_forces = PDEExternalForcesNN(
            nstates, hidden_dim,
            timedependent, spacedependent, False,
            period)

    def forward(self, x, t, xspatial):
        return self.baseline_nn(x, t, xspatial) + self.external_forces(x, t, xspatial)


class A_estimator(torch.nn.Module):
    '''
    Creates an estimator of a symmetric convolution operator to apply to
    the left-hand side of the PDE system or the integral V.

    Parameters
    ----------
    kernel_size : int
    ttype : torch type, default torch.float32

    '''

    def __init__(self, kernel_size=3, ttype=torch.float32):
        super().__init__()

        self.ttype = ttype
        self.kernel_size = kernel_size
        d = int((kernel_size-1)/2)
        self.ls = torch.nn.Parameter(torch.zeros(d, dtype=self.ttype), requires_grad=True)

    def forward(self, x=None):
        """
        Returns
        -------
        (N, N) tensor
            Damping matrix
        """
        if self.kernel_size == 0:
            return torch.tensor([0], dtype=self.ttype).reshape(1,1,1)
        else:
            return torch.concat([self.ls,torch.tensor([1], dtype=self.ttype), self.ls]
                                ).reshape(1,1,self.kernel_size)


class S_estimator(torch.nn.Module):
    '''
    Creates an estimator of a skew-symmetric convolution operator to apply to
    the integral H.

    Parameters
    ----------
    kernel_size : int
    ttype : torch type, default torch.float32

    '''

    def __init__(self, kernel_size=3, ttype=torch.float32):
        super().__init__()

        self.ttype = ttype
        self.kernel_size = kernel_size
        if self.kernel_size > 1:
            d = int((kernel_size-3)/2)
            self.ls = torch.nn.Parameter(torch.zeros(d, dtype=self.ttype), requires_grad=True)

    def forward(self, x=None):
        """
        Returns
        -------
        (N, N) tensor
            Damping matrix
        """
        if self.kernel_size == 1 or self.kernel_size == 0:
            return torch.tensor([0], dtype=self.ttype).reshape(1,1,1)
        else:
            return torch.concat([-self.ls,torch.tensor([-1.,0.,1.], dtype=self.ttype),self.ls]
                                ).reshape(1,1,self.kernel_size)


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

    nstates = metadict['nstates']
    init_sampler = metadict['init_sampler']
    controller = metadict['controller']
    ttype = metadict['ttype']

    if 'external_forces_filter_x' in metadict['rhs_model'].keys():
        hidden_dim = metadict['rhs_model']['hidden_dim']
        noutputs_x = metadict['rhs_model']['noutputs_x']
        noutputs_t = metadict['rhs_model']['noutputs_t']
        external_forces_filter_x = metadict['rhs_model']['external_forces_filter_x']
        external_forces_filter_t = metadict['rhs_model']['external_forces_filter_t']
        rhs_model = BaselineSplitNN(
            nstates, hidden_dim, noutputs_x=noutputs_x,
            noutputs_t=noutputs_t,
            external_forces_filter_x=external_forces_filter_x,
            external_forces_filter_t=external_forces_filter_t,
            ttype=ttype)
    elif 'split' in metadict['rhs_model'].keys():
        hidden_dim = metadict['rhs_model']['hidden_dim']
        timedependent = metadict['rhs_model']['timedependent']
        statedependent = metadict['rhs_model']['statedependent']
        spacedependent = metadict['rhs_model']['spacedependent']
        period = metadict['rhs_model']['period']
        number_of_intermediate_outputs = metadict['rhs_model']['number_of_intermediate_outputs']
        rhs_model = PDEBaselineSplitNN(
            nstates, hidden_dim, timedependent, statedependent, spacedependent,
            period, number_of_intermediate_outputs)
    elif 'spacedependent' in metadict['rhs_model'].keys():
        hidden_dim = metadict['rhs_model']['hidden_dim']
        timedependent = metadict['rhs_model']['timedependent']
        statedependent = metadict['rhs_model']['statedependent']
        spacedependent = metadict['rhs_model']['spacedependent']
        period = metadict['rhs_model']['period']
        number_of_intermediate_outputs = metadict['rhs_model']['number_of_intermediate_outputs']
        rhs_model = PDEBaselineNN(
            nstates, hidden_dim, timedependent, statedependent, spacedependent,
            period, number_of_intermediate_outputs)
    else:
        hidden_dim = metadict['rhs_model']['hidden_dim']
        timedependent = metadict['rhs_model']['timedependent']
        statedependent = metadict['rhs_model']['statedependent']
        rhs_model = BaselineNN(
            nstates, hidden_dim, timedependent, statedependent)
    rhs_model.load_state_dict(metadict['rhs_model']['state_dict'])

    model = DynamicSystemNN(nstates, rhs_model=rhs_model,
                            init_sampler=init_sampler,
                            controller=controller, ttype=ttype)

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(metadict['traininginfo']['optimizer_state_dict'])

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

    metadict['nstates'] = model.nstates
    metadict['init_sampler'] = model._initial_condition_sampler
    metadict['controller'] = model.controller
    metadict['ttype'] = model.ttype

    if isinstance(model.rhs_model, BaselineNN):
        metadict['rhs_model'] = {}
        metadict['rhs_model']['hidden_dim'] = model.rhs_model.hidden_dim
        metadict['rhs_model']['timedependent'] = model.rhs_model.timedependent
        metadict['rhs_model']['statedependent'] = model.rhs_model.statedependent
        metadict['rhs_model']['state_dict'] = model.rhs_model.state_dict()

        metadict['traininginfo'] = {}
        metadict['traininginfo']['optimizer_state_dict'] = optimizer.state_dict()
        for key, value in kwargs.items():
            metadict['traininginfo'][key] = value

    elif isinstance(model.rhs_model, BaselineSplitNN):
        metadict['rhs_model'] = {}
        metadict['rhs_model']['hidden_dim'] = model.rhs_model.hidden_dim
        metadict['rhs_model']['noutputs_x'] = model.rhs_model.noutputs_x
        metadict['rhs_model']['noutputs_t'] = model.rhs_model.noutputs_t
        metadict['rhs_model']['external_forces_filter_x'] = model.rhs_model.network_x.external_forces_filter.T
        metadict['rhs_model']['external_forces_filter_t'] = model.rhs_model.network_t.external_forces_filter.T
        metadict['rhs_model']['state_dict'] = model.rhs_model.state_dict()

    elif isinstance(model.rhs_model, PDEBaselineNN) or isinstance(model.rhs_model, PDEBaselineSplitNN):
        metadict['rhs_model'] = {}
        metadict['rhs_model']['hidden_dim'] = model.rhs_model.hidden_dim
        metadict['rhs_model']['timedependent'] = model.rhs_model.timedependent
        metadict['rhs_model']['statedependent'] = model.rhs_model.statedependent
        metadict['rhs_model']['spacedependent'] = model.rhs_model.spacedependent
        metadict['rhs_model']['period'] = model.rhs_model.period
        metadict['rhs_model']['number_of_intermediate_outputs'] = model.rhs_model.number_of_intermediate_outputs
        metadict['rhs_model']['state_dict'] = model.rhs_model.state_dict()
        if isinstance(model.rhs_model, PDEBaselineSplitNN):
            metadict['rhs_model']['split'] = model.rhs_model.split

    metadict['traininginfo'] = {}
    metadict['traininginfo']['optimizer_state_dict'] = optimizer.state_dict()
    for key, value in kwargs.items():
        metadict['traininginfo'][key] = value

    torch.save(metadict, storepath)