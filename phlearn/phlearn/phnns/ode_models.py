import torch
import torch.nn as nn
import numpy as np

__all__ = ['BaseNN', 'BaselineNN', 'BaselineSplitNN', 'HamiltonianNN',
           'ExternalForcesNN', 'R_NN', 'R_estimator']

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