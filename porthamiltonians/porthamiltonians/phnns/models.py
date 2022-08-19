
import torch
import torch.nn as nn

from .dynamic_system_neural_network import DynamicSystemNN


__all__ = ['BaseNN', 'BaselineNN', 'HamiltonianNN', 'ExternalPortNN', 'R_NN',
           'R_estimator', 'load_baseline_model', 'store_baseline_model']


class BaseNN(torch.nn.Module):
    """
    Neural network with three hidden layers, where the first has
    Tanh-activation, the second has ReLU-activation and the third has
    linear activation. The network can take either system states, or
    time or both as input. Independently of whether the network uses
    state and/or time, it can be called with both state and time::

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
                 timedependent, statedependent):
        super().__init__()
        self.nstates = nstates
        self.noutputs = noutputs
        self.hidden_dim = hidden_dim
        self.timedependent = timedependent
        self.statedependent = statedependent
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

        if not statedependent:
            self.forward = self._forward_without_state
        elif not timedependent:
            self.forward = self._forward_without_time
        else:
            self.forward = self._forward_with_state_and_time

    def _forward_with_state_and_time(self, x=None, t=None):
        return self.model(torch.cat([x, t], dim=-1))

    def _forward_without_time(self, x=None, t=None):
        return self.model(x)

    def _forward_without_state(self, x=None, t=None):
        return self.model(t)


class BaselineNN(BaseNN):
    """
    Neural network for estimating the right hand side of a set of
    dynamic system equations with three hidden layers, where the first
    has Tanh-activation, the second has ReLU-activation and the third has
    linear activation. The network can take either system states, or
    time or both as input. Independently of whether the network uses
    state and/or time, it can be called with both state and time::

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
    def __init__(self, nstates, hidden_dim, timedependent,
                 statedependent):
        super().__init__(nstates, nstates, hidden_dim,
                         timedependent, statedependent)


class HamiltonianNN(BaseNN):
    """
    Neural network for estimating a Hamiltonian function H(X)
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
    def __init__(self, nstates, hidden_dim):
        super().__init__(nstates, 1, hidden_dim, False, True)


class ExternalPortNN(BaseNN):
    """
    Neural network for estimating esternal ports of a port-Hamiltonian
    system with three hidden layers, where the first has
    Tanh-activation, the second has ReLU-activation and the third has
    linear activation. The network can take either system states, or
    time or both as input. Independently of whether the network uses
    state and/or time, it can be called with both state and time::

        pred = network(x=x, t=t)

    The output dimension of the network is always 1.

    Parameters
    ----------
    nstates : int
        Number of states in a potential state input.
    noutputs : int
        Number of external ports to estimate.
    timedependent : bool
        If True, time input is expected.
    statedependent : bool
        If True, state input is expected.
    external_port_filter : listlike of ints or None, default None
        If None, *noutputs* == *nstates* must be true. In this case,
        one external port is estimated for each state. If
        *noutputs* != *nstates*, *external_port_filter* must decribe
        which states external ports should be estimated for. Either,
        *external_port_filter* must be a 1d liststructure of length
        nstates filled with 0 and 1, where 1 indicates that an external
        port should be estimated for state corresponding to that index.
        Alternatively, *external_port_filter* can be an array of shape
        (nstates, noutputs) of 0s and 1s, such that when it is
        multiplied with the network outout of shape (noutputs,), the
        right output is applied to the correct state.
    ttype : torch type, default torch.float32

    """

    def __init__(self, nstates, noutputs, hidden_dim, timedependent,
                 statedependent, external_port_filter=None,
                 ttype=torch.float32):
        super().__init__(nstates, noutputs, hidden_dim,
                         timedependent, statedependent)
        self.nstates = nstates
        self.noutputs = noutputs
        self.ttype = ttype

        self.external_port_filter = self._format_external_port_filter(
            external_port_filter)

    def _forward_with_state_and_time(self, x=None, t=None):
        return self.model(torch.cat([x, t], dim=-1))@self.external_port_filter

    def _forward_without_time(self, x=None, t=None):
        return self.model(x)@self.external_port_filter

    def _forward_without_state(self, x=None, t=None):
        return self.model(t)@self.external_port_filter

    def _format_external_port_filter(self, external_port_filter):
        if external_port_filter is None:
            assert self.noutputs == self.nstates, (
                f'noutputs ({self.noutputs}) != nstates ({self.nstates}) is '
                'not allowed when external_port_filter is not provided.')
            return torch.eye(self.noutputs, dtype=self.ttype)

        if not isinstance(external_port_filter, torch.Tensor):
            external_port_filter = torch.tensor(external_port_filter > 0)
        external_port_filter = external_port_filter.int()

        if ((len(external_port_filter.shape) == 1) or
                (external_port_filter.shape[-1] == 1)):

            external_port_filter = external_port_filter.flatten()
            assert external_port_filter.shape[-1] == self.nstates, (
                'external_port_filter is a vector of '
                f'length {external_port_filter.shape[-1]} != nstates, but '
                f'({self.nstates}). external_port_filter must be a '
                'vector of length nstates or a matrix of shape'
                '(nstates x noutputs).')
            expanded = torch.zeros((self.nstates, external_port_filter.sum()),
                                   dtype=self.ttype)
            c = 0
            for i, e in enumerate(external_port_filter):
                if e > 0:
                    expanded[i, c] = 1
                    c += 1
            return expanded.T

        assert external_port_filter.shape == (self.nstates, self.noutputs), (
            f'external_port_filter.shape == {external_port_filter.shape}, but '
            'external_port_filter must be a vector of length nstates or '
            'a matrix of shape (naffected_states x noutputs).')
        return torch.tensor(external_port_filter, dtype=self.ttype).T


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


def load_baseline_model(modelpath):
    """
    Loads a :py:class:`BaslineNN` that has been stored using the
    :py:meth:`store_baseline_model`.

    Parameters
    ----------
    modelpath : str

    Returns
    -------
    model : BaslineNN
    optimizer : torch.optim.Adam
    metadict : dict
        Contains information about the model and training details.

    """

    metadict = torch.load(modelpath)

    nstates = metadict['nstates']
    init_sampler = metadict['init_sampler']
    controller = metadict['controller']
    ttype = metadict['ttype']
    hidden_dim = metadict['rhs_model']['hidden_dim']
    timedependent = metadict['rhs_model']['timedependent']
    statedependent = metadict['rhs_model']['statedependent']

    rhs_model = BaselineNN(nstates, hidden_dim, timedependent, statedependent)
    rhs_model.load_state_dict(metadict['rhs_model']['state_dict'])

    model = DynamicSystemNN(nstates, rhs_model=rhs_model,
                            init_sampler=init_sampler,
                            controller=controller, ttype=ttype)

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(metadict['traininginfo']['optimizer_state_dict'])

    return model, optimizer, metadict


def store_baseline_model(storepath, model, optimizer, **kwargs):
    """
    Stores a :py:class:`BaslineNN` with additional information
    to disc. The stored model can be read into memory again with
    :py:meth:`load_baseline_model`.

    Parameters
    ----------
    storepath : str
    model : BaslineNN
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

    metadict['rhs_model'] = {}
    metadict['rhs_model']['hidden_dim'] = model.rhs_model.hidden_dim
    metadict['rhs_model']['timedependent'] = model.rhs_model.timedependent
    metadict['rhs_model']['statedependent'] = model.rhs_model.statedependent
    metadict['rhs_model']['state_dict'] = model.rhs_model.state_dict()

    metadict['traininginfo'] = {}
    metadict['traininginfo']['optimizer_state_dict'] = optimizer.state_dict()
    for key, value in kwargs.items():
        metadict['traininginfo'][key] = value

    torch.save(metadict, storepath)
