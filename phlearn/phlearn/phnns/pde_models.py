import torch
import torch.nn as nn
import numpy as np

__all__ = ['CentralPadding', 'ForwardPadding', 'Summation',
           'PDEBaseNN', 'PDEBaselineNN', 'PDEIntegralNN', 'PDEExternalForcesNN',
           'PDEBaselineSplitNN', 'A_estimator', 'S_estimator']

class CentralPadding(nn.Module):
    """
    Module that performs periodic even padding on the last dimension of the input tensor.
    The tensor is padded by adding the first d elements to the end and the last d elements before
    the beginning of the tensor.

    Parameters
    ----------
    d : int
        The number of elements to pad on each side of the tensor.
    """
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x):
        return torch.cat([x[..., -self.d :], x, x[..., : self.d]], dim=-1)


class ForwardPadding(nn.Module):
    """
    Module that performs periodic forward padding on the last dimension of the input tensor.
    The tensor is padded by adding the first d elements to the end of the tensor.

    Parameters
    ----------
    d : int
        The number of elements to pad at the end of the tensor.
    """
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x):
        return torch.cat([x, x[..., : self.d]], dim=-1)


class Summation(nn.Module):
    """
    Module that performs summation along all dimensions except the batch dimension.
    It computes the sum of elements across each dimension and keeps the dimensionality
    intact by using the keepdims=True argument.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        axis = tuple(range(1, np.ndim(x)))
        return x.sum(axis=axis, keepdims=True)


class PDEBaseNN(torch.nn.Module):
    """
    Base neural network module for solving partial differential equations (PDEs).
    The network can handle various combinations of time, state, and spatial inputs.
    It consists of multiple convolutional layers, linear layers, and activation functions.
    The specific architecture depends on the input dependencies specified during initialization.
    
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
    spacedependent : bool, optional
        If True, spatial input is expected. Default is False.
    ttype : torch.dtype, optional
        Data type for the time input tensor. Default is torch.float32.

    """
    def __init__(
        self,
        nstates,
        noutputs,
        hidden_dim,
        timedependent,
        statedependent,
        spacedependent=False,
        ttype=torch.float32,
    ):
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
            pad = ForwardPadding(d=1)
            conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=2)
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
        xsbasis = torch.cat(
            [
                torch.sin(2 * torch.pi / self.period * xspatial),
                torch.cos(2 * torch.pi / self.period * xspatial),
            ],
            axis=-2,
        )
        ts = t.repeat_interleave(x.shape[-1], dim=-1)
        return self.model(torch.cat([x, xsbasis, ts], dim=-2))

    def _forward_with_time_and_space(self, x=None, t=None, xspatial=None):
        xsbasis = torch.cat(
            [
                torch.sin(2 * torch.pi / self.period * xspatial),
                torch.cos(2 * torch.pi / self.period * xspatial),
            ],
            axis=-2,
        )
        ts = t.repeat_interleave(x.shape[-1], dim=-1)
        return self.model(torch.cat([xsbasis, ts], dim=-2))

    def _forward_with_state_and_time(self, x=None, t=None, xspatial=None):
        ts = t.repeat_interleave(x.shape[-1], dim=-1)
        return self.model(torch.cat([x, ts], dim=-2))

    def _forward_with_state_and_space(self, x=None, t=None, xspatial=None):
        xsbasis = torch.cat(
            [
                torch.sin(2 * torch.pi / self.period * xspatial),
                torch.cos(2 * torch.pi / self.period * xspatial),
            ],
            axis=-2,
        )
        return self.model(torch.cat([x, xsbasis], dim=-2))

    def _forward_with_time(self, x=None, t=None, xspatial=None):
        return self.model(t)

    def _forward_with_space(self, x=None, t=None, xspatial=None):
        xsbasis = torch.cat(
            [
                torch.sin(2 * torch.pi / self.period * xspatial),
                torch.cos(2 * torch.pi / self.period * xspatial),
            ],
            axis=-2,
        )
        return self.model(xsbasis)

    def _forward_with_state(self, x=None, t=None, xspatial=None):
        return self.model(x)

    def _forward_without_state_or_time_nor_space(self, x=None, t=None, xspatial=None):
        return self.model


class PDEBaselineNN(PDEBaseNN):
    """
    Neural network for estimating the right-hand side of spatially discretized PDEs.
    It is based on the PDEBaseNN architecture and includes additional layers and parameters
    specific to the baseline model.

    Parameters
    ----------
    nstates : int
        Number of states in a potential state input.
    hidden_dim : int, optional
        Dimension of hidden layers. Default is 100.
    timedependent : bool, optional
        If True, time input is expected. Default is False.
    statedependent : bool, optional
        If True, state input is expected. Default is True.
    spacedependent : bool, optional
        If True, spatial input is expected. Default is False.
    period : int, optional
        Period value used in the model. Default is 20.
    number_of_intermediate_outputs : int, optional
        Number of intermediate outputs. Default is 4.
    """
    def __init__(
        self,
        nstates,
        hidden_dim=100,
        timedependent=False,
        statedependent=True,
        spacedependent=False,
        period=20,
        number_of_intermediate_outputs=4,
    ):
        noutputs = 1
        super().__init__(
            nstates,
            noutputs,
            hidden_dim,
            timedependent,
            statedependent,
            spacedependent=spacedependent,
        )
        self.period = period
        self.number_of_intermediate_outputs = number_of_intermediate_outputs
        input_dim = int(statedependent) + int(timedependent) + 2 * int(spacedependent)
        pad = CentralPadding(d=2)
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
            nn.Conv1d(
                hidden_dim_pre,
                number_of_intermediate_outputs * input_dim,
                kernel_size=1,
            ),
        )

        conv1 = nn.Conv1d(
            number_of_intermediate_outputs * input_dim, hidden_dim, kernel_size=5
        )
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
    Neural network for estimating the quadrature approximation of an integral over the spatial
    discretization points where the states of a PDE is evaluated.
    It is based on the PDEBaseNN architecture and takes state variables as input and outputs a
    scalar function.

    Parameters
----------
    nstates : int
        Number of states in the input.
    hidden_dim : int, optional
    Dimension of hidden layers. Default is 100.

    """
    def __init__(self, nstates, hidden_dim=100):
        super().__init__(nstates, 1, hidden_dim, False, True, False)


class PDEExternalForcesNN(PDEBaseNN):
    """
    Neural network for estimating the external forces in a pseudo-Hamiltonian PDE.
    It is based on the PDEBaseNN architecture but sets up a network where the kernel size is 1 for
    all convolutional layers, including the first one, so that it cannot learn approximations of
    spatial derivatives. The network can take any combination of state, space and time as input.

    Parameters
    ----------
    nstates : int
        Number of states in a potential state input.
    hidden_dim : int, optional
        Dimension of hidden layers. Default is 100.
    timedependent : bool, optional
        If True, time input is expected. Default is False.
    spacedependent : bool, optional
        If True, spatial input is expected. Default is True.
    statedependent : bool, optional
        If True, state input is expected. Default is False.
    period : int, optional
        Period value used in the model. Default is 20.
    ttype : torch.dtype, optional
        Data type for the time input tensor. Default is torch.float32.

    """

    def __init__(
        self,
        nstates,
        hidden_dim=100,
        timedependent=False,
        spacedependent=True,
        statedependent=False,
        period=20,
        ttype=torch.float32,
    ):
        noutputs = 1
        super().__init__(
            nstates,
            noutputs,
            hidden_dim,
            timedependent,
            statedependent,
            spacedependent=spacedependent,
        )
        self.nstates = nstates
        self.noutputs = noutputs
        self.hidden_dim = hidden_dim
        self.spacedependent = spacedependent
        self.timedependent = timedependent
        self.statedependent = statedependent
        self.period = period
        self.ttype = ttype

        if not statedependent and not timedependent and not spacedependent:
            self.model = nn.Parameter(torch.tensor([0.0], dtype=ttype))
        else:
            input_dim = (
                int(statedependent) + int(timedependent) + 2 * int(spacedependent)
            )
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

    Parameters
    ----------
    nstates : int
        Number of states in a potential state input.
    hidden_dim : int, optional
        Dimension of hidden layers. Default is 100.
    timedependent : bool, optional
        If True, time input is expected. Default is False.
    statedependent : bool, optional
        If True, state input is expected. Default is True.
    spacedependent : bool, optional
        If True, spatial input is expected. Default is False.
    period : int, optional
        Period value used in the PDEExternalForcesNN model. Default is 20.
    number_of_intermediate_outputs : int, optional
        Number of intermediate outputs in the PDEBaselineNN model. Default is 4.
    """

    def __init__(
        self,
        nstates,
        hidden_dim=100,
        timedependent=False,
        statedependent=True,
        spacedependent=False,
        period=20,
        number_of_intermediate_outputs=4,
    ):
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
            nstates,
            hidden_dim,
            False,
            True,
            False,
            number_of_intermediate_outputs=number_of_intermediate_outputs,
        )
        self.external_forces = PDEExternalForcesNN(
            nstates, hidden_dim, timedependent, spacedependent, False, period
        )

    def forward(self, x, t, xspatial):
        return self.baseline_nn(x, t, xspatial) + self.external_forces(x, t, xspatial)


class A_estimator(torch.nn.Module):
    """
    Creates an estimator of a symmetric convolution operator to apply to
    the left-hand side of the PDE system or the integral V.

    Parameters
    ----------
    kernel_size : int
    ttype : torch type, default torch.float32

    """

    def __init__(self, kernel_size=3, ttype=torch.float32):
        super().__init__()

        self.ttype = ttype
        self.kernel_size = kernel_size
        d = int((kernel_size - 1) / 2)
        self.ls = torch.nn.Parameter(
            torch.zeros(d, dtype=self.ttype), requires_grad=True
        )

    def forward(self, x=None):
        """
        Returns
        -------
        (N, N) tensor
            Damping matrix
        """
        if self.kernel_size == 0:
            return torch.tensor([0], dtype=self.ttype).reshape(1, 1, 1)
        else:
            return torch.concat(
                [self.ls, torch.tensor([1], dtype=self.ttype), self.ls]
            ).reshape(1, 1, self.kernel_size)


class S_estimator(torch.nn.Module):
    """
    Creates an estimator of a skew-symmetric convolution operator to apply to
    the integral H.

    Parameters
    ----------
    kernel_size : int
    ttype : torch type, default torch.float32

    """

    def __init__(self, kernel_size=3, ttype=torch.float32):
        super().__init__()

        self.ttype = ttype
        self.kernel_size = kernel_size
        if self.kernel_size > 1:
            d = int((kernel_size - 3) / 2)
            self.ls = torch.nn.Parameter(
                torch.zeros(d, dtype=self.ttype), requires_grad=True
            )

    def forward(self, x=None):
        """
        Returns
        -------
        (N, N) tensor
            Damping matrix
        """
        if self.kernel_size == 1 or self.kernel_size == 0:
            return torch.tensor([0], dtype=self.ttype).reshape(1, 1, 1)
        else:
            return torch.concat(
                [-self.ls, torch.tensor([-1.0, 0.0, 1.0], dtype=self.ttype), self.ls]
            ).reshape(1, 1, self.kernel_size)
