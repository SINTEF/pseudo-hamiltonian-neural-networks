
import torch
import numpy as np


class PortHamiltonianController:
    """
    Abstract base class for controllers of port-hamiltonian systems
    of the form::

        dx/dt = (S - R)*grad[H(x)] + F(x, t) + u(x, t)

    where this class implements u(x, t), i.e. known and controlled
    external ports. Implementations of controllers
    must subclass this class and implement all of its methods.

    parameters
    ----------
    control_port_filter : (nstates, nstates) or (nstates,) ndarray
        A binary ndarray where 1 signifies that the corresponding
        state has a control input in its derivative right-hand-side.

    """
    def __init__(self, control_port_filter):
        self.ncontrols = int(np.sum(control_port_filter))
        self.control_port_filter = self._format_control_port_filter(control_port_filter)

    def __call__(self, x, t=None):
        """
        Control inputs are computed by calling the controller with a
        system state, and optionally with system time for controllers
        that depend on time (e.g. through time-varying references).
        The returned control input will have the
        same shape as the system state.

        parameters
        ----------
            x : (nstates,) ndarray
                System state
            t : number, default None
                System time

        Returns
        -------
            (nstates,) ndarray
        """
        if torch.is_tensor(x):
            x = x.detach().numpy()
        if torch.is_tensor(t):
            t = t.detach().numpy()
        # TODO: what about batch operation?
        return (self.control_port_filter @ self._get_input(x, t)).ravel()

    def _get_input(self, x, t=None):
        raise NotImplementedError

    def reset(self):
        """
        Function called before starting control of a new trajectory.
        Resets the controllers' internal state, as well as
        any reference objects.
        """
        raise NotImplementedError

    def set_reference(self, references):
        """
        Function used to change the Reference objects specified during
        instantiation.

        Parameters
        ----------
        references  :dict of references
            Dictionary of reference name as keys and Reference
            object as values.

        """
        raise NotImplementedError

    def _format_control_port_filter(self, control_port_filter):
        control_port_filter = (np.array(control_port_filter) > 0).astype(int)

        if (len(control_port_filter.shape) == 1) or (len(control_port_filter.shape[-1]) == 1):
            control_port_filter = control_port_filter.flatten()
            expanded = np.zeros((control_port_filter.shape[-1], control_port_filter.sum()))
            c = 0
            for i, e in enumerate(control_port_filter):
                if e > 0:
                    expanded[i, c] = 1
                    c += 1
            return expanded

        return control_port_filter
