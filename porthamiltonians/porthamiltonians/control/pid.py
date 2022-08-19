import numpy as np
from .phcontroller import PortHamiltonianController

__all__ = ['PIDController']


class PIDController(PortHamiltonianController):
    """
    This class implements a proportional-integral-derivative (PID)
    controller. The PID controller is a SISO controller,
    but several PIDs can be configured to run in parallel,
    up to one per state of the system.

    parameters
    ----------
    control_port_filter : (nstates, nstates) or (nstates,) ndarray
        A binary ndarray where 1 signifies that the corresponding
        state has a control input in its derivative right-hand-side.
    gains : dict
        Dictionary where the key is the index of the state that the
        PID controls, and the value is anotherdictionary of the form
        {"p": proportional gain, "i": integral gain", "d":derivative
        gain}. Missing gains are assumed to be zero.
    references : dict of references
        Dictionary where the key is the index of the state that the PID
        controls, and the value is a Reference object for that PID
        controller.
    input_bounds : dict
        Dictionary where the key is the index of the state that the PID
        controls, and the value is a two-element list where the first
        element gives a lower bound for the control input and the second
        element gives an upper bound.
    """
    def __init__(self, control_port_filter, gains, references, input_bounds=None):
        self.gains = {}
        for idx, idx_gains in gains.items():
            self.gains[idx] = {}
            for gain in ['p', 'i', 'd']:
                self.gains[idx][gain] = idx_gains.get(gain, 0.0)

        super().__init__(control_port_filter=control_port_filter)

        self.state_idxs = [np.argwhere(self.control_port_filter[:, i]).item() for i in range(self.ncontrols)]

        assert input_bounds is None or len(input_bounds.keys()) == self.ncontrols
        self.input_bounds = input_bounds

        assert len(references.keys()) == self.ncontrols
        self.references = references

        self.integrator = [0] * self.ncontrols
        self.prev_state = [None] * self.ncontrols

        self.last_t = 0
        self.history = {'t': [], 'input': []}

    def set_reference(self, references):
        assert isinstance(references, dict)
        for idx, ref_fun in references.items():
            self.references[idx] = ref_fun

    def reset(self):
        self.integrator = [0] * self.ncontrols
        self.prev_state = [None] * self.ncontrols
        self.history = {'t': [], 'input': []}
        self.last_t = 0
        for ref_fun in self.references.values():
            ref_fun.reset()

    def _get_input(self, state, t=None):
        us = []
        if t >= self.last_t:  # some solvers are not monotonic wrt. time
            step_dt = t - self.last_t

            for i, idx in enumerate(self.state_idxs):
                state = np.atleast_2d(state)
                error = self.references[idx](t) - state[..., idx].ravel()

                self.integrator[i] += step_dt * error

                u = self.gains[idx]['p'] * error + self.gains[idx]['i'] * self.integrator[i]
                if self.prev_state[i] is not None and self.gains[idx]['d'] > 0:
                    u += self.gains[idx]['d'] * (state[..., idx] - self.prev_state[i]) / (step_dt + 1e-12)
                self.prev_state[i] = state[..., idx]

                # Constrain input
                if self.input_bounds is not None and idx in self.input_bounds:
                    u = np.clip(u, *self.input_bounds[idx])

                self.last_t = t
                us.append(np.squeeze(u))
            self.history['t'].append(t)
            self.history['input'].append(np.array(us))
        else:  # if earlier in time, interpolate between closest previously computed inputs
            closest_index = np.abs(np.array(self.history['t']) - t).argmin()
            next_closest_index = (closest_index - 1) if self.history['t'][closest_index] - t > 0 else (closest_index + 1)
            t_span = np.abs(self.history['t'][closest_index] - self.history['t'][next_closest_index])
            while t_span == 0:
                next_closest_index += 1 * (np.sign(next_closest_index - closest_index))
                t_span = np.abs(self.history['t'][closest_index] - self.history['t'][next_closest_index])
            interpol_factor = np.abs((self.history['t'][closest_index] - t) / t_span)
            assert interpol_factor <= 1, 'Something was wrong with interpolation logic'
            us = self.history['input'][closest_index] * (1 - interpol_factor) + self.history['input'][next_closest_index] * interpol_factor

        return us
