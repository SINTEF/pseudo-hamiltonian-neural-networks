from .phcontroller import PortHamiltonianController
from ..phnns.models import R_estimator
from .casadiPH import CasadiPortHamiltonianSystem
from .casadiNN import CasadiFCNN, get_pytorch_model_parameters, get_pytorch_model_architecture
import do_mpc
import numpy as np
import casadi
import torch

__all__ = ['PortHamiltonianMPC']


class PortHamiltonianMPC(PortHamiltonianController):
    """
    This class implements a model predictive controller (MPC) that
    solves an optimal control problem to decide control inputs,
    where the model is formulated as a port-hamiltonian system::

        dx/dt = (S - R) * grad[H(x)] + F(x, t) + u(x, t)

    or if the baseline argument is provided::

        dx/dt = Baseline(x, t) + u(x, t)

    Each component of the model can be provided as either a python
    function or as a pytorch neural network (NN). Note that for any
    component implemented as a python function mist use only operations
    that are compatible with casadi variable types.

    The MPC is based on the do-mpc python toolbox, see its documentation
    for configuration options and behaviour (do-mpc.com).

    Parameters
    ----------
    control_port_filter : matrix
        A binary matrix of (nstates, nstates) or vector of (nstates)
        where 1 signifies that the corresponding state has a control
        input in its derivative right-hand-side.
    S : matrix, default None
        (nstates, nstates) ndarray. nstates if inferred from this.
    dH : callable, default None
        Function/NN computing the gradient of the Hamiltonian. Takes one
        argument (state). Accepts an ndarray both of size (nstates,) and
        of size (nsamples, nstates), and returns either an ndarray of
        size (nstates,) or an ndarray of size (nsamples, nstates),
        correspondingly.
    H  : callable, default None
        Function/NN computing the Hamiltonian of the system. Takes one
        argument (state,). Accepts ndarrays both of size (nstates,) and
        of size (nsamples, nstates), and returns either a scalar or an
        ndarray of size (nstates,), correspondingly.
    R : matrix or array, default None
        (nstates, nstates) ndarray or (N,) ndarray of diagonal elements
        or NN
    F : callable, default None
        Function/NN computing external ports taking two arguments
        (state and time), which can be both an ndarray of size
        (nstates,) + a scalar, and an ndarray of size
        (nsamples, nstates) + an ndarray of size (nstates,). Returns
        either an ndarray of size (nstates,) or an ndarray of size
        (nsamples, nstates), correspondingly.
    baseline : callable, default None
        Alternative model formulation
    state_names : list, default None
        List of length (nstates) where each entry sets the name of the
        model state variables such that states can be referenced by
        these names.
    control_names : list, default None
        List of size (ncontrols) where each entry sets the name of the
        model control input variables such that inputs can be referenced
        by these names.
    references : dict, default None
        Dictionary with reference names as keys and Reference as values.
        Note that all references must be specified using this argument
        on object instantiation as they must be added as variables to
        the model.
    model_callback : callable, default None
        Callback function for the end of model creation taking the model
        as argument, and returns the modified model. Can be used to add
        additional variables to the model.
    p_callback : callable, default None
        Callback function calles before computing the MPC solution
        (get_input) where values for additional parameters must be set
        by user. Takes as argument the parameter object and the current
        model time, and must return the modified parameter object.
    tvp_callback : callable, default None
        Callback function called before computing the MPC solution
        (get_input) where values for additional time-varying parameter
        must be set by user. Takes as argument the
        time-varying-parameter object and the current model time, and
        must return the modified time-varying-parameter object object.

    """

    def __init__(self, control_port_filter, S=None, dH=None, H=None, F=None,
                 R=None, baseline=None, state_names=None,
                 control_names=None, references=None,
                 model_callback=None, p_callback=None, tvp_callback=None):
        self.baseline = baseline

        if baseline is None:
            assert S is not None
            self.S = S

            if dH is None:
                assert H is not None, 'Must provide either dH or H'
                self.H = H
                self.dH = None
            else:
                assert H is None, 'Please only provide one of dH and H'
                self.H = None
                self.dH = dH

            if F is None:
                self.F = lambda x: np.zeros_like(x)
            else:
                self.F = F

            if R is None:
                self.R = lambda x: np.zeros_like(x)
            else:
                self.R = R
        else:
            pass

        self.nstates = control_port_filter.shape[0]
        super().__init__(control_port_filter)

        self.references = references

        self.p_callback = p_callback
        self.tvp_callback = tvp_callback
        self._p_template = None
        self._tvp_template = None
        self._use_time_variable = False

        self.model_callback = model_callback

        self.has_been_reset = False

        self.dynamics_params = None

        model = self._get_model(state_names=state_names, control_names=control_names)
        self.mpc = do_mpc.controller.MPC(model)

    def _get_model(self, state_names=None, control_names=None):
        def pytorch_parameter_getter(module):
            return lambda: np.concatenate([p.flatten() for p in get_pytorch_model_parameters(module)]).reshape(-1, 1)

        model = do_mpc.model.Model('continuous')
        if state_names is None:
            state_names = [f'x{x_i}' for x_i in range(self.nstates)]
        else:
            assert len(state_names) == self.nstates

        if control_names is None:
            control_names = [f'u{u_i}' for u_i in range(self.ncontrols)]
        else:
            assert len(control_names) == self.ncontrols

        for x_i in range(self.nstates):
            _ = model.set_variable(var_type='_x', var_name=state_names[x_i], shape=(1, 1))

        for u_i in range(self.ncontrols):
            _ = model.set_variable(var_type='_u', var_name=control_names[u_i], shape=(1, 1))

        if self.references is not None:
            for ref_name in self.references.keys():
                _ = model.set_variable(var_type='_tvp', var_name=ref_name, shape=(1, 1))

        self.dynamics_params = {}
        u = self.control_port_filter @ model.u.cat

        if self.baseline is None:
            if self.H is not None:
                if isinstance(self.H, torch.nn.Module):
                    nn_H = CasadiFCNN(layers=get_pytorch_model_architecture(self.H))
                    dH = casadi.gradient(nn_H.create_forward(model.x.cat), model.x.cat)

                    model._p["name"].append("H_params")
                    model._p["var"].append(nn_H.params)

                    self.dynamics_params['H'] = pytorch_parameter_getter(self.H)
                else:
                    dH = casadi.gradient(self.H(model.x.cat), model.x.cat)
            else:
                if isinstance(self.dH, torch.nn.Module):
                    nn_dH = CasadiFCNN(layers=get_pytorch_model_architecture(self.dH))
                    dH = nn_dH.create_forward(model.x.cat)

                    model._p["name"].append("H_params")
                    model._p["var"].append(nn_dH.params)

                    self.dynamics_params['H'] = pytorch_parameter_getter(self.dH)
                else:
                    dH = self.dH(model.x.cat)

            if isinstance(self.F, torch.nn.Module):
                nn_F = CasadiFCNN(layers=get_pytorch_model_architecture(self.F))
                F_inputs = []
                if self.F.statedependent:
                    F_inputs.append(model.x)
                if self.F.timedependent:
                    t = model.set_variable(var_type="_tvp", var_name="time", shape=(1, 1))
                    F_inputs.append(t)
                    self._use_time_variable = True
                F = (self.F.external_port_filter.detach().numpy() @ nn_F.create_forward(*F_inputs)).T

                model._p["name"].append("F_params")
                model._p["var"].append(nn_F.params)
                self.dynamics_params['F'] = pytorch_parameter_getter(self.F)
            else:
                F = self.F(model.x.cat)

            if isinstance(self.R, R_estimator):
                R_weights = casadi.SX.sym('R_weights', self.R.rs.shape[0])
                R = casadi.diag(self.R.pick_rs.detach().numpy() @ R_weights)

                model._p["name"].append("R_params")
                model._p["var"].append(R_weights)
                self.dynamics_params['R'] = lambda: self.R.get_parameters()
            elif isinstance(self.R, torch.nn.Module):
                nn_R = CasadiFCNN(layers=get_pytorch_model_architecture(self.R))
                R = nn_R.create_forward(model.x.cat)

                model._p["name"].append("R_params")
                model._p["var"].append(nn_R.params)
                self.dynamics_params['R'] = pytorch_parameter_getter(self.R)
            elif callable(self.R):
                R = self.R(model.x.cat)
            else:
                R = self.R

            dynamics = CasadiPortHamiltonianSystem(S=self.S, dH=dH, u=u, R=R, F=F)
            rhs = dynamics.create_forward()
        else:
            nn_baseline = CasadiFCNN(layers=get_pytorch_model_architecture(self.baseline))
            nn_baseline.set_weights_and_biases(get_pytorch_model_parameters(self.baseline))

            baseline_inputs = []
            if self.baseline.statedependent:
                baseline_inputs.append(model.x)
            if self.baseline.timedependent:
                t = model.set_variable(var_type="_tvp", var_name="time", shape=(1, 1))
                baseline_inputs.append(t)
                self._use_time_variable = True

            rhs = nn_baseline.create_forward(*baseline_inputs).T + u

            model._p["name"].append("baseline_params")
            model._p["var"].append(nn_baseline.params)

            self.dynamics_params['baseline'] = pytorch_parameter_getter(self.baseline)

        for state_i, state_name in enumerate(state_names):
            model.set_rhs(state_name, rhs[state_i])

        if self.model_callback is not None:
            model = self.model_callback(model)

        model.setup()

        return model

    def setup(self, setup_callback):
        """
        Function to finalize MPC creation. Must be called prior to use
        of the MPC for getting control inputs. Note that the objective
        must be set by the user. Other MPC options such as constraints
        and optimization horizon can also
        be configured here through the setup_callback argument.

        parameters
        ----------
        setup_callback : callable
            Function for user to finalize the MPC configuration. Takes
            as argument the MPC object, and returns the modified mpc
            object. Note that set_objective must be called by user on
            the mpc object.

        """

        default_settings = {
            'n_horizon': 10,
            't_step': 0.01,
            'n_robust': 0,
            'store_full_solution': True,
            'nlpsol_opts': {'ipopt.max_iter': 200,
                            'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0
                            },
        }
        self.mpc.set_param(**default_settings)

        self.mpc = setup_callback(self.mpc)

        if self.dynamics_params is not None or self.p_callback is not None:
            self._p_template = self.mpc.get_p_template(1)

            def mpc_p_fun(t_now):
                if self.p_callback is not None:
                    self._p_template = self.p_callback(self._p_template, t_now)

                if self.dynamics_params is not None:
                    for p_name, p_getter in self.dynamics_params.items():
                        self._p_template['_p', :, p_name + '_params'] = p_getter()

                return self._p_template

            self.mpc.set_p_fun(mpc_p_fun)

        if self.references is not None or self.tvp_callback is not None:
            self._tvp_template = self.mpc.get_tvp_template()

            def mpc_tvp_fun(t_now):
                if self.tvp_callback is not None:
                    self._tvp_template = self.tvp_callback(self._tvp_template, t_now)

                if self.references is not None:
                    for t_i in range(self.mpc.n_horizon + 1):
                        for name, fun in self.references.items():
                            self._tvp_template['_tvp', t_i, name] = fun(float(t_now) + t_i * self.mpc.t_step)
                if self._use_time_variable:
                    self._tvp_template['_tvp', :, "time"] = (self.mpc.t0 + np.arange(self.mpc.n_horizon + 1) * self.mpc.t_step).tolist()

                return self._tvp_template

            self.mpc.set_tvp_fun(mpc_tvp_fun)
        self.mpc.setup()

    def reset(self):
        """
        Function called before starting control of a new trajectory.
        Resets the MPC state, and the reference object.
        """
        self.mpc.reset_history()
        self.has_been_reset = True
        if self.references is not None:
            for ref_fun in self.references.values():
                ref_fun.reset()

    def set_reference(self, references):
        """
        Function used to change the Reference objects specified during
        instantiation. Note that only existing references
        can be updated using this function.

        Parameters
        ----------
        references : dict
            Dictionary of reference name as keys and Reference object
            as values.

        """

        assert isinstance(references, dict), 'The PortHamiltonianMPC class only supports references in dictionary format'
        assert all(name in self.references for name in references), 'New references can not be added after the MPC is created.'
        for name, ref_fun in references.items():
            self.references[name] = ref_fun

    def _get_input(self, x, t=None):
        """
        Function used to compute the MPC solution and obtain the
        corresponding control input.

        Parameters
        ----------
        x : array
            Initial state of the MPC optimal control problem.
        t : number
            System time for the optimal control problem. Affects
            parameters and time-varying parameters such as
            references.

        """

        if self.has_been_reset:
            self.mpc.x0 = x
            self.mpc.u0 = np.zeros((self.ncontrols,))
            self.mpc.set_initial_guess()
            self.has_been_reset = False
        if t is not None:
            self.mpc.t0 = t
        return self.mpc.make_step(x)
