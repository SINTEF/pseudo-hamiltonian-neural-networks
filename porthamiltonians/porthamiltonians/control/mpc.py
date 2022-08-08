from .phcontroller import PortHamiltonianController
from ..phnns.models import R_estimator
from .casadiPH import CasadiPortHamiltonianSystem
from .casadiNN import CasadiFCNN, get_pytorch_model_parameters, get_pytorch_model_architecture
import do_mpc
import numpy as np
import casadi
import torch


class PortHamiltonianMPC(PortHamiltonianController):
    """
    This class implements a model predictive controller (MPC) that solves an optimal control problem to decide control
    inputs, where the model is formulated as a port-hamiltonian system:
        dx/dt = (S - R)*grad[H(x)] + F(x, t) + u(x, t)
    or if the baseline argument is provided:
        dx/dt = Baseline(x, t) + u(x, t)
    Each component of the model can be provided as either a python function or as a pytorch neural network (NN). Note
    that for any component implemented as a python function mist use only operations that are compatible with casadi
    variable types.

    The MPC is based on the do-mpc python toolbox, see its documentation for configuration options and behaviour
    (do-mpc.com).

    parameters
    ----------
        control_port_filter :  A binary matrix of (nstates, nstates) or vector of (nstates) where 1 signifies that the
        corresponding state has a control input in its derivative right-hand-side.
        S   :   (nstates, nstates) ndarray. nstates if inferred from this.
        dH  :   Function/NN computing the gradient of the Hamiltonian. Takes one argument (state).
                Accepts an ndarray both of size (nstates,) and of size (nsamples, nstates), and
                returns either an ndarray of size (nstates,) or an ndarray of size (nsamples, nstates),
                correspondingly.
        H   :   Function/NN computing the Hamiltonian of the system. Takes one argument (state,).
                Accepts ndarrays both of size (nstates,) and of size (nsamples, nstates), and
                returns either a scalar or an ndarray of size (nstates,), correspondingly.
        R   :   (nstates, nstates) ndarray or (N,) ndarray of diagonal elements or NN
        F   :   Function/NN computing external ports taking two arguments (state and time), which can be both
                an ndarray of size (nstates,) + a scalar, and an ndarray of size (nsamples, nstates) + an
                ndarray of size (nstates,). Returns either an ndarray of size (nstates,) or an ndarray of size
                (nsamples, nstates), correspondingly.
        baseline   :   Alternative model formulation
        state_names :   List of size (nstates) where each entry sets the name of the model state variables such that
        states can be referenced by these names.
        control_names   : List of size (ncontrols) where each entry sets the name of the model control input variables such
        that inputs can be referenced by these names.
        references  : Dictionary with reference names as keys and Reference as values. Note that all references must be
        specified using this argument on object instantiation as they must be added as variables to the model.
        model_callback  : Callback function for the end of model creation taking the model as argument,
        and returns the modified model. Can be used to add additional variables to the model.
        p_callback  :   Callback function called before computing the MPC solution (get_input) where values for
        additional parameters must be set by user. Takes as argument the parameter object and the current model time,
        and must return the modified parameter object.
        tvp_callback    :   Callback function called before computing the MPC solution (get_input) where values for
        additional time-varying parameters must be set by user. Takes as argument the time-varying-parameter object and
        the current model time, and must return the modified time-varying-parameter object object.
    """
    def __init__(self, control_port_filter, S=None, dH=None, H=None, F=None, R=None, baseline=None, state_names=None, control_names=None, references=None, model_callback=None, p_callback=None, tvp_callback=None):
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
                self.F = lambda x: np.zeros_like(x)  # TODO:
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

                    model._p.append(casadi.tools.entry('H_params', sym=nn_H.params))

                    self.dynamics_params['H'] = pytorch_parameter_getter(self.H)
                else:
                    dH = casadi.gradient(self.H(model.x.cat), model.x.cat)
            else:
                if isinstance(self.dH, torch.nn.Module):
                    nn_dH = CasadiFCNN(layers=get_pytorch_model_architecture(self.dH))
                    dH = nn_dH.create_forward(model.x.cat)

                    model._p.append(casadi.tools.entry('H_params', sym=nn_dH.params))

                    self.dynamics_params['H'] = pytorch_parameter_getter(self.dH)
                else:
                    dH = self.dH(model.x.cat)

            if isinstance(self.F, torch.nn.Module):
                nn_F = CasadiFCNN(layers=get_pytorch_model_architecture(self.F))
                F = (self.F.external_port_filter.detach().numpy() @ nn_F.create_forward(model.x)).T

                model._p.append(casadi.tools.entry('F_params', sym=nn_F.params))
                self.dynamics_params['F'] = pytorch_parameter_getter(self.F)
            else:
                F = self.F(model.x.cat)

            if isinstance(self.R, R_estimator):
                R_weights = casadi.SX.sym('R_weights', self.R.rs.shape[0])
                R = casadi.diag(self.R.pick_rs.detach().numpy() @ R_weights)

                model._p.append(casadi.tools.entry('R_params', sym=R_weights))
                self.dynamics_params['R'] = lambda: self.R.get_parameters()
            elif isinstance(self.R, torch.nn.Module):
                nn_R = CasadiFCNN(layers=get_pytorch_model_architecture(self.R))
                R = nn_R.create_forward(model.x)

                model._p.append(casadi.tools.entry('R_params', sym=nn_R.params))
                self.dynamics_params['R'] = pytorch_parameter_getter(self.R)
            elif callable(self.R):
                R = self.R(model.x.cat)
            else:
                R = self.R

            dynamics = CasadiPortHamiltonianSystem(S=self.S, dH=dH, u=u, R=R, F=F) # TODO: how to get time variable from MPC for time-dependent external port?
            rhs = dynamics.create_forward()
        else:  # TODO: maybe this can be general right-hand-side? i.e. also used for analytical
            nn_baseline = CasadiFCNN(layers=get_pytorch_model_architecture(self.baseline))
            nn_baseline.set_weights_and_biases(get_pytorch_model_parameters(self.baseline))
            rhs = nn_baseline.create_forward(model.x.cat).T + u

            model._p.append(casadi.tools.entry('baseline_params', sym=nn_baseline.params))

            self.dynamics_params['baseline'] = nn_baseline

        for state_i, state_name in enumerate(state_names):
            model.set_rhs(state_name, rhs[state_i])

        if self.model_callback is not None:
            model = self.model_callback(model)

        model.setup()

        return model

    def setup(self, setup_callback):
        """
        Function to finalize MPC creation. Must be called prior to use of the MPC for getting control inputs. Note that
        the objective must be set by the user. Other MPC options such as constraints and optimization horizon can also
        be configured here through the setup_callback argument.

        parameters
        ----------
            setup_callback  :   Function for user to finalize the MPC configuration. Takes as argument the MPC object,
            and returns the modified mpc object. Note that set_objective must be called by user on the mpc object.
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

                return self._tvp_template

            self.mpc.set_tvp_fun(mpc_tvp_fun)
        self.mpc.setup()

    def reset(self):
        """
        Function called before starting control of a new trajectory. Resets the MPC state, and the reference object.
        """
        self.mpc.reset_history()
        self.has_been_reset = True
        if self.references is not None:
            for ref_fun in self.references.values():
                ref_fun.reset()

    def set_reference(self, references):
        """
        Function used to change the Reference objects specified during instantiation. Note that only existing references
        can be updated using this function.

        parameters
        ----------
            references  :   Dictionary of reference name as keys and Reference object as values.
        """
        assert isinstance(references, dict), 'The PortHamiltonianMPC class only supports references in dictionary format'
        assert all(name in self.references for name in references), 'New references can not be added after the MPC is created.'
        for name, ref_fun in references.items():
            self.references[name] = ref_fun

    def _get_input(self, x, t=None):
        """
        Function used to compute the MPC solution and obtain the corresponding control input.

        parameters
        ----------
            x   :   Initial state of the MPC optimal control problem.
            t   :   System time for the optimal control problem. Affects parameters and time-varying parameters such as
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
