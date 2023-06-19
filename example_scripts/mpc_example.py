import argparse

import matplotlib.pyplot as plt
import numpy as np

from phlearn.phsystems import init_tanksystem
from phlearn.control import StepReference, PseudoHamiltonianMPC
from phlearn.phnns import load_dynamic_system_model

try:
    import casadi
except ModuleNotFoundError:
    raise ModuleNotFoundError("To use the phlearn.control module install via 'pip install phlearn[control]' ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str,
                        help='Path to a learned model (PHNN/baseline) to use as the MPC dynamics model and to simulate the tank system. If None, an analytical tank system is initialized and used instead.')

    args = parser.parse_args()
    modelpath = args.modelpath
    seed = 1

    if modelpath is not None:
        pH_system, optimizer, metadict = load_dynamic_system_model(modelpath)

        if metadict["traininginfo"]["baseline"]:
            S = None
            H = None
            dH = None
            R = None
            baseline = pH_system.rhs_model
            external_forces = None
        else:
            S = pH_system.skewsymmetric_matrix.copy()
            H = pH_system.hamiltonian
            dH = None
            R = pH_system.R
            baseline = None
            external_forces = pH_system.external_forces
        npipes = 5
        ntanks = 4
    else:
        pH_system = init_tanksystem()
        pH_system.seed(seed)

        S = pH_system.skewsymmetric_matrix
        H = None
        dH = pH_system.dH
        baseline = None
        npipes = pH_system.npipes
        ntanks = pH_system.ntanks
        R = pH_system.dissipation_matrix
        external_forces = pH_system.external_forces

    control_forces_filter = np.zeros((pH_system.nstates,))
    control_forces_filter[5] = 1
    control_reference = StepReference(0, 1, step_interval=0.75, seed=seed)

    mpc = PseudoHamiltonianMPC(control_forces_filter,
                             S=S,
                             dH=dH,
                             H=H,
                             F=external_forces,
                             R=R,
                             baseline=baseline,
                             state_names=[f'flow_{i+1}' for i in range(npipes)] + [f'level_{i+1}' for i in range(ntanks)],
                             control_names=['inflow'],
                             references={'Controlled tank level': control_reference})

    def mpc_setup(mpc):
        stage_cost = (mpc.model.x[f'level_{ntanks - 1}'] - mpc.model.tvp['Controlled tank level']) ** 2
        terminal_cost = casadi.DM(np.zeros((1, 1)))
        mpc.set_objective(lterm=stage_cost, mterm=terminal_cost)
        mpc.set_rterm(inflow=1e-2)
        mpc.bounds['lower', '_u', 'inflow'] = -10
        mpc.bounds['upper', '_u', 'inflow'] = 10
        mpc.set_param(t_step=0.025)

        return mpc

    mpc.setup(setup_callback=mpc_setup)

    pH_system.controller = mpc

    t_trajectory = np.linspace(0, 2, 201)
    if modelpath is not None:
        xs, us = pH_system.simulate_trajectory('rk4', t_trajectory)
    else:
        xs, _, _, us = pH_system.sample_trajectory(t_trajectory, noise_std=0)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    for i in range(xs.shape[-1]):
        axs[0].plot(t_trajectory, xs[:, i])
    axs[0].set_title('States')

    axs[1].plot(t_trajectory, xs[:, 5])
    axs[1].plot(t_trajectory, control_reference.get_reference_data(t_trajectory)[0], linestyle='dashed')
    axs[1].set_title('Controlled state')

    axs[2].plot(t_trajectory[:-1], us[:, 5])
    axs[2].axhline(mpc.mpc.bounds['lower', '_u', 'inflow'].__float__(), linestyle='dotted')
    axs[2].axhline(mpc.mpc.bounds['upper', '_u', 'inflow'].__float__(), linestyle='dotted')
    axs[2].set_title('Control inputs')
    axs[2].set_xlabel('Time')
    plt.show()
