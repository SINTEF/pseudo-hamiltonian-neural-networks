
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from phlearn.phnns import load_dynamic_system_model, PseudoHamiltonianNN, BaselineSplitNN
from phlearn.phsystems.ode import init_tanksystem, init_msdsystem

ttype = torch.float32
torch.set_default_dtype(ttype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelfolder', type=str,
                        help='Path to folder containing model dicts.')
    parser.add_argument('--system', type=str, choices=['tank', 'msd'], required=True,
                        help='Choose to train a tank or a forced mass spring damper.')
    parser.add_argument('--nrollouts', '-n', type=int, default=10,
                        help='Number of trajectories to roll out per model.')
    parser.add_argument('--dt_rollout', '-d', type=float,
                        help='Sample time of rollouts.')
    parser.add_argument('--t_max', '-t', type=float,
                        help='Duration of each rollout. If not provided, duration from training of each model is used.')

    args = parser.parse_args()

    modelfolder = args.modelfolder
    system = args.system
    nrollouts = args.nrollouts
    seed = 100

    if system == 'tank':
        pH_system = init_tanksystem()
    else:
        pH_system = init_msdsystem()

    nstates = pH_system.nstates

    index = []
    for f in glob.glob(modelfolder+"/*.model"):
        modelname = f.split('/')[-1]
        index += [modelname]

    df = pd.DataFrame(index=index)

    for modelname in df.index:
        modelpath = os.path.join(modelfolder, modelname)
        model, optimizer, metadict = load_dynamic_system_model(modelpath)

        pH_system.seed(seed)
        dt = args.dt_rollout
        if dt is None:
            dt = metadict['traininginfo']['sampling_time']
        t_max = args.t_max
        if t_max is None:
            t_max = metadict['traininginfo']['t_max']
        t_sample = np.arange(0, t_max, dt)
        nsamples = t_sample.shape[0]

        x_exact, dxdt, _, _ = pH_system.sample_trajectory(t_sample)
        x0 = x_exact[0, :]
        x_phnn, _ = model.simulate_trajectory(False, t_sample, x0=x0)

        if isinstance(model, PseudoHamiltonianNN):
            if (not model.external_forces_provided):
                F_phnn = model.external_forces(torch.tensor(x_phnn, dtype=ttype),
                                                torch.tensor(t_sample.reshape(nsamples, 1), dtype=ttype)).detach().numpy()
                F_phnn -= F_phnn.mean(axis=0)
                F_exact = pH_system.external_forces(x_exact, t_sample)
                df.loc[modelname, 'External force MSE'] = ((F_phnn - F_exact)**2).mean()
                df.loc[modelname, 'External force MAE'] = np.abs(F_phnn - F_exact).mean()
            if (not model.dissipation_provided):
                df.loc[modelname, 'R MSE'] = ((model.R(x_exact).detach().numpy() - pH_system.R(x_exact))**2).mean()
                df.loc[modelname, 'R MAE'] = np.abs(model.R(x_exact).detach().numpy() - pH_system.R(x_exact)).mean()
            if (not model.hamiltonian_provided):
                if pH_system.H is not None:
                    hamiltonian_exact = pH_system.H(x_exact)
                    hamiltonian_phnn = model.hamiltonian(torch.tensor(x_phnn, dtype=ttype)).detach().numpy()
                    hamiltonian_exact = pH_system.H(x_exact)
                    hamiltonian_phnn = model.hamiltonian(torch.tensor(x_phnn, dtype=ttype)).detach().numpy()
                    df.loc[modelname, 'H MSE'] = ((hamiltonian_exact - hamiltonian_phnn)**2).mean()
                    df.loc[modelname, 'H MAE'] = np.abs(hamiltonian_exact - hamiltonian_phnn).mean()
                dH_exact = pH_system.dH(x_exact)
                dH_phnn = model.dH(torch.tensor(x_phnn, dtype=ttype)).detach().numpy()
                if system == 'tank':
                    df.loc[modelname, 'dH tanks MSE'] = ((pH_system.tanklevels(dH_exact) - pH_system.tanklevels(dH_phnn))**2).mean()
                    df.loc[modelname, 'dH tanks MAE'] = np.abs(pH_system.tanklevels(dH_exact) - pH_system.tanklevels(dH_phnn)).mean()
                    df.loc[modelname, 'dH pipes MSE'] = ((pH_system.pipeflows(dH_exact) - pH_system.pipeflows(dH_phnn))**2).mean()
                    df.loc[modelname, 'dH pipes MAE'] = np.abs(pH_system.pipeflows(dH_exact) - pH_system.pipeflows(dH_phnn)).mean()
                else:
                    df.loc[modelname, 'dH x1 MSE'] = ((dH_exact[:, 0] - dH_phnn[:, 0])**2).mean()
                    df.loc[modelname, 'dH x1 MAE'] = np.abs(dH_exact[:, 0] - dH_phnn[:, 0]).mean()
                    df.loc[modelname, 'dH x2 MSE'] = ((dH_exact[:, 1] - dH_phnn[:, 1])**2).mean()
                    df.loc[modelname, 'dH x2 MAE'] = np.abs(dH_exact[:, 1] - dH_phnn[:, 1]).mean()
        elif isinstance(model.rhs_model, BaselineSplitNN):
            F_baseline = model.rhs_model.network_t(torch.tensor(x_phnn, dtype=ttype),
                                        torch.tensor(t_sample.reshape(nsamples, 1), dtype=ttype)).detach().numpy()
            F_baseline -= F_baseline.mean(axis=0)
            F_exact = pH_system.external_forces(x_exact, t_sample)
            df.loc[modelname, 'External force MSE'] = ((F_baseline - F_exact)**2).mean()
            df.loc[modelname, 'External force MAE'] = np.abs(F_baseline - F_exact).mean()

        if system == 'tank':
            df.loc[modelname, 'Tanks MSE'] = ((pH_system.tanklevels(x_exact) - pH_system.tanklevels(x_phnn))**2).mean()
            df.loc[modelname, 'Pipes MSE'] = ((pH_system.pipeflows(x_exact) - pH_system.pipeflows(x_phnn))**2).mean()
            df.loc[modelname, 'Tanks MAE'] = np.abs(pH_system.tanklevels(x_exact) - pH_system.tanklevels(x_phnn)).mean()
            df.loc[modelname, 'Pipes MAE'] = np.abs(pH_system.pipeflows(x_exact) - pH_system.pipeflows(x_phnn)).mean()
        else:
            df.loc[modelname, 'x1 MSE'] = ((x_exact[:, 0] - x_phnn[:, 0])**2).mean()
            df.loc[modelname, 'x1 MAE'] = np.abs(x_exact[:, 0] - x_phnn[:, 0]).mean()
            df.loc[modelname, 'x2 MSE'] = ((x_exact[:, 1] - x_phnn[:, 1])**2).mean()
            df.loc[modelname, 'x2 MAE'] = np.abs(x_exact[:, 1] - x_phnn[:, 1]).mean()

        for i in range(1,nrollouts):
            x_exact, dxdt, _, _ = pH_system.sample_trajectory(t_sample)
            x0 = x_exact[0, :]
            x_phnn, _ = model.simulate_trajectory(False, t_sample, x0=x0)

            if isinstance(model, PseudoHamiltonianNN):
                if (not model.external_forces_provided):
                    F_phnn = model.external_forces(torch.tensor(x_phnn, dtype=ttype),
                                                 torch.tensor(t_sample.reshape(nsamples, 1), dtype=ttype)).detach().numpy()
                    F_phnn -= F_phnn.mean(axis=0)
                    F_exact = pH_system.external_forces(x_exact, t_sample)
                    df.loc[modelname, 'External force MSE'] += ((F_phnn - F_exact)**2).mean()
                    df.loc[modelname, 'External force MAE'] += np.abs(F_phnn - F_exact).mean()
                if (not model.dissipation_provided):
                    df.loc[modelname, 'R MSE'] += ((model.R(x_exact).detach().numpy() - pH_system.R(x_exact))**2).mean()
                    df.loc[modelname, 'R MAE'] += np.abs(model.R(x_exact).detach().numpy() - pH_system.R(x_exact)).mean()
                if (not model.hamiltonian_provided):
                    if pH_system.H is not None:
                        hamiltonian_exact = pH_system.H(x_exact)
                        hamiltonian_phnn = model.hamiltonian(torch.tensor(x_phnn, dtype=ttype)).detach().numpy()
                        hamiltonian_exact = pH_system.H(x_exact)
                        hamiltonian_phnn = model.hamiltonian(torch.tensor(x_phnn, dtype=ttype)).detach().numpy()
                        df.loc[modelname, 'H MSE'] += ((hamiltonian_exact - hamiltonian_phnn)**2).mean()
                        df.loc[modelname, 'H MAE'] += np.abs(hamiltonian_exact - hamiltonian_phnn).mean()
                    dH_exact = pH_system.dH(x_exact)
                    dH_phnn = model.dH(torch.tensor(x_phnn, dtype=ttype)).detach().numpy()
                    if system == 'tank':
                        df.loc[modelname, 'dH tanks MSE'] += ((pH_system.tanklevels(dH_exact) - pH_system.tanklevels(dH_phnn))**2).mean()
                        df.loc[modelname, 'dH tanks MAE'] += np.abs(pH_system.tanklevels(dH_exact) - pH_system.tanklevels(dH_phnn)).mean()
                        df.loc[modelname, 'dH pipes MSE'] += ((pH_system.pipeflows(dH_exact) - pH_system.pipeflows(dH_phnn))**2).mean()
                        df.loc[modelname, 'dH pipes MAE'] += np.abs(pH_system.pipeflows(dH_exact) - pH_system.pipeflows(dH_phnn)).mean()
                    else:
                        df.loc[modelname, 'dH x1 MSE'] += ((dH_exact[:, 0] - dH_phnn[:, 0])**2).mean()
                        df.loc[modelname, 'dH x1 MAE'] += np.abs(dH_exact[:, 0] - dH_phnn[:, 0]).mean()
                        df.loc[modelname, 'dH x2 MSE'] += ((dH_exact[:, 1] - dH_phnn[:, 1])**2).mean()
                        df.loc[modelname, 'dH x2 MAE'] += np.abs(dH_exact[:, 1] - dH_phnn[:, 1]).mean()
            elif isinstance(model.rhs_model, BaselineSplitNN):
                F_baseline = model.rhs_model.network_t(torch.tensor(x_phnn, dtype=ttype),
                                            torch.tensor(t_sample.reshape(nsamples, 1), dtype=ttype)).detach().numpy()
                F_baseline -= F_baseline.mean(axis=0)
                F_exact = pH_system.external_forces(x_exact, t_sample)
                df.loc[modelname, 'External force MSE'] += ((F_baseline - F_exact)**2).mean()
                df.loc[modelname, 'External force MAE'] += np.abs(F_baseline - F_exact).mean()

            if system == 'tank':
                df.loc[modelname, 'Tanks MSE'] += ((pH_system.tanklevels(x_exact) - pH_system.tanklevels(x_phnn))**2).mean()
                df.loc[modelname, 'Pipes MSE'] += ((pH_system.pipeflows(x_exact) - pH_system.pipeflows(x_phnn))**2).mean()
                df.loc[modelname, 'Tanks MAE'] += np.abs(pH_system.tanklevels(x_exact) - pH_system.tanklevels(x_phnn)).mean()
                df.loc[modelname, 'Pipes MAE'] += np.abs(pH_system.pipeflows(x_exact) - pH_system.pipeflows(x_phnn)).mean()
            else:
                df.loc[modelname, 'x1 MSE'] += ((x_exact[:, 0] - x_phnn[:, 0])**2).mean()
                df.loc[modelname, 'x1 MAE'] += np.abs(x_exact[:, 0] - x_phnn[:, 0]).mean()
                df.loc[modelname, 'x2 MSE'] += ((x_exact[:, 1] - x_phnn[:, 1])**2).mean()
                df.loc[modelname, 'x2 MAE'] += np.abs(x_exact[:, 1] - x_phnn[:, 1]).mean()

        df.loc[modelname, :] = df.loc[modelname].values / nrollouts
 
        df.to_csv(os.path.join(modelfolder, f'testresults_dt{dt:.0e}_n{int(nrollouts)}_t{int(t_max)}.csv'))