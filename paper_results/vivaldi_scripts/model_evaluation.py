
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

import phlearn.phsystems as phsys
import phlearn.phnns as phnn

# For testing:
import matplotlib.pyplot as plt
colors = [(0,0.4,1),(1,0.7,0.3),(0.2,0.7,0.2),(0.8,0,0.2),(0.5,0.3,.9)]

ttype = torch.float32
torch.set_default_dtype(ttype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelfolder', type=str,
                        help='Path to folder containing model dicts.')
    parser.add_argument('--system', type=str, choices=['kdv', 'burgers', 'bbm'], required=True,
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

    if system == 'kdv':
    #     PDE_system = init_kdv()
        x_max = 20
        x_points = 100
        dx = x_max/x_points
        x = np.linspace(0,x_max-dx,x_points)
        eta = 6.
        gamma = 1.
        nu = 0.3 # coefficient of viscosity term
        force = None
        # def F(u, t):
        #     return 0
        def F(u, t):
            return 1*.6*np.sin(2*2*np.pi/x_max*x)
        def JF(u,t):
            return np.zeros((u.shape[0],u.shape[0]))
        PDE_system = phsys.KdVSystem(x=x, eta=eta, gamma=gamma, nu=nu, force=F, force_jac=JF, init_sampler=phsys.initial_condition_kdv(x, eta))
    elif system == 'bbm':
        x_max = 50
        x_points = 100
        dx = x_max/x_points
        x = np.linspace(0,x_max-dx,x_points)
        nu = 1. # coefficient of viscosity term
        def F(u, t):
            return 1*np.sin(2*2*np.pi/x_max*x)
        def JF(u,t):
            return np.zeros((u.shape[0],u.shape[0]))
        PDE_system = phsys.BBMSystem(x=x, nu=nu, force=F, force_jac=JF,
                               init_sampler=phsys.initial_condition_bbm(x))

    nstates = PDE_system.nstates

    index = []
    for f in glob.glob(modelfolder+"/*.model"):
        modelname = f.split('/')[-1]
        index += [modelname]

    df = pd.DataFrame(index=index)

    for modelname in df.index:
        modelpath = os.path.join(modelfolder, modelname)
        model, optimizer, metadict = phnn.load_dynamic_system_model(modelpath)

        PDE_system.seed(seed)
        dt = args.dt_rollout
        if dt is None:
            dt = metadict['traininginfo']['sampling_time']
        t_max = args.t_max
        if t_max is None:
            t_max = metadict['traininginfo']['t_max']
        t_sample = np.arange(0, t_max, dt)
        nsamples = t_sample.shape[0]

        x_exact, dxdt, _, _ = PDE_system.sample_trajectory(t_sample)
        x0 = x_exact[0, :]
        x_phnn, _ = model.simulate_trajectory('rk4', t_sample, x0=x0, xspatial=x)

        df.loc[modelname, 'MSE'] = ((x_exact - x_phnn)**2).mean()#axis=1)
        df.loc[modelname, 'MSE endtime'] = ((x_exact[-1,:] - x_phnn[-1,:])**2).mean()#axis=1)

        for i in range(1,nrollouts):
            x_exact, dxdt, _, _ = PDE_system.sample_trajectory(t_sample)
            x0 = x_exact[0, :]
            x_phnn, _ = model.simulate_trajectory('rk4', t_sample, x0=x0, xspatial=x)

            df.loc[modelname, 'MSE'] += ((x_exact - x_phnn)**2).mean()#axis=1)
            df.loc[modelname, 'MSE endtime'] += ((x_exact[-1,:] - x_phnn[-1,:])**2).mean()#axis=1)

        df.loc[modelname, :] = df.loc[modelname].values / nrollouts

        #df.loc[modelname, 'PDE system'] = PDE_system
 
        df.to_csv(os.path.join(modelfolder, f'testresults_dt{dt:.0e}_n{int(nrollouts)}_t{int(t_max)}.csv'))