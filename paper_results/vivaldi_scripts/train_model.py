
import argparse

import numpy as np
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
    parser.add_argument('--system', type=str, choices=['kdv', 'heat', 'bbm'],
                        required=True,
                        help='PDE system to train')
    parser.add_argument('--baseline', type=int, default=0,  choices=[0, 1, 2, 3],
                        help='If 1 use baseline network x_dot = network(x, t). '
                                'If 2 use split baseline network '
                                'x_dot = network_x(x) + network_t(x).')
    parser.add_argument('--storedir', type=str,
                        help='Directory for storing the best model in terms '
                                'of validation loss.')
    parser.add_argument('--modelname', type=str,
                        help='Name to use for the stored model.')
    parser.add_argument('--modelpath', type=str,
                        help='Path to existing model to continue training.')
    parser.add_argument('--ntrainingpoints', '-n', type=int, default=3000,
                        help='Number of training points.')
    parser.add_argument('--ntrajectories_val', type=int, default=0,
                        help='Number of trajectories for validation.')
    parser.add_argument('--sampling_time', type=float, default=1/30,
                        help='Sampling time.')
    parser.add_argument('--t_max', type=float, default=1,
                        help='Length of trajectory.')
    parser.add_argument('--true_derivatives', action='store_true',
                        help='Use the true derivative values for training. '
                                'If not provided derivatives in the training '
                                'data are estimated by the finite differences.')
    parser.add_argument('--integrator', type=str,
                        choices=['False', 'euler', 'rk4', 'midpoint', 'srk4', 'cm4', 'cs6'],
                        default='midpoint',
                        help='Integrator used during training.')
    parser.add_argument('--F_timedependent', type=int, default=0,
                        choices=[0, 1],
                        help='If 1, make external forces NN (or baseline NN) '
                                'depend on time.')
    parser.add_argument('--F_spacedependent', type=int, default=1,
                        choices=[0, 1],
                        help='If 1, make external forces NN (or baseline NN) '
                                'depend on space.')
    parser.add_argument('--F_statedependent', type=int, default=0,
                        choices=[0, 1],
                        help='If 1, make external forces NN (or baseline NN) '
                                'depend on state.')
    parser.add_argument('--kernel_sizes', type=str, default='[1,3,0,0]',
                        help='Kernel sizes of the convolutional operators acting on the '
                                'left-hand side of the PDE, the Hamiltonian term, the '
                                'dissipative term and the external forces, respectively.')
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help='Hidden dimension of fully connected neural '
                                'network layers.')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate of Adam optimizer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size used in training.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--l1_param_forces', type=float, default=0.,
                        help='L1 penalty parameter of external forces estimate.')
    parser.add_argument('--l1_param_dissipation', type=float, default=0.,
                        help='L1 penalty parameter of dissipation estimate.')
    parser.add_argument('--early_stopping_patience', type=int,
                        help=('Number of epochs to continue training without '
                                'a decrease in validation loss '
                                'of at least early_stopping_delta.'))
    parser.add_argument('--early_stopping_delta', type=float,
                        help='Minimum accepted decrease in validation loss to '
                                'prevent early stopping.')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='Shuffle training data at every epoch.')
    parser.add_argument('--noise_std', type=float, default=0.,
                        help='Noise level for training.')
    parser.add_argument('--store_results', '-s', action='store_true',
                        help='Store trained model and prediction results.')
    parser.add_argument('--seed', type=int,
                        help='Seed for the simulated dynamic system.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print information while training.')
    args = parser.parse_args()

    system = args.system
    baseline = int(args.baseline)
    storedir = args.storedir
    modelname = args.modelname
    modelpath = args.modelpath
    ntrainingpoints = args.ntrainingpoints
    sampling_time = args.sampling_time
    t_max = args.t_max
    true_derivatives = args.true_derivatives
    if args.integrator=='False':
        true_derivatives = True
    if true_derivatives:
        integrator = False
        print('Warning: As exact derivatives are used when generating '
                'training data, (true_derivatives = True) integrator'
                'is set to False.')
    else:
        integrator = args.integrator

    F_timedependent = bool(args.F_timedependent)
    F_spacedependent = bool(args.F_spacedependent)
    F_statedependent = bool(args.F_statedependent)
    kernel_sizes = eval(args.kernel_sizes)
    hidden_dim = args.hidden_dim
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    l1_param_forces = args.l1_param_forces
    l1_param_dissipation = args.l1_param_dissipation
    shuffle = args.shuffle
    early_stopping_patience = args.early_stopping_patience
    early_stopping_delta = args.early_stopping_delta
    noise_std = args.noise_std
    store_results = args.store_results
    seed = args.seed
    verbose = args.verbose
    ntrajectories_val = args.ntrajectories_val

    ntrajectories_train, t_sample = phnn.npoints_to_ntrajectories_tsample(
        ntrainingpoints, t_max, sampling_time)

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
        def F(u, t):
            return 1*.6*np.sin(2*2*np.pi/x_max*x)
        def JF(u,t):
            return np.zeros((u.shape[0],u.shape[0]))
        PDE_system = phsys.KdVSystem(x=x, eta=eta, gamma=gamma, nu=nu, force=F, force_jac=JF,
                                     init_sampler=phsys.initial_condition_kdv(x, eta))
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

    PDE_system.seed(seed)
    nstates = PDE_system.nstates

    if modelpath is not None:
        model, optimizer, metadict = phnn.load_dynamic_system_model(modelpath)
    else:
        if baseline==1:
            baseline_nn = phnn.PDEBaselineNN(nstates, hidden_dim,
                                                True, True, True,
                                                period=x_max)
            model = phnn.DynamicSystemNN(nstates, baseline_nn)
        elif baseline==2:
            baseline_nn = phnn.PDEBaselineSplitNN(nstates, hidden_dim,
                                                    F_timedependent, True, F_spacedependent,
                                                    period=x_max)
            # baseline_nn = phnn.PDEBaselineSplitNN(nstates, hidden_dim,
            #                                         True, True, True,
            #                                         period=x_max)
            model = phnn.DynamicSystemNN(nstates, baseline_nn)
        elif baseline==3:
            ext_forces_nn = phnn.PDEExternalForcesNN(PDE_system.nstates, hidden_dim=100,
                                                timedependent=True,
                                                spacedependent=True,
                                                statedependent=True,
                                                period=x_max)
            model = phnn.ConservativeDissipativeNN(nstates,
                                                    kernel_sizes=[3,3,3,1],
                                                    external_forces_est=ext_forces_nn
                                                    )
        else:
            ext_forces_nn = phnn.PDEExternalForcesNN(PDE_system.nstates, hidden_dim=100,
                                                timedependent=F_timedependent,
                                                spacedependent=F_spacedependent,
                                                statedependent=F_statedependent,
                                                period=x_max)
            if system == 'kdv':
                model = phnn.ConservativeDissipativeNN(nstates,
                                                       kernel_sizes=kernel_sizes,
                                                       external_forces_est=ext_forces_nn
                                                       )
            elif system == 'bbm':
                model = phnn.ConservativeDissipativeNN(nstates,
                                                       kernel_sizes=kernel_sizes,
                                                       symmetric_matrix=PDE_system.lhs_matrix_flat,
                                                       external_forces_est=ext_forces_nn
                                                       )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)

    traindata = phnn.generate_dataset(
        PDE_system, ntrajectories_train, t_sample, true_derivatives,
        nsamples=ntrainingpoints, noise_std=noise_std, xspatial=x)
    t_sample_val = np.linspace(0, .2, 21)
    valdata = phnn.generate_dataset(
        PDE_system, ntrajectories_val, t_sample_val, true_derivatives, noise_std=noise_std, xspatial=x)

    bestmodel, vloss = phnn.train(
        model,
        integrator,
        traindata,
        optimizer,
        valdata=valdata,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        l1_param_forces=l1_param_forces,
        l1_param_dissipation=l1_param_dissipation,
        loss_fn=torch.nn.MSELoss(),
        verbose=verbose,
        early_stopping_patience=early_stopping_patience,
        early_stopping_delta=early_stopping_delta,
        return_best=True,
        store_best=store_results,
        store_best_dir=storedir,
        modelname=modelname,
        trainingdetails=vars(args)
        )