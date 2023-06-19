
import argparse

import numpy as np
import torch

from phlearn.phsystems.ode import init_tanksystem, init_msdsystem
from phlearn.phnns import PseudoHamiltonianNN, DynamicSystemNN, load_dynamic_system_model
from phlearn.phnns import R_estimator, BaselineNN, BaselineSplitNN, HamiltonianNN, ExternalForcesNN
from phlearn.phnns import npoints_to_ntrajectories_tsample, train, generate_dataset

ttype = torch.float32
torch.set_default_dtype(ttype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, choices=['tank', 'msd'],
                        required=True,
                        help='Choose to train a tank or a '
                             'forced mass spring damper.')
    parser.add_argument('--baseline', type=int, default=0,  choices=[0, 1, 2],
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
                        choices=[False, 'euler', 'rk4', 'midpoint', 'srk4'],
                        default='midpoint',
                        help='Integrator used during training.')
    parser.add_argument('--F_timedependent', type=int, default=1,
                        choices=[0, 1],
                        help='If 1, make external force NN (or baseline NN) '
                             'depend on time.')
    parser.add_argument('--F_statedependent', type=int, default=1,
                        choices=[0, 1],
                        help='If 1, make external force NN (or baseline NN) '
                             'depend on state.')
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
                        help='L1 penalty parameter of external force estimate.')
    parser.add_argument('--l1_param_dissipation', type=float, default=0.,
                        help='L1 penalty parameter of dissipation estimate.')
    parser.add_argument('--early_stopping_patience', type=int,
                        help=('Number of epochs to continue training without '
                              'a decrease in validation loss '
                              'of at least early_stopping_delta.'))
    parser.add_argument('--early_stopping_delta', type=float,
                        help='Minimum accepted decrease in validation loss to '
                             'prevent early stopping.')
    parser.add_argument('--shuffle', action='store_true',
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
    baseline = bool(args.baseline)
    storedir = args.storedir
    modelname = args.modelname
    modelpath = args.modelpath
    ntrainingpoints = args.ntrainingpoints
    sampling_time = args.sampling_time
    t_max = args.t_max
    true_derivatives = args.true_derivatives
    if true_derivatives:
        integrator = False
        print('Warning: As exact derivatives are used when generating '
              'training data, (true_derivatives = True) integrator'
              'is set to False.')
    else:
        integrator = args.integrator
    F_timedependent = bool(args.F_timedependent)
    F_statedependent = bool(args.F_statedependent)
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

    ntrajectories_train, t_sample = npoints_to_ntrajectories_tsample(
        ntrainingpoints, t_max, sampling_time)

    if system == 'tank':
        pH_system = init_tanksystem()
        damped_states = np.arange(pH_system.nstates) < pH_system.npipes
    else:
        pH_system = init_msdsystem()
        damped_states = [False, True]

    pH_system.seed(seed)
    nstates = pH_system.nstates

    if modelpath is not None:
        model, optimizer, metadict = load_dynamic_system_model(modelpath)
    else:
        if baseline == 1:
            baseline_nn = BaselineNN(
                nstates, hidden_dim,
                timedependent=F_timedependent, statedependent=True)
            model = DynamicSystemNN(nstates, baseline_nn)
        elif baseline == 2:
            external_forces_filter_t = np.zeros(nstates)
            external_forces_filter_t[-1] = 1
            baseline_nn = BaselineSplitNN(
                nstates, hidden_dim, noutputs_x=nstates,
                noutputs_t=1, external_forces_filter_x=None,
                external_forces_filter_t=external_forces_filter_t,
                ttype=ttype)
            model = DynamicSystemNN(nstates, baseline_nn)
        else:
            hamiltonian_nn = HamiltonianNN(nstates, hidden_dim)
            external_forces_filter = np.zeros(nstates)
            external_forces_filter[-1] = 1
            ext_forces_nn = ExternalForcesNN(
                nstates, 1, hidden_dim=hidden_dim,
                timedependent=F_timedependent,
                statedependent=F_statedependent,
                external_forces_filter=external_forces_filter)

            r_est = R_estimator(damped_states)

            model = PseudoHamiltonianNN(
                nstates,
                pH_system.skewsymmetric_matrix,
                hamiltonian_est=hamiltonian_nn,
                dissipation_est=r_est,
                external_forces_est=ext_forces_nn)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                     weight_decay=1e-4)

    traindata = generate_dataset(
        pH_system, ntrajectories_train, t_sample, true_derivatives,
        nsamples=ntrainingpoints, noise_std=noise_std)
    valdata = generate_dataset(
        pH_system, ntrajectories_val, t_sample, true_derivatives, noise_std=noise_std)

    bestmodel, vloss = train(
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
