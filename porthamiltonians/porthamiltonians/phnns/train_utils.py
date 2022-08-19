
import copy
import datetime
import os

import numpy as np
import torch
import torch.nn as nn

from .port_hamiltonian_neural_network import (PortHamiltonianNN,
                                              load_phnn_model,
                                              store_phnn_model)
from .models import load_baseline_model, store_baseline_model

__all__ = ['generate_dataset', 'train', 'compute_validation_loss',
           'batch_data', 'train_one_epoch', 'l1_loss_pHnn',
           'npoints_to_ntrajectories_tsample', 'EarlyStopping',
           'load_dynamic_system_model', 'store_dynamic_system_model']


def generate_dataset(pH_system, ntrajectories, t_sample,
                     true_derivatives=False, nsamples=None, seed=None,
                     noise_std=0., references=None, ttype=torch.float32):
    """
    Generate a dataset consisting of simulated data.

    Parameters
    ----------
    pH_system : porthamiltonians.phsystems.PortHamiltonianSystem
    ntrajectories : int
        Number of trajectories in dataset
    t_sample : (T, 1) ndarray
        Times to sample the trajectories at.
    true_derivatives : bool, default False
        If True, let the returned derivatives *dxdt* be the true
        value of the derivatives. If False, estimate derivatives
        with the finite differences.
    nsamples : int, default None
        Number of samples to keep. If smaller than ntrajectories*T,
        the last overflowing samples will be dropped. If None, all
        samples are kept.
    noise_std : number, default 0.
        Standard deviation of Gaussian white noise added to the
        samples of the trajectory.
    references : list of porthamiltonian.control.Reference, default
        None
            If the *pH_system* has a controller a list of ntrajectories
            reference objects may be passed.
    ttype : torch type, default torch.float32

    Returns
    -------

    (x_start, x_end, t_start, t_end, dt, u) : tuple
        x_start : (ntrajectories * (T-1), N) or (nsamples, N) tensor
        x_end : (ntrajectories * (T-1), N) or (nsamples, N) tensor
        t_start : (ntrajectories * (T-1), 1) or (nsamples, 1) tensor
        t_end : (ntrajectories * (T-1), 1) or (nsamples, 1) tensor
        dt : (ntrajectories * (T-1), 1) or (nsamples, 1) tensor
        u : (ntrajectories * (T-1), N) or (nsamples, N) tensor
    dxdt : (ntrajectories * (T-1), N) or (nsamples, N) tensor

    """

    if ntrajectories == 0:
        return None
    pH_system.seed(seed)
    nstates = pH_system.nstates
    traj_length = t_sample.shape[0]
    x = np.zeros((ntrajectories, traj_length, nstates))
    dxdt = np.zeros((ntrajectories, traj_length, nstates))
    t = np.zeros((ntrajectories, traj_length))
    u = np.zeros((ntrajectories, traj_length - 1, nstates))
    if references is None:
        references = [None] * ntrajectories

    for i in range(ntrajectories):
        x[i], dxdt[i], t[i], u[i] = pH_system.sample_trajectory(
            t_sample, noise_std=noise_std, reference=references[i])

    dt = torch.tensor([t[0, 1] - t[0, 0]], dtype=ttype)

    x_start = torch.tensor(x[:, :-1], dtype=ttype).reshape(-1, nstates)
    x_end = torch.tensor(x[:, 1:], dtype=ttype).reshape(-1, nstates)
    t_start = torch.tensor(t[:, :-1], dtype=ttype).reshape(-1, 1)
    t_end = torch.tensor(t[:, 1:], dtype=ttype).reshape(-1, 1)
    dt = dt*torch.ones_like(t_start, dtype=ttype)
    if pH_system.controller is None:
        u = torch.zeros_like(x_start, dtype=ttype)
    else:
        u = torch.tensor(u[:, :-1], dtype=ttype).reshape(-1, nstates)

    if not true_derivatives:
        dxdt = torch.tensor(dxdt[:, :-1], dtype=ttype).reshape(-1, nstates)
    else:
        dxdt = (x_end - x_start).clone().detach() / dt[0, 0]

    if nsamples is not None:
        x_start, x_end = x_start[:nsamples], x_end[:nsamples]
        t_start, t_end = t_start[:nsamples], t_end[:nsamples]
        dxdt = dxdt[:nsamples]

    return (x_start, x_end, t_start, t_end, dt, u), dxdt


def train(model, integrator, traindata, optimizer, valdata=None, epochs=1,
          batch_size=1, shuffle=False, l1_param_port=0.0,
          l1_param_dissipation=0.0, loss_fn=torch.nn.MSELoss(),
          batch_size_val=None, verbose=False, early_stopping_patience=None,
          early_stopping_delta=0., return_best=False, store_best=False,
          store_best_dir=None, modelname=None, trainingdetails={}):
    """
    Train a :py:meth:~`DynamicSystemNN`.

    Parameters
    ----------
    model : DynamicSystemNN
    integrator : str or False
        Specifies which solver to use during simulation. If False,
        the problem is left to scipy's solve_ivp. If 'euler',
        'midpoint', 'rk4' or 'srk4' the system is simulated with
        the forward euler method, the implicit midpoint method,
        the explicit Runge-Kutta 4 method or a symmetric fourth
        order Runge-Kutta method, respectively.
    traindata : tuple
        ((x_start, x_end, t_start, t_end, dt, u), dxdt)
    optimizer : torch optimizer
    valdata : tuple
        ((x_start, x_end, t_start, t_end, dt, u), dxdt)
        Validation loss is computed at the end of every epoch.
        Validation data is required to use early stopping.
    epochs : int, default 1
    batch_size : int, default 1
    shuffle : bool, delfault False
    l1_param_port : number, default 0.
        L1 penalty parameter punishing the size of the external port
        estimates.
    l1_param_dissipation : number, default 0.
        L1 penalty parameter punishing the size of the dissipation
        estimates.
    loss_fn : callable, default torch.nn.MSELoss()
    batch_size_val : int, default None
        If not provided, all of the validation data is in one batch.
    verbose : bool, default False
        If True, print information about training progress.
    early_stopping_patience : int, default None
        Patience for early stopping. If None, patience is infinite.
    early_stopping_delta : number, default 0.
    return_best : bool, default False
        If True, return a copy of the model achieving the lowest
        validation loss. If False, return the completely trained model.
    store_best : bool, default False
        If True, store the model every time a new lowest validation
        loss is achieved. If validation data is not provided, the
        model is saved after every epoch. If False, no model is saved.
    store_best_dir : str, default None
        Directory for storing the best model. If None, the
        models/<timestamp> directory is created and used.
    modelname : str, default None
        Directory for storing the best model. If None, the model name is
        set to <timestamp>.model. If a name is provided, the model will
        be overwritten.
    trainingdetails : dict, default {}
        Dictionary of information to store together with the model.
        The number of epochs trainged, loss and validation loss is
        added to this dict before saving a model.

    """

    traindata_batched = batch_data(traindata, batch_size, shuffle)
    if batch_size_val is not None:
        valdata_batched = batch_data(traindata, batch_size, False)
    else:
        valdata_batched = None

    vloss = None
    vloss_best = np.inf
    newbest = True
    early_stopping = None

    if early_stopping_patience is not None:
        early_stopping = EarlyStopping(patience=early_stopping_patience,
                                       min_delta=early_stopping_delta)
    if store_best:
        if store_best_dir is None:
            store_best_dir = 'models/' + str(datetime.datetime.now()).replace(
                '.', '').replace('-', '').replace(':', '').replace(' ', '')
        if not os.path.exists(store_best_dir):
            os.makedirs(store_best_dir)
        best_path = None

    best_model = model
    for epoch in range(epochs):
        if shuffle:
            traindata_batched = batch_data(traindata, batch_size, shuffle)
        model.train(True)
        start = datetime.datetime.now()
        avg_loss = train_one_epoch(model, traindata_batched, loss_fn,
                                   optimizer, integrator, l1_param_port,
                                   l1_param_dissipation)
        end = datetime.datetime.now()
        model.train(False)

        if verbose:
            print(f'\nEpoch {epoch}')
            print(f'Training loss: {np.format_float_scientific(avg_loss, 2)}')
            delta = end - start
            print('Epoch training time:'
                  f' {delta.seconds:d}.{int(delta.microseconds / 1e4):d}'
                  'seconds')

        if valdata is not None:
            start = datetime.datetime.now()
            vloss = compute_validation_loss(model, integrator, valdata,
                                            valdata_batched, loss_fn)
            end = datetime.datetime.now()
            if verbose:
                print(f'Validation loss: {np.format_float_scientific(vloss, 2)}')
                delta = end - start
                print('Validation loss computed in'
                      f' {delta.seconds:d}.{int(delta.microseconds / 1e4):d}'
                      'seconds')
            if vloss <= vloss_best:
                newbest = True
                if verbose:
                    print('New best validation loss')
                vloss_best = vloss
                if return_best:
                    best_model = copy.deepcopy(model)

            if early_stopping is not None:
                if early_stopping(vloss):
                    if verbose:
                        print(f'Early stopping at epoch {epoch}/{epochs}')
                    break
        else:
            newbest = True
        if store_best and newbest:
            if best_path is not None:
                os.remove(best_path)
            if modelname is None:
                best_path = os.path.join(
                    store_best_dir, str(datetime.datetime.now()).replace(
                        '.', '').replace('-', '').replace(':', '').replace(
                        ' ', '') + '.model')
            else:
                best_path = os.path.join(store_best_dir, modelname)
            trainingdetails['epochs'] = epoch
            trainingdetails['val_loss'] = vloss
            trainingdetails['train_loss'] = avg_loss
            store_dynamic_system_model(best_path, model, optimizer,
                                       **trainingdetails)
            if verbose:
                print(f'Stored new best model {best_path}')
            newbest = False

    return best_model, vloss


def compute_validation_loss(model, integrator, valdata=None,
                            valdata_batched=None, loss_fn=torch.nn.MSELoss()):
    """
    Compute the validation loss of *model*.

    Parameters
    ----------
    model : DynamicSystemNN
    integrator : str or False
        Specifies which solver to use during simulation. If False,
        the problem is left to scipy's solve_ivp. If 'euler',
        'midpoint', 'rk4' or 'srk4' the system is simulated with
        the forward euler method, the implicit midpoint method,
        the explicit Runge-Kutta 4 method or a symmetric fourth
        order Runge-Kutta method, respectively.
    valdata : tuple, default None
        ((x_start, x_end, t_start, t_end, dt, u), dxdt)
        Either valdata or valdata_batched must be provided.
    valdata_batched : list of tuples, default None
        Batched validation data. A list of tuples containing validation
        data batches. Either valdata or valdata_batched must be
        provided.
    loss_fn : callable, default torch.nn.MSELoss()

    Returns
    -------
    vloss : float

    """

    vloss = 0
    if valdata_batched is not None:
        for (input_tuple, dxdt) in valdata_batched:
            dxdt_hat = model.time_derivative(integrator, *input_tuple)
            vloss += loss_fn(dxdt_hat, dxdt)
        vloss = vloss / len(valdata_batched)
    else:
        dxdt_hat = model.time_derivative(integrator, *valdata[0])
        vloss = loss_fn(dxdt_hat, valdata[1])
    return float(vloss.detach().numpy())


def batch_data(data, batch_size, shuffle):
    """
    Partition *data* into batches of *batch_size*.

    Parameters
    ----------
    data : tuple
        ((x_start, x_end, t_start, t_end, dt, u), dxdt)
    batch_size : int
    shuffle : bool

    Returns
    -------
    batches : list of tuples

    """

    nsamples = data[1].shape[0]
    if shuffle:
        permutation = torch.randperm(nsamples)
    else:
        permutation = torch.arange(nsamples)
    nbatches = np.ceil(nsamples / batch_size).astype(int)
    batched = [(None, None)]*nbatches
    for i in range(0, nbatches):
        indices = permutation[i*batch_size:(i+1)*batch_size]
        input_tuple = [data[0][j][indices] for j in range(len(data[0]))]
        dxdt = data[1][indices]
        batched[i] = (input_tuple, dxdt)
    return batched


def train_one_epoch(model, traindata_batched, loss_fn, optimizer, integrator,
                    l1_param_port, l1_param_dissipation):
    """
    Train *model* for one epoch.

    Parameters
    ----------
    model : DynamicSystemNN
    traindata_batched : list of tuples
    loss_fn : callable
    optimizer : torch optimizer
    integrator : str or False
        Specifies which solver to use during simulation. If False,
        the problem is left to scipy's solve_ivp. If 'euler',
        'midpoint', 'rk4' or 'srk4' the system is simulated with
        the forward euler method, the implicit midpoint method,
        the explicit Runge-Kutta 4 method or a symmetric fourth
        order Runge-Kutta method, respectively.
    l1_param_port : number
    l1_param_dissipation : number

    Returns
    -------
    training_loss : number

    """

    running_loss = 0.
    optimizer.zero_grad()
    for (input_tuple, dxdt) in traindata_batched:
        with torch.cuda.amp.autocast():
            dxdt_hat = model.time_derivative(integrator, *input_tuple)
            loss = loss_fn(dxdt_hat, dxdt)
            if (isinstance(model, PortHamiltonianNN)
                    and ((l1_param_port > 0) or (l1_param_dissipation > 0))):
                loss += l1_loss_pHnn(model, l1_param_port, l1_param_dissipation,
                                     input_tuple[0], input_tuple[2])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

    return running_loss / len(traindata_batched)


def l1_loss_pHnn(pHnn_model, l1_param_port, l1_param_dissipation, x, t=None):
    """
    Compute L1 penalty loss of external port and dissipation terms::

        L1 = l1_param_port * | f(x, t) | + l1_param_dissipation * | R(x) |

    Parameters
    ----------
    pHnn_model : PortHamiltonianNN
    l1_param_port : number
    l1_param_dissipation : number
    x : (nsamples, N) tensor
    t : (nsamples, 1) tensor, default None

    Returns
    -------
    penalty : number

    """
    penalty = 0
    if (isinstance(pHnn_model.external_port, nn.Module) and (l1_param_port > 0)
            and (not pHnn_model.external_port_provided)):
        penalty += l1_param_port*torch.abs(pHnn_model.external_port(x, t)).mean()
    if (isinstance(pHnn_model.R, nn.Module) and (l1_param_dissipation > 0)
            and (not pHnn_model.R_provided)):
        penalty += l1_param_dissipation*torch.abs(pHnn_model.R(x)).mean()

    return penalty


def npoints_to_ntrajectories_tsample(npoints, tmax, dt):
    """
    Compute number of trajectories and the necessary sample time to
    achieve *npoints* samples from trajectories lasting *tmax* time with
    a dt sample rate.

    Parameters
    ----------
    npoints : int
    tmax : number
    dt : number

    Returns
    -------
    int
    (npoints, ) ndarray

    """

    points_per_trajectory = round(tmax / dt)
    t_sample = np.linspace(0, tmax, points_per_trajectory + 1)
    return int(np.ceil(npoints / points_per_trajectory)), t_sample[:npoints+1]


class EarlyStopping():
    """
    Keeps track of whether the validation loss has decreased by at least
    *min_delta* for the last *patience* calls.

    Parameters
    ----------
    patience : int, default None
        If None, patience is infinite
    min_delta : number, default 0.

    """

    def __init__(self, patience=None, min_delta=0.):
        if patience is None:
            self.patience = np.inf
        else:
            self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        
    def __call__(self, val_loss):
        """
        Checks if *val_loss* is at least *self.min_delta* smaller
        than the so far observed lowest validation loss. If True,
        the internal counter is reset. If False, the internal counter
        is increased.

        Parameters
        ----------
        val_loss : number

        Returns
        -------
        bool
            If the counter exceeds self.patience, True is returned as a
            sign to early stop. If not, False.

        """

        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def load_dynamic_system_model(modelpath):
    """
    Loads a DynamicSystemNN from disk. See
    :py:meth:~`port_hamiltonian_neural_network.load_phnn_model` and
    :py:meth:~`model.load_baseline_model`.
    """

    metadict = torch.load(modelpath)
    if 'structure_matrix' in metadict.keys():
        return load_phnn_model(modelpath)
    else:
        return load_baseline_model(modelpath)


def store_dynamic_system_model(storepath, model, optimizer, **kwargs):
    """
    Stores a DynamicSystemNN to disk. See
    :py:meth:~`port_hamiltonian_neural_network.store_phnn_model` and
    :py:meth:~`model.store_baseline_model`.
    """

    if isinstance(model, PortHamiltonianNN):
        store_phnn_model(storepath, model, optimizer, **kwargs)
    else:
        store_baseline_model(storepath, model, optimizer, **kwargs)
