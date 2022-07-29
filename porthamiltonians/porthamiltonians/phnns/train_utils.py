
import copy
import datetime
import os

import numpy as np
import torch
import torch.nn as nn

from .port_hamiltonian_neural_network import PortHamiltonianNN, load_phnn_model, store_phnn_model
from .models import load_baseline_model, store_baseline_model


def generate_dataset(pH_system, integrator, ntrajectories, t_sample, nsamples=None,
                     seed=None, noise_std=0., control_refs=None, ttype=torch.float32):
    if ntrajectories == 0:
        return None
    pH_system.seed(seed)
    nstates = pH_system.nstates
    traj_length = t_sample.shape[0]
    x = np.zeros((ntrajectories, traj_length, nstates))
    dxdt = np.zeros((ntrajectories, traj_length, nstates))
    t = np.zeros((ntrajectories, traj_length))
    u = np.zeros((ntrajectories, traj_length - 1, nstates))
    if control_refs is None:
        control_refs = [None] * ntrajectories

    for i in range(ntrajectories):
        x[i], dxdt[i], t[i], u[i] = pH_system.sample_trajectory(t_sample, noise_std=noise_std, reference=control_refs[i])

    dt = torch.tensor([t[0, 1] - t[0, 0]], dtype=ttype)

    # Set up pairs of succesive points to train on an integrator
    x_start = torch.tensor(x[:, :-1], dtype=ttype).reshape(-1, nstates)
    x_end = torch.tensor(x[:, 1:], dtype=ttype).reshape(-1, nstates)
    t_start = torch.tensor(t[:, :-1], dtype=ttype).reshape(-1, 1)
    t_end = torch.tensor(t[:, 1:], dtype=ttype).reshape(-1, 1)
    dt = dt*torch.ones_like(t_start, dtype=ttype)
    if pH_system.controller is None:
        u = torch.zeros_like(x_start, dtype=ttype)
    else:
        u = torch.tensor(u[:, :-1], dtype=ttype).reshape(-1, nstates)

    if not integrator:
        dxdt = torch.tensor(dxdt[:, :-1], dtype=ttype).reshape(-1, nstates)
    else:
        dxdt = (x_end - x_start).clone().detach() / dt[0, 0]

    if nsamples is not None:
        x_start, x_end = x_start[:nsamples], x_end[:nsamples]
        t_start, t_end = t_start[:nsamples], t_end[:nsamples]
        dxdt = dxdt[:nsamples]

    return (x_start, x_end, t_start, t_end, dt, u), dxdt


def train(model, integrator, traindata, optimizer, valdata=None, epochs=1, batch_size=1,
          shuffle=False, l1_param_port=0.0, l1_param_dissipation=0.0, loss_fn=torch.nn.MSELoss(),
          batch_size_val=None, verbose=False, early_stopping_patience=None, early_stopping_delta=0.,
          return_best=False, store_best=False, store_best_dir=None, modelname=None, trainingdetails={}):
    
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
        early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_delta)
    if store_best:
        if store_best_dir is None:
            store_best_dir = 'models/' + str(datetime.datetime.now()).replace('.', '').replace('-', '').replace(':', '').replace(' ', '')
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
                                   optimizer, integrator, l1_param_port, l1_param_dissipation)
        end = datetime.datetime.now()
        model.train(False)

        if verbose:
            print(f'\nEpoch {epoch}')
            print(f'Training loss: {np.format_float_scientific(avg_loss, 2)}')
            delta = end - start
            print(f'Epoch training time: {delta.seconds:d}.{int(delta.microseconds / 1e4):d} seconds')

        if valdata is not None:
            start = datetime.datetime.now()
            vloss = compute_validation_loss(model, integrator, valdata, valdata_batched, loss_fn)
            end = datetime.datetime.now()
            if verbose:
                print(f'Validation loss: {np.format_float_scientific(vloss, 2)}')
                delta = end - start
                print(f'Validation loss computed in {delta.seconds:d}.{int(delta.microseconds / 1e4):d} seconds')
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
                    store_best_dir, str(datetime.datetime.now()).replace('.', '').replace('-', '').replace(':', '').replace(' ', '') + '.model')
            else:
                best_path = os.path.join(store_best_dir, modelname)
            trainingdetails['epochs'] = epoch
            trainingdetails['val_loss'] = vloss
            trainingdetails['train_loss'] = avg_loss
            store_dynamic_system_model(best_path, model, optimizer, **trainingdetails)
            if verbose:
                print(f'Stored new best model {best_path}')
            newbest = False

    return best_model, vloss


def compute_validation_loss(model, integrator, valdata=None, valdata_batched=None, loss_fn=torch.nn.MSELoss()):
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


def train_one_epoch(model, traindata_batched, loss_fn, optimizer, integrator, l1_param_port, l1_param_dissipation):
    running_loss = 0.
    optimizer.zero_grad()
    for (input_tuple, dxdt) in traindata_batched:
        with torch.cuda.amp.autocast():
            dxdt_hat = model.time_derivative(integrator, *input_tuple)
            loss = loss_fn(dxdt_hat, dxdt)
            if isinstance(model, PortHamiltonianNN) and ((l1_param_port > 0) or (l1_param_dissipation > 0)):
                loss += l1_loss_pHnn(model, l1_param_port, l1_param_dissipation, input_tuple[0], input_tuple[2])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

    return running_loss / len(traindata_batched)


def l1_loss_pHnn(pHnn_model, l1_param_port, l1_param_dissipation, x, t=None):
    penalty = 0
    if (isinstance(pHnn_model.external_port, nn.Module) and (l1_param_port > 0)
            and (not pHnn_model.external_port_provided)):
        penalty += l1_param_port*torch.abs(pHnn_model.external_port(x, t)).mean()
    if (isinstance(pHnn_model.R, nn.Module) and (l1_param_dissipation > 0)
            and (not pHnn_model.R_provided)):
        penalty += l1_param_dissipation*torch.abs(pHnn_model.R(x)).mean()

    return penalty


def npoints_to_ntrajectories_tsample(npoints, tmax, dt):
    points_per_trajectory = round(tmax / dt)
    t_sample = np.linspace(0, tmax, points_per_trajectory + 1)
    return int(np.ceil(npoints / points_per_trajectory)), t_sample[:npoints+1]


class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        
    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def load_dynamic_system_model(modelpath):
    metadict = torch.load(modelpath)
    if 'structure_matrix' in metadict.keys():
        return load_phnn_model(modelpath)
    else:
        return load_baseline_model(modelpath)


def store_dynamic_system_model(storepath, model, optimizer, **kwargs):
    if isinstance(model, PortHamiltonianNN):
        store_phnn_model(storepath, model, optimizer, **kwargs)
    else:
        store_baseline_model(storepath, model, optimizer, **kwargs)
