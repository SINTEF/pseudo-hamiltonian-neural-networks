# When developing tests, from the top level dir phlearn/ run
# `python3 -m phlearn.phnns.tests.test_phnns`
# to run this as a module, allowing use of relative imports
# otherwise run `pytest` from root to run as a test suite. 

# Disclaimer: a lot of this code was written by ChatGPT and then corrected afterwards. 

import numpy as np
from ..dynamic_system_neural_network import DynamicSystemNN
from ..models import BaseNN, HamiltonianNN, ExternalForcesNN
from copy import copy 

import torch
import pytest

NSTATES = 5
NOUTPUTS = 2
NSAMPLES = 3

kwargs = dict(
    nstates=NSTATES,
    noutputs=NOUTPUTS,
    hidden_dim=10,
    timedependent=True,
    statedependent=True,
    ttype=torch.float32,
)


def test_base_nn_forward_with_state_and_time():
    base_nn = BaseNN(**kwargs)
    x = torch.rand((NSAMPLES, NSTATES))
    t = torch.rand((NSAMPLES, 1))
    output = base_nn(x=x, t=t)
    assert output.shape == (NSAMPLES, NOUTPUTS)


def test_base_nn_forward_without_time():
    _kwargs = copy(kwargs)
    _kwargs["timedependent"] = False
    base_nn = BaseNN(**_kwargs)
    x = torch.rand((NSAMPLES, NSTATES))
    output = base_nn(x=x)
    assert output.shape == (NSAMPLES, NOUTPUTS)


def test_base_nn_forward_without_state():
    _kwargs = copy(kwargs)
    _kwargs["statedependent"] = False
    print(_kwargs)
    base_nn = BaseNN(**_kwargs)
    t = torch.rand((NSAMPLES, 1))
    output = base_nn(t=t)
    assert output.shape == (NSAMPLES, NOUTPUTS)


def test_base_nn_forward_without_state_or_time():
    _kwargs = copy(kwargs)
    _kwargs["timedependent"] = False
    _kwargs["statedependent"] = False
    base_nn = BaseNN(**_kwargs)
    output = base_nn()
    assert output is not None


def test_base_nn_trainable_parameters():
    base_nn = BaseNN(**kwargs)
    base_nn.train()
    for param in base_nn.parameters():
        assert param.requires_grad == True

def test_HamiltonianNN_output_shape():
        batch_size = 10
        nstates = 2
        x = torch.rand(batch_size, nstates)
        model = HamiltonianNN(nstates)
        output = model(x=x)
        assert output.shape == (batch_size, 1)