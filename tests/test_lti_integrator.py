#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

import torch

import qpmpc_layers.lti as lti
from qpmpc_layers import Dynamics
from qpmpc_layers.plants import DoubleIntegrator


def test_integrator_fixed_dynamics():
    nb_steps = 10
    dynamics: Dynamics = DoubleIntegrator(0.1)()
    integrator = lti.Integrator(
        nb_steps=nb_steps,
        dynamics=dynamics,
    )
    inputs = torch.arange(0.0, 2.0, 0.2)
    x_init = torch.zeros(2)
    states = integrator(U=inputs, x_init=x_init)
    nx = integrator.dynamics.A.shape[0]
    assert states.ndim == 1
    assert states.shape == ((nb_steps + 1) * nx,)
