#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2025 Inria

import numpy as np
import torch
from matplotlib import pyplot

import qpmpc_layers.lti as lti
from qpmpc_layers import Constraint, QPLayer, StageCost, TerminalCost
from qpmpc_layers.plants import TripleIntegrator

MAX_ACCEL = 3.0  # [m] / [s]²
HORIZON_DURATION = 1.0  # [s]
NB_TIMESTEPS = 16

STEP_STATE_WEIGHT = 1e-10
STEP_INPUT_WEIGHT = 1e-6
TERMINAL_COST_WEIGHT = 1.0


def plot_trajectory(t, X: torch.Tensor):
    if X.ndim == 1:
        X = X.reshape((X.shape[0] // 3, 3))
        positions, velocities, accelerations = X[:, 0], X[:, 1], X[:, 2]
    pyplot.figure(1)
    pyplot.plot(t, positions.detach().numpy())
    pyplot.plot(t, velocities.detach().numpy())
    pyplot.plot(t, accelerations.detach().numpy())
    pyplot.grid(True)


def plot_inputs(t, U, label="jerk"):
    pyplot.figure(2)
    pyplot.plot(t[:-1], U.detach().numpy().flatten())
    pyplot.grid(True)


if __name__ == "__main__":
    nu = 1
    nx = 3
    x_goal = torch.tensor([1.0, 0.0, 0.0])

    ocp_dynamics = TripleIntegrator(0.05)
    integrator_dynamics = TripleIntegrator(0.2)()
    accel_from_state = torch.tensor([[0.0, 0.0, 1.0]])

    ocp = lti.OptimalControlProblem(
        nb_steps=NB_TIMESTEPS,
        stage_cost=StageCost(
            Q=STEP_STATE_WEIGHT * torch.eye(nx),
            R=STEP_INPUT_WEIGHT * torch.eye(nu),
        ),
        dynamics=ocp_dynamics,
        constraint=Constraint(
            C=torch.cat([+accel_from_state, -accel_from_state]),
            D=None,
            e=None,
            f=torch.tensor([+MAX_ACCEL, +MAX_ACCEL]),
        ),
        terminal_cost=TerminalCost(
            Q=TERMINAL_COST_WEIGHT * torch.eye(nx),
            q=-TERMINAL_COST_WEIGHT * x_goal,
        ),
    )
    qplayer = QPLayer()

    integrator = lti.Integrator(NB_TIMESTEPS, integrator_dynamics)

    optimizer = torch.optim.SGD(ocp.parameters(), lr=1e-4)

    for epoch in range(100):
        optimizer.zero_grad()
        x_init = torch.rand(nx)
        qp_result = qplayer(ocp(x_init)).flatten()
        inputs = qp_result[nx :: nx + 1].reshape((NB_TIMESTEPS, 1))
        x_terminal = integrator(U=inputs, x_init=x_init)[-nx:]

        loss = torch.linalg.vector_norm(x_terminal - x_goal, ord=2)
        loss.to("cuda")
        loss.backward(retain_graph=True)
        optimizer.step()
        name, value = next(ocp.named_parameters())
        print(
            f"Epoch {epoch + 1}, Loss: {loss.item()}, {name}: {value.item()}"
        )

    x_init = torch.zeros(3)
    qp_result = qplayer(ocp(x_init)).flatten()
    episode = qp_result[:-3].reshape((NB_TIMESTEPS, nx + nu))
    terminal_state = qp_result[-3:]
    predicted_states = torch.cat(
        [
            episode[:, :nx],
            terminal_state.reshape((1, nx)),
        ]
    ).flatten()
    inputs = qp_result[nx :: nx + 1].reshape((NB_TIMESTEPS, 1))
    states = integrator(U=inputs, x_init=x_init)

    t = np.linspace(0.0, HORIZON_DURATION, NB_TIMESTEPS + 1)
    plot_trajectory(t, predicted_states)
    plot_trajectory(t, states)
    pyplot.legend(
        (
            "predicted_position",
            "predicted_velocity",
            "predicted_acceleration",
            "actual_position",
            "actual_velocity",
            "actual_acceleration",
        )
    )
    plot_inputs(t, inputs)
    pyplot.legend(("inputs",))
    pyplot.show(block=True)
