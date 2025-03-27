#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2023-2025 Inria

import torch
from torch.nn.functional import mse_loss

import qpmpc_layers.lti as lti
from qpmpc_layers import Constraint, QPLayer, StageCost, TerminalCost
from qpmpc_layers.plants import LinearizedWheeledInvertedPendulum

NB_TIMESTEPS: int = 12

STEP_STATE_WEIGHT = 1e-4
STEP_INPUT_WEIGHT = 1e-6
TERMINAL_COST_WEIGHT = 1.0

if __name__ == "__main__":
    nx = 4
    nu = 1
    x_goal = torch.zeros(nx)

    ocp = lti.OptimalControlProblem(
        nb_steps=NB_TIMESTEPS,
        stage_cost=StageCost(
            Q=STEP_STATE_WEIGHT * torch.eye(nx),
            R=STEP_INPUT_WEIGHT * torch.eye(nu),
        ),
        dynamics=LinearizedWheeledInvertedPendulum(length=0.1, T=0.1),
        constraint=Constraint(
            C=torch.cat([+torch.eye(nx), -torch.eye(nx)]),
            D=None,
            e=None,
            f=torch.tensor([1e2] * (2 * nx)),
        ),
        terminal_cost=TerminalCost(
            Q=TERMINAL_COST_WEIGHT * torch.eye(nx),
            q=-TERMINAL_COST_WEIGHT * x_goal,
        ),
    )
    qplayer = QPLayer()
    integrator = lti.Integrator(
        NB_TIMESTEPS,
        dynamics=LinearizedWheeledInvertedPendulum(length=0.8, T=0.3)(),
    )

    optimizer = torch.optim.Adam(ocp.parameters(), lr=1e-2)
    for epoch in range(2_000):
        optimizer.zero_grad()
        x_init = (2.0 * torch.rand(nx) - 1.0) * torch.tensor(
            [
                0.5,  # ground position [m]
                0.1,  # pitch angle [rad]
                1.0,  # ground velocity [m/s]
                1.0,  # pitch angular velocity [rad/s]
            ]
        )
        qp_result = qplayer(ocp(x_init)).flatten()
        x_last_ocp = qp_result[-nx:]
        inputs = qp_result[nx :: nx + 1].reshape((NB_TIMESTEPS, nu))
        rollout = integrator(U=inputs, x_init=x_init)

        loss_type = "mse"
        if loss_type == "terminal":
            x_terminal = rollout[-nx:]
            loss = torch.linalg.vector_norm(x_terminal - x_goal, ord=2)
        elif loss_type == "mse":
            ocp_integrator = lti.Integrator(
                NB_TIMESTEPS,
                dynamics=ocp.dynamics(),
            )
            ocp_rollout = ocp_integrator(U=inputs, x_init=x_init)
            loss = mse_loss(ocp_rollout, rollout)
        else:
            raise Exception(f"unknown loss {loss_type}")

        loss.to("cuda")
        loss.backward(retain_graph=True)
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.2e}", end="")
        for name, value in ocp.named_parameters():
            print(f"\t{name}: {value.item()}", end="")
        print("")
