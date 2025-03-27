#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2025 Inria

import numpy as np
import qpmpc
import qpsolvers
import torch
from matplotlib import pyplot

import qpmpc_layers.lti as lti
from qpmpc_layers import (
    QP,
    Constraint,
    Dynamics,
    QPLayer,
    StageCost,
    TerminalCost,
)

MAX_ACCEL = 3.0  # [m] / [s]²
HORIZON_DURATION = 1.0  # [s]
NB_TIMESTEPS = 16
T = HORIZON_DURATION / NB_TIMESTEPS

# x' = A x + B u
DYN_A = np.array(
    [[1.0, T, T**2 / 2.0], [0.0, 1.0, T], [0.0, 0.0, 1.0]], dtype=np.float32
)
DYN_B = np.array([T**3 / 6.0, T**2 / 2.0, T], dtype=np.float32).reshape((3, 1))

# C x <= e
accel_from_state = np.array([0.0, 0.0, 1.0])
INEQ_C = np.vstack([+accel_from_state, -accel_from_state], dtype=np.float32)
INEQ_VEC = np.array([+MAX_ACCEL, +MAX_ACCEL], dtype=np.float32)

problem = qpmpc.MPCProblem(
    transition_state_matrix=DYN_A,
    transition_input_matrix=DYN_B,
    ineq_state_matrix=INEQ_C,
    ineq_input_matrix=None,
    ineq_vector=INEQ_VEC,
    initial_state=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    goal_state=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    nb_timesteps=NB_TIMESTEPS,
    terminal_cost_weight=1.0,
    stage_state_cost_weight=None,
    stage_input_cost_weight=1e-6,
)


def plot_trajectory(t, X):
    positions, velocities, accelerations = X[:, 0], X[:, 1], X[:, 2]
    pyplot.figure(1)
    pyplot.plot(t, positions)
    pyplot.plot(t, velocities)
    pyplot.plot(t, accelerations)
    pyplot.grid(True)


def plot_inputs(t, U, label="jerk"):
    pyplot.figure(2)
    pyplot.plot(t[:-1], U.flatten())
    pyplot.grid(True)


if __name__ == "__main__":
    # Version 1: qpmpc
    mpc_qp = qpmpc.MPCQP(problem)
    result = qpsolvers.solve_problem(mpc_qp.problem, solver="proxqp")
    plan = qpmpc.Plan(problem, result)

    # Version 2: QPLayer on a qpmpc QP
    P, q, G, h, A, b, _, _ = [
        (
            torch.tensor(x, requires_grad=True)
            if x is not None
            else torch.empty(0)
        )
        for x in mpc_qp.problem.unpack()
    ]
    l = -1.0e20 * torch.ones(h.shape)
    qp2 = QP(P, q, A, b, G, l, h)
    qp2_solution = QPLayer()(qp2)
    U2 = qp2_solution.detach().numpy()

    # Version 3: QPLayer on a qpmpc_layers QP
    nx = 3
    nu = 1
    ocp = lti.OptimalControlProblem(
        nb_steps=NB_TIMESTEPS,
        stage_cost=StageCost(
            Q=1e-10 * torch.eye(nx),
            R=problem.stage_input_cost_weight * torch.eye(nu),
        ),
        dynamics=Dynamics(
            A=torch.tensor(DYN_A, requires_grad=True),
            B=torch.tensor(DYN_B, requires_grad=True),
        ),
        constraint=Constraint(
            C=torch.tensor(INEQ_C),
            D=None,
            e=None,
            f=torch.tensor(INEQ_VEC),
        ),
        terminal_cost=TerminalCost(
            Q=problem.terminal_cost_weight * torch.eye(nx),
            q=problem.terminal_cost_weight * torch.tensor(-problem.goal_state),
        ),
    )
    qplayer = QPLayer()
    x_0 = torch.tensor(problem.initial_state)
    result3 = qplayer(ocp(x_init=x_0))
    full3 = result3.detach().numpy().flatten()
    XU3 = full3[:-3].reshape((NB_TIMESTEPS, nx + nu))
    X3 = np.vstack([XU3[:, :nx], full3[-nx:].reshape((1, nx))])
    U3 = XU3[:, nx:]
    U3 = full3[nx :: nx + 1]

    loss = torch.linalg.norm(result3, ord=2)
    loss.to("cuda")
    loss.backward()

    integrator = lti.Integrator(NB_TIMESTEPS, ocp.dynamics)
    X4 = integrator(U=torch.tensor(U3), x_init=x_0)
    X4 = X4.detach().numpy().flatten()
    X4 = X4.reshape((X4.shape[0] // 3, 3))

    U1 = plan.inputs
    X1 = plan.states
    t = np.linspace(0.0, HORIZON_DURATION, NB_TIMESTEPS + 1)
    plot_trajectory(t, X1)
    plot_trajectory(t, X3)
    plot_trajectory(t, X4)
    pyplot.legend(
        (
            "position_qpmpc",
            "velocity_qpmpc",
            "acceleration_qpmpc",
            "position_layers",
            "velocity_layers",
            "acceleration_layers",
            "position_integrator",
            "velocity_integrator",
            "acceleration_integrator",
        )
    )
    plot_inputs(t, U1)
    plot_inputs(t, U2)
    plot_inputs(t, U3)
    pyplot.legend(
        (
            "original",
            "layer",
            "pipeline",
        )
    )
    pyplot.show(block=True)
