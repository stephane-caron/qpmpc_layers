#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Initial step of an optimal control problem."""

from typing import Optional, Union

import torch
import torch.nn as nn

from ..exceptions import ShapeMismatch
from ..optimal_control import Start, Step, Terminate, Wrap
from ..types import QP, Constraint, Dynamics, StageCost, TerminalCost


class OptimalControlProblem(nn.Module):
    r"""Optimal control problem for a linear time-invariant system.

    The cost function of our problem is the sum of
    :class:`qpmpc_layers.types.StageCost`'s and a
    :class:`qpmpc_layers.types.TerminalCost`:

    .. math::

        J(X_N, U_N) = \sum_{k=0}^{N-1} \ell(x_k, u_k) + \ell(x_N)

    where :math:`X_N = (x_0, \ldots, x_N)` is the state trajectory vector and
    :math:`U_N = (u_0, \ldots, u_{N-1})` is the input trajectory vector. Our
    overall cost is quadratic, so that the optimal control problem can be cast
    as a :class:`qpmpc_layers.types.QP`.

    Attributes:
        constraint: Optional inequality constraints on the state and action of
            a step.
        dynamics: Dynamics of the system.
        stage_cost: Contribution of an action step to the overall optimization
            cost.
        terminal_cost: Terminal-state cost.
    """

    _qp: QP
    _wrap: Wrap
    constraint: Constraint
    dynamics: Dynamics
    stage_cost: StageCost
    terminal_cost: TerminalCost

    def __init__(
        self,
        nb_steps: int,
        stage_cost: Union[StageCost, nn.Module],
        dynamics: Union[Dynamics, nn.Module],
        terminal_cost: Optional[Union[TerminalCost, nn.Module]] = None,
        constraint: Optional[Union[Constraint, nn.Module]] = None,
    ):
        """Initialize module.

        Raises:
            DataTypeError: If some tensor data types are incompatible.
        """
        super().__init__()
        self.constraint = constraint
        self.dynamics = dynamics
        self.nb_steps = nb_steps
        self.stage_cost = stage_cost
        self.terminal_cost = terminal_cost
        #
        self.__check_shapes()

    def __check_shapes(self):
        """Check consistency of cost and dynamics matrices.

        Raises:
            ShapeMismatch: If some matrix shapes don't match.

        This dynamic type check is meant to be run from the constructor only.
        """
        stage_cost: StageCost = (
            self.stage_cost()
            if isinstance(self.stage_cost, nn.Module)
            else self.stage_cost
        )
        dynamics: Dynamics = (
            self.dynamics()
            if isinstance(self.dynamics, nn.Module)
            else self.dynamics
        )
        if stage_cost.R.shape[1] != dynamics.B.shape[1]:
            raise ShapeMismatch(
                "stage_cost.R",
                stage_cost.R.shape,
                "dynamics.B",
                dynamics.B.shape,
            )

    def forward(self, x_init: torch.Tensor) -> QP:
        """Forward computation performed by the module.

        Args:
            x_init: Initial state of the problem.

        Returns:
            QP for the OCP starting from the prescribed initial state.
        """
        stage_cost: StageCost = (
            self.stage_cost()
            if isinstance(self.stage_cost, nn.Module)
            else self.stage_cost
        )
        dynamics: Dynamics = (
            self.dynamics()
            if isinstance(self.dynamics, nn.Module)
            else self.dynamics
        )
        constraint: Optional[Constraint] = (
            self.constraint()
            if isinstance(self.constraint, nn.Module)
            else self.constraint
        )
        terminal_cost: Optional[TerminalCost] = (
            self.terminal_cost()
            if isinstance(self.terminal_cost, nn.Module)
            else self.terminal_cost
        )
        state_dim = dynamics.A.shape[1]
        start = Start(state_dim)
        step = Step(
            cost=stage_cost,
            dynamics=dynamics,
            constraint=constraint,
        )
        wrap = Wrap(state_dim)
        qp: QP = start()
        for _ in range(self.nb_steps):
            qp = step(qp)
        if terminal_cost is not None:
            terminate = Terminate(terminal_cost)
            qp = terminate(qp)
        return wrap(qp, x_init)
