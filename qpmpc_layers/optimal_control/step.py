#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Transition of the optimal control problem."""

from typing import Optional

import torch
import torch.nn as nn

from ..types import QP, Constraint, Dynamics, StageCost


class Step(nn.Module):
    r"""Transition of the optimal control problem.

    An action step from step :math:`k` extends the episode vector :math:`\tau_k
    = (x_0, u_0, \ldots, x_{k-1}, u_{k-1}, x_k)` with the pair :math:`(u_k,
    x_{k+1})`, where :math:`u_k` is the action taken at step :math:`k` and
    :math:`x_{k+1}` is the next state after applying transition
    :class:`Dynamics`:

    .. math::

        x_{k+1} = A x_k + B u_k

    The action step contributes a :class:`qpmpc_layers.types.StageCost`
    :math:`\ell(x_k, u_k)` of the action step to the total cost:

    .. math::

        \ell(x_k, u_k) = \frac{1}{2} x_k^T Q x_k + q^T x_k +
        \frac{1}{2} u_k^T R u_k + r^T u_k

    We can also apply a state-action inequality :class:`Constraint` to the
    step:

    .. math::

        e \leq C x_k + D u_k \leq f

    Attributes:
        cost: Contribution of the action step to the overall optimization cost.
        dynamics: Linear dynamics of the action step.
        constraint: Optional inequality constraints on the state and action of
            the step.
    """

    def __init__(
        self,
        cost: StageCost,
        dynamics: Dynamics,
        constraint: Optional[Constraint] = None,
    ):
        """Initialize module."""
        super().__init__()
        self.cost = cost
        self.dynamics = dynamics
        self.constraint = constraint

    def forward(self, prev: QP) -> QP:
        """Forward calculation of the action step.

        Args:
            prev: Quadratic program of the optimal control problem so far.

        Returns:
            Quadratic program of the optimal control problem with the added
            action step.
        """
        nx = self.dynamics.A.shape[1]
        nu = self.dynamics.B.shape[1]
        old_eq = (
            torch.cat(
                [
                    prev.A,
                    torch.zeros((prev.A.shape[0], nu + nx)),
                ],
                dim=1,
            )
            if prev.A.shape[0] > 0
            else prev.A
        )
        new_eq = torch.cat(
            [
                torch.zeros((nx, prev.H.shape[1] - nx)),
                self.dynamics.A,
                self.dynamics.B,
                -torch.eye(nx),
            ],
            dim=1,
        )
        old_ineq = (
            torch.cat(
                [
                    prev.C,
                    torch.zeros((prev.C.shape[0], nu + nx)),
                ],
                dim=1,
            )
            if prev.C.shape[0] > 0
            else prev.C
        )
        nc = self.constraint.size if self.constraint is not None else 0
        new_ineq = (
            torch.cat(
                [
                    torch.zeros((nc, prev.H.shape[1] - nx)),
                    (
                        self.constraint.C
                        if self.constraint.C is not None
                        else torch.zeros((nc, nx))
                    ),
                    (
                        self.constraint.D
                        if self.constraint.D is not None
                        else torch.zeros((nc, nu))
                    ),
                    torch.zeros((nc, nx)),
                ],
                dim=1,
            )
            if self.constraint is not None
            else torch.tensor([])
        )
        new_lower = (
            self.constraint.e
            if self.constraint is not None
            else torch.tensor([])
        )
        new_upper = (
            self.constraint.f
            if self.constraint is not None
            else torch.tensor([])
        )
        H_update = torch.zeros(prev.H.shape)
        H_update[-nx:, -nx:] = self.cost.Q
        g_update = torch.zeros(prev.g.shape)
        g_update[-nx:] = self.cost.q
        return QP(
            H=torch.block_diag(
                prev.H + H_update,
                self.cost.R,
                torch.zeros((nx, nx)),
            ),
            g=torch.cat(
                [
                    prev.g + g_update,
                    self.cost.r,
                    torch.zeros(nx),
                ]
            ),
            A=torch.cat([old_eq, new_eq]),
            b=torch.cat([prev.b, torch.zeros(nx)]),
            C=torch.cat([old_ineq, new_ineq]),
            l=torch.cat([prev.l, new_lower]),
            u=torch.cat([prev.u, new_upper]),
        )
