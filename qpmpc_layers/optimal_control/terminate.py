#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Terminal cost of an optimal control problem."""

import torch
import torch.nn as nn

from ..types import QP, TerminalCost


class Terminate(nn.Module):
    r"""Set terminal cost of an optimal control problem (OCP) graph.

    The terminal step adds a :class:`qpmpc_layers.types.TerminalCost`
    :math:`\ell(x_N)` based on the terminal state :math:`x_N`:

    .. math::

        \ell(x_N) = \frac{1}{2} x_N^T Q x_N + q^T x_N

    Note:
        Don't append :class:`Step` layers after this one.

    Attributes:
        cost: Cost on the terminal state.
    """

    cost: TerminalCost

    def __init__(self, cost: TerminalCost) -> None:
        """Initialize layer.

        Args:
            cost: Cost on the terminal state.
        """
        super().__init__()
        self.cost = cost

    def forward(self, prev: QP) -> QP:
        """Forward calculation of the terminal step.

        Args:
            prev: Quadratic program of the optimal control problem so far.

        Returns:
            Quadratic program of the terminated optimal control problem.
        """
        nx = self.cost.Q.shape[1]
        terminal_H = torch.zeros(prev.H.shape)
        terminal_H[-nx:, -nx:] = self.cost.Q
        terminal_g = torch.zeros(prev.g.shape)
        terminal_g[-nx:] = self.cost.q
        return QP(
            H=prev.H + terminal_H,
            g=prev.g + terminal_g,
            A=prev.A,
            b=prev.b,
            C=prev.C,
            l=prev.l,
            u=prev.u,
        )
