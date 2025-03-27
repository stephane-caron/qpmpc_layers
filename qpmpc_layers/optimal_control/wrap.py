#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Initial state constraint of an optimal control problem."""

import torch
import torch.nn as nn

from ..exceptions import DimensionError
from ..types import QP


class Wrap(nn.Module):
    r"""Set the initial state constraint to complete an OCP."""

    def __init__(self, state_dim: int) -> None:
        """Initialize module.

        Args:
            state_dim: Dimension of the state space.
        """
        super().__init__()
        self.state_dim = state_dim

    def forward(self, prev: QP, x_init: torch.Tensor) -> QP:
        """Forward calculation of the wrapping step.

        Args:
            prev: Quadratic program of the optimal control problem so far.
            x_init: Initial state of the problem.

        Returns:
            Quadratic program of the completed optimal control problem.
        """
        if x_init.ndim != 1 or x_init.shape[0] != self.state_dim:
            raise DimensionError("x_init", x_init.shape, (self.state_dim,))
        nx = self.state_dim
        ntau = prev.A.shape[1]
        A_init = torch.cat(
            [torch.eye(nx), torch.zeros((nx, ntau - nx))],
            dim=1,
        )
        return QP(
            H=prev.H,
            g=prev.g,
            A=torch.cat([prev.A, A_init]),
            b=torch.cat([prev.b, x_init]),
            C=prev.C,
            l=prev.l,
            u=prev.u,
        )
