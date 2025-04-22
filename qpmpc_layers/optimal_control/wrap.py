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

    def __init__(self, state_dim: int, reg : int = 1e-8 , sr: int = 1e-3) -> None:
        """Initialize module.

        Args:
            state_dim: Dimension of the state space.
        """
        super().__init__()
        self.state_dim = state_dim
        self.reg = reg
        self.sr = sr
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
        A = torch.cat([prev.A, A_init])
        b = torch.cat([prev.b, x_init])
        if self.sr is not None and self.reg is not None :
            return self.add_slack_variable(prev.H, prev.g, A, b, prev.C, prev.l, prev.u)
        else :
            return QP(
                H=prev.H,
                g=prev.g,
                A=A,
                b=b,
                C=prev.C,
                l=prev.l,
                u=prev.u,
            )
    def add_slack_variable(self, H,g,A,b,C,l,u):
        n, m = H.shape[0], C.shape[0]
        H2 = torch.block_diag(*[H ,+self.reg*torch.eye(2*m) ])
        g2 = torch.cat([g, self.sr*torch.ones(m), -self.sr*torch.ones(m)])
        A2 = torch.block_diag(*[A, torch.zeros(2*m,2*m)])
        b2 = torch.cat([b, torch.zeros(2*m)])
        C2 = torch.cat([C,C,torch.zeros(2*C.shape[0], C.shape[1])],0)
        E = torch.cat([-torch.eye(2*m), torch.eye(2*m)])
        C2 = torch.cat([C2,E],1)
        l2 = torch.cat([torch.zeros(m), l , -1e8 * torch.ones(m) , torch.zeros(m)])
        u2 = torch.cat([u, torch.zeros(2*m), 1e8*torch.ones(m)])
        print(g2.shape, A2.shape,b2.shape, H2.shape, C2.shape, l2.shape ,u2.shape)
        return QP(H=H2, g= g2 , A = A2 , b= b2, C=C2, l = l2, u=u2)