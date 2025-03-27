#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Transition of the integrator."""

import torch
import torch.nn as nn

from ..types import Dynamics, IntegrationMap


class Step(nn.Module):
    r"""Transition of the integrator.

    A transition from step :math:`k` extends the trajectory vector :math:`X_k
    = (x_0, \ldots, x_k)` with the next state :math:`x_{k+1}` after applying
    transition :class:`Dynamics`:

    .. math::

        x_{k+1} = A x_k + B u_k

    The resulting state vector is :math:`X_{k+1}`.

    Attributes:
        dynamics: Linear dynamics of the action step.
    """

    def __init__(self, dynamics: Dynamics) -> None:
        """Initialize module."""
        super().__init__()
        self.dynamics = dynamics

    def forward(self, prev: IntegrationMap) -> IntegrationMap:
        """Forward calculation of the action step.

        Args:
            prev: Quadratic program of the optimal control problem so far.

        Returns:
            Quadratic program of the optimal control problem with the added
            action step.
        """
        # x_k = psi U_k + phi x_0
        # x_next = A x_k + B u_k = [A psi, B] U_next + (A phi) x_0
        nu = self.dynamics.B.shape[1]
        new_phi = self.dynamics.A @ prev.phi
        new_psi = (
            torch.cat([self.dynamics.A @ prev.psi, self.dynamics.B], dim=1)
            if prev.psi.shape[0] > 0
            else self.dynamics.B
        )
        old_Psi = torch.cat(
            [prev.Psi, torch.zeros((prev.Psi.shape[0], nu))],
            dim=1,
        )
        return IntegrationMap(
            Phi=torch.cat([prev.Phi, new_phi]),
            Psi=torch.cat([old_Psi, new_psi]),
            phi=new_phi,
            psi=new_psi,
        )
