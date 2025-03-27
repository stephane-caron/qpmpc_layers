#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Integrator for a linear time-invariant system."""

import torch
import torch.nn as nn

from ..integration import Start, Step
from ..types import Dynamics, IntegrationMap


class Integrator(nn.Module):
    r"""Integrator for a linear time-invariant system.

    An integrator applies an integration scheme to system dynamics. In this
    instance: explicit Euler to linear time-invariant dynamics.

    Attributes:
        dynamics: Dynamics of the system.
    """

    _integration_map: IntegrationMap
    dynamics: Dynamics

    def __init__(self, nb_steps: int, dynamics: Dynamics) -> None:
        """Initialize module."""
        super().__init__()
        state_dim = dynamics.A.shape[1]
        start = Start(state_dim, dtype=dynamics.A.dtype)
        step = Step(dynamics)
        integration_map: IntegrationMap = start()
        for _ in range(nb_steps):
            integration_map = step(integration_map)
        self._integration_map = integration_map
        self.dynamics = dynamics

    def forward(self, U: torch.Tensor, x_init: torch.Tensor) -> torch.Tensor:
        r"""Forward computation performed by the module.

        Args:
            U: Input trajectory :math:`U_N = (u_0, \ldots, u_{N-1})`.
            x_init: Initial state :math:`x_\mathit{init}`.

        Returns:
            Flat state trajectory vector :math:`X_N = [x_0 \ldots x_N]`
            resulting from integrating the input trajectory from the initial
            state.
        """
        Psi = self._integration_map.Psi
        Phi = self._integration_map.Phi
        Psi_U = (Psi @ U).flatten()
        Phi_x_init = (Phi @ x_init).flatten()
        X = Psi_U + Phi_x_init
        return X
