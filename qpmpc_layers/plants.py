#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Miscellanous functions."""

import torch
import torch.nn as nn

from .types import Dynamics


class DoubleIntegrator(nn.Module):
    """Double integrator dynamics with timestep as a parameter."""

    def __init__(self, T: float):
        """Initialize integrator from its timestep.

        Args:
            T: Integrator timestep in seconds.
        """
        super().__init__()
        self.T = nn.Parameter(torch.tensor([T]))

    def forward(self) -> Dynamics:
        """Compute the dynamics of the double integrator.

        Returns:
            Dynamics of the double integrator.
        """
        _1 = torch.ones(1, dtype=self.T.dtype)
        _0 = torch.zeros(1, dtype=self.T.dtype)
        _T = self.T
        return Dynamics(
            A=torch.tensor(
                [
                    [_1, _T],
                    [_0, _1],
                ],
            ),
            B=torch.tensor(
                [
                    [_T**2 / 2.0],
                    [_T],
                ],
            ),
        )


class TripleIntegrator(nn.Module):
    """Triple integrator dynamics with timestep as a parameter."""

    def __init__(self, T: float):
        """Initialize integrator from its timestep.

        Args:
            T: Integrator timestep in seconds.
        """
        super().__init__()
        self.T = nn.Parameter(torch.tensor([T]))

    def forward(self) -> Dynamics:
        """Compute the dynamics of the triple integrator.

        Returns:
            Dynamics of the triple integrator.
        """
        _1 = torch.ones(1, dtype=self.T.dtype)
        _0 = torch.zeros(1, dtype=self.T.dtype)
        _T = self.T
        return Dynamics(
            A=torch.stack(
                [
                    torch.cat([_1, _T, _T**2 / 2.0]),
                    torch.cat([_0, _1, _T]),
                    torch.cat([_0, _0, _1]),
                ],
            ),
            B=torch.stack(
                [
                    _T**3 / 6.0,
                    _T**2 / 2.0,
                    _T,
                ],
            ),
        )


class LinearizedWheeledInvertedPendulum(nn.Module):
    """Wheeled inverted pendulum linearized around a vertical state.

    State vector:

    - ground position [m]
    - pitch angle [rad]
    - ground velocity [m]
    - pitch angular velocity [rad/s]
    """

    def __init__(self, length: float, T: float, gravity: float = 9.81):
        """Define wheeled inverted pendulum model from its parameters.

        Args:
            length: Length of the pendulum.
            T: Integration timestep in seconds.
            gravity: Gravity constant in m/s².
        """
        super().__init__()
        self.gravity = gravity
        self.length = nn.Parameter(torch.tensor([length]))
        self.T = nn.Parameter(torch.tensor([T]))

    def forward(self) -> Dynamics:
        """Compute the dynamics of the linearized wheeled inverted pendulum.

        Returns:
            Dynamics of the linearized wheeled inverted pendulum.
        """
        _1 = torch.ones(1, dtype=self.T.dtype)
        _0 = torch.zeros(1, dtype=self.T.dtype)
        _T = self.T
        omega = torch.sqrt(self.gravity / self.length)
        _cosh = torch.cosh(self.T * omega)
        _sinh = torch.sinh(self.T * omega)
        return Dynamics(
            A=torch.stack(
                [
                    torch.cat([_1, _0, _T, _0]),
                    torch.cat([_0, _cosh, _0, _sinh / omega]),
                    torch.cat([_0, _0, _1, _0]),
                    torch.cat([_0, omega * _sinh, _0, _cosh]),
                ]
            ),
            B=torch.stack(
                [
                    _T**2 / 2.0,
                    (_1 - _cosh) / self.gravity,
                    _T,
                    -omega * _sinh / self.gravity,
                ]
            ),
        )
