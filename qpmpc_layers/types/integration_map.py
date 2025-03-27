#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Type for integration maps."""

from dataclasses import dataclass

import torch


@dataclass
class IntegrationMap:
    r"""Linear map from (initial state, inputs) to future state trajectory.

    Given the initial state :math:`x_0` and the stacked vector of control
    inputs :math:`U_k = (u_0, u_1, \ldots, u_{k-1})`, the state map provides
    the stacked vector :math:`X_k = (x_0, \ldots, x_k)` of future states as:

    .. math::

        X_k = \Psi U_k + \Phi x_0

    This datastructure also maintains :math:`\psi` and :math:`\phi` mapping to
    the last state in the trajectory:

    .. math::

        x_k = \psi U_k + \phi x_0

    Attributes:
        Phi: Linear map from initial state to future states.
        Psi: Linear map from stacked inputs to future states.
        phi: Linear map from initial state to last state.
        psi: Linear map from stacked inputs to last state.
    """

    Phi: torch.Tensor
    Psi: torch.Tensor
    phi: torch.Tensor
    psi: torch.Tensor
