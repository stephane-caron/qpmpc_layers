#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Initial step of an optimal control problem."""

from typing import Optional

import torch
import torch.nn as nn

from ..types import QP


class Start(nn.Module):
    r"""First layer of an optimal control problem (OCP) graph.

    Attributes:
        dtype: Desired data type of intermediate tensors.
        state_dim: Dimension of the state space.
    """

    dtype: torch.tensor
    state_dim: int

    def __init__(
        self, state_dim: int, dtype: Optional[torch.dtype] = None
    ) -> None:
        """Initialize module.

        Args:
            state_dim: Dimension of the state space.
            dtype: Desired data type of intermediate tensors.
        """
        super().__init__()
        self.dtype = dtype
        self.state_dim = state_dim

    def forward(self) -> QP:
        """Get a blank OCP quadratic program.

        Returns:
            Quadratic program for a new OCP.
        """
        return QP(
            H=torch.eye(self.state_dim, dtype=self.dtype),
            g=torch.zeros(self.state_dim, dtype=self.dtype),
            A=torch.tensor([], dtype=self.dtype),
            b=torch.tensor([], dtype=self.dtype),
            C=torch.tensor([], dtype=self.dtype),
            l=torch.tensor([], dtype=self.dtype),
            u=torch.tensor([], dtype=self.dtype),
        )
