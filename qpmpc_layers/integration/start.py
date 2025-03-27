#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Initial step of an integration pipeline."""

from typing import Optional

import torch
import torch.nn as nn

from ..types import IntegrationMap


class Start(nn.Module):
    r"""First layer of an integration pipeline.

    Attributes:
        dtype: Desired data type of intermediate tensors.
        state_dim: Dimension of the state space.
    """

    dtype: torch.dtype
    state_dim: int

    def __init__(
        self,
        state_dim: int,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize module.

        Args:
            state_dim: Dimension of the state space.
            dtype: Desired data type of intermediate tensors.
        """
        super().__init__()
        self.dtype = dtype
        self.state_dim = state_dim

    def forward(self) -> IntegrationMap:
        """Get a blank trajectory map.

        Returns:
            Quadratic program for a new OCP.
        """
        return IntegrationMap(
            Psi=torch.zeros((self.state_dim, 0), dtype=self.dtype),
            Phi=torch.eye(self.state_dim, dtype=self.dtype),
            psi=torch.tensor([], dtype=self.dtype),
            phi=torch.eye(self.state_dim, dtype=self.dtype),
        )
