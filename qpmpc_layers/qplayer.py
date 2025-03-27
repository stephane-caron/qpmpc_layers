#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Shallow wrapper around QPLayer from ProxQP."""

import torch
import torch.nn as nn
from proxsuite.torch.qplayer import QPFunction

from .types import QP


class QPLayer(nn.Module):
    """Shallow wrapper around QPLayer from ProxQP."""

    def __init__(self, max_iter: int = 1000):
        """Initialize QP layer.

        Args:
            max_iter: Maximum number of iterations the QP solver will perform.
        """
        super().__init__()
        self._qp_function = QPFunction(maxIter=max_iter)

    def forward(self, qp: QP) -> torch.Tensor:
        """Solve QP and return its primal solution.

        Args:
            qp: Tensors of the quadratic program to solve.

        Returns:
            Primal solution to the quadratic program.
        """
        x, _, _ = self._qp_function(*qp.unpack())
        return x.float()
