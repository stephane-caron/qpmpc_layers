#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Dynamics of an optimal control problem step."""

import torch

from ..exceptions import DimensionError
from ..utils import check_dtypes


class Dynamics:
    r"""Dynamics of an optimal control problem step.

    The dynamics of step :math:`k` are given by:

    .. math::

        x_{k+1} = A x_k + B u_k

    Attributes:
        A: State transition matrix in :math:`A x_k`.
        B: Action transition matrix in :math:`B u_k`.
    """

    A: torch.Tensor
    B: torch.Tensor

    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> None:
        """Initialize dynamics.

        Args:
            A: State transition matrix in :math:`A x_k`.
            B: Action transition matrix in :math:`B u_k`.
        """
        nx = A.shape[1]
        nu = B.shape[1]
        if not (A.shape[0] == nx and A.shape[1] == nx):
            raise DimensionError("A", A.shape, (nx, nx))
        if not (B.shape[0] == nx and B.shape[1] == nu):
            raise DimensionError("B", B.shape, (nx, nu))
        self.A = A
        self.B = B
        check_dtypes(self, ["A", "B"])

    @property
    def dtype(self) -> torch.dtype:
        """Get the PyTorch datatype of the dynamics step tensors."""
        check_dtypes(self, ["A", "B"])
        return self.A.dtype
