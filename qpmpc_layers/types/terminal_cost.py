#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Cost on the last state of the optimal control problem."""

from typing import Optional

import torch

from ..exceptions import DimensionError
from ..utils import check_dtypes


class TerminalCost:
    r"""Cost on the last state of the optimal control problem.

    The quadratic cost expression is defined by matrix :math:`Q` and
    vector :math:`q` so that the cost :math:`\ell(x_k)` is:

    .. math::

        \ell(x_k) = \frac{1}{2} x_k^T Q x_k + q^T x_k

    where :math:`k` is the index of the last step in the optimal control
    problem.

    Attributes:
        Q: State cost matrix in :math:`x_k^T Q x_k`.
        q: Optional state cost vector in :math:`q^T x_k` (default: zero).
    """

    Q: torch.Tensor
    q: torch.Tensor

    def __init__(
        self,
        Q: torch.Tensor,
        q: Optional[torch.Tensor] = None,
    ):
        """Initialize cost.

        Args:
            Q: State cost matrix in :math:`x_k^T Q x_k`.
            q: Optional state cost vector in :math:`q^T x_k` (default: zero).
        """
        super().__init__()
        nx = Q.shape[0]
        if not (Q.shape[0] == nx and Q.shape[1] == nx):
            raise DimensionError("Q", Q.shape, (nx, nx))
        if q is not None and q.shape[0] != nx:
            raise DimensionError("q", q.shape, (nx,))
        self.Q = Q
        self.q = q if q is not None else torch.zeros(nx, dtype=Q.dtype)
        check_dtypes(self, ["Q", "q"])

    @property
    def dtype(self) -> torch.dtype:
        """Get the PyTorch datatype of the terminal cost tensors."""
        check_dtypes(self, ["Q", "q"])
        return self.Q.dtype
