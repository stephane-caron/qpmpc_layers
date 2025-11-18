#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Cost of a transition in the optimal control problem."""

from typing import Optional

import torch

from ..exceptions import DimensionError
from ..utils import check_dtypes


class StageCost:
    r"""Cost of a transition in the optimal control problem.

    The quadratic cost expression is defined by matrices :math:`(Q, R)` and
    vectors :math:`(q, r)` so that the contribution :math:`\ell(x_k, u_k)`
    to the total cost is:

    .. math::

        \ell(x_k, u_k) = \frac{1}{2} x_k^T Q x_k + q^T x_k +
        \frac{1}{2} u_k^T R u_k + r^T u_k

    Attributes:
        Q: State cost matrix in :math:`x_k^T Q x_k`.
        R: Action cost matrix in :math:`u_k^T R u_k`.
        q: Optional state cost vector in :math:`q^T x_k` (default: zero).
        r: Optional action cost vector in :math:`r^T u_k` (default: zero).
    """

    Q: torch.Tensor
    R: torch.Tensor
    q: torch.Tensor
    r: torch.Tensor

    def __init__(
        self,
        Q: torch.Tensor,
        R: torch.Tensor,
        q: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
    ):
        """Initialize cost.

        Args:
            Q: State cost matrix in :math:`x_k^T Q x_k`.
            R: Action cost matrix in :math:`u_k^T R u_k`.
            q: Optional state cost vector in :math:`q^T x_k` (default:
                zero).
            r: Optional action cost vector in :math:`r^T u_k` (default: zero).
        """
        super().__init__()
        nx = Q.shape[0]
        nu = R.shape[0]
        if not (Q.shape[0] == nx and Q.shape[1] == nx):
            raise DimensionError("Q", Q.shape, (nx, nx))
        if not (R.shape[0] == nu and R.shape[1] == nu):
            raise DimensionError("R", R.shape, (nu, nu))
        if q is not None and q.shape[0] != nx:
            raise DimensionError("q", q.shape, (nx,))
        if r is not None and r.shape[0] != nu:
            raise DimensionError("r", r.shape, (nu,))
        self.Q = Q
        self.R = R
        self.q = q if q is not None else torch.zeros(nx, dtype=Q.dtype)
        self.r = r if r is not None else torch.zeros(nu, dtype=R.dtype)
        check_dtypes(self, ["Q", "R", "q", "r"])

    @property
    def dtype(self) -> torch.dtype:
        """Get the PyTorch datatype of the stage cost tensors."""
        check_dtypes(self, ["Q", "R", "q", "r"])
        return self.Q.dtype
