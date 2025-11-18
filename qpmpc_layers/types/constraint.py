#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Inequality constraint along the optimal control problem."""

from typing import Optional

import torch

from ..exceptions import DimensionError
from ..utils import check_dtypes


class Constraint:
    r"""Inequality constraint along the optimal control problem.

    Inequalities are expressed as:

    .. math::

        e \leq C x_k + D u_k \leq f

    Attributes:
        C: State constraint matrix in :math:`C x_k`.
        D: Input constraint matrix in :math:`D u_k`.
        e: Optional constraint lower bound :math:`e` (default: no lower bound).
        f: Optional constraint upper bound :math:`f` (default: no upper bound).
    """

    C: Optional[torch.Tensor]
    D: Optional[torch.Tensor]
    e: torch.Tensor
    f: torch.Tensor

    def __init__(
        self,
        C: Optional[torch.Tensor] = None,
        D: Optional[torch.Tensor] = None,
        e: Optional[torch.Tensor] = None,
        f: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize constraint.

        Args:
            C: State constraint matrix in :math:`C x_k`.
            D: Input constraint matrix in :math:`D u_k`.
            e: Optional constraint lower bound :math:`e` (default: no lower
                bound).
            f: Optional constraint upper bound :math:`f` (default: no upper
                bound).
        """
        super().__init__()
        nc = max(
            C.shape[0] if C is not None else 0,
            D.shape[0] if D is not None else 0,
        )
        if C is not None and D is not None and C.shape[0] != D.shape[0]:
            raise DimensionError("C", C.shape, D.shape)
        if e is not None and e.shape[0] != nc:
            raise DimensionError("e", e.shape, (nc,))
        if f is not None and f.shape[0] != nc:
            raise DimensionError("f", f.shape, (nc,))
        self.C = C
        self.D = D
        self.e = e if e is not None else -1e20 * torch.ones(nc)
        self.f = f if f is not None else +1e20 * torch.ones(nc)
        check_dtypes(self, ["C", "D", "e", "f"])

    @property
    def dtype(self) -> torch.dtype:
        """Get the PyTorch datatype of the constraint tensors."""
        check_dtypes(self, ["C", "D", "e", "f"])
        return self.C.dtype

    @property
    def size(self) -> int:
        """Number of inequality constraints."""
        return self.e.shape[0]
