#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Type for quadratic programs."""

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class QP:
    r"""Quadratic program in ProxQP format.

    .. math::

        \begin{split}\begin{array}{ll}
            \underset{x}{\mbox{minimize}} &
                \frac{1}{2} x^T H x + g^T x \\
            \mbox{subject to}
                & A x = b                   \\
                & l \leq C x \leq u
        \end{array}\end{split}

    Attributes:
        H: Cost matrix.
        g: Cost vector.
        A: Equality-constraint matrix.
        b: Equality-constraint vector.
        C: Inequality-constraint matrix.
        l: Inequality-constraint lower bound.
        u: Inequality-constraint upper bound..
    """

    H: torch.Tensor
    g: torch.Tensor
    A: torch.Tensor
    b: torch.Tensor
    C: torch.Tensor
    l: torch.Tensor  # noqa: E741
    u: torch.Tensor

    def unpack(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Unpack dataclass as a tuple.

        Returns:
            Tuple :math:`(H, g, A, b, C, l, u)` of the quadratic program.
        """
        return (
            self.H,
            self.g,
            self.A,
            self.b,
            self.C,
            self.l,
            self.u,
        )
