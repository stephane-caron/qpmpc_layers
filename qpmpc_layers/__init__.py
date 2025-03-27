#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""QP-based model predictive control layers in PyTorch."""

from .qplayer import QPLayer
from .types import QP, Constraint, Dynamics, TerminalCost, StageCost

__version__ = "4.0.0"

__all__ = [
    "Constraint",
    "Dynamics",
    "QP",
    "QPLayer",
    "StageCost",
    "TerminalCost",
]
