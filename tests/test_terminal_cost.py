#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

import pytest
import torch

from qpmpc_layers.exceptions import DataTypeError, DimensionError
from qpmpc_layers.types import TerminalCost


def test_completion():
    cost = TerminalCost(Q=torch.eye(3))
    assert isinstance(cost.q, torch.Tensor)


def test_dimensions():
    with pytest.raises(DimensionError):
        TerminalCost(Q=torch.ones((3, 4)))
    with pytest.raises(DimensionError):
        TerminalCost(Q=torch.eye(3), q=torch.ones(5))


def test_dtype():
    with pytest.raises(DataTypeError):
        TerminalCost(
            Q=torch.eye(4, dtype=torch.float32),
            q=torch.ones((4,), dtype=torch.float64),
        )
