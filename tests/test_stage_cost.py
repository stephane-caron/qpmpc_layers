#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

import pytest
import torch

from qpmpc_layers.exceptions import DataTypeError, DimensionError
from qpmpc_layers.types import StageCost


def test_completion():
    cost = StageCost(Q=torch.eye(3), R=torch.eye(5))
    assert isinstance(cost.q, torch.Tensor)
    assert isinstance(cost.r, torch.Tensor)


def test_dimensions():
    with pytest.raises(DimensionError):
        StageCost(Q=torch.ones((3, 4)), R=torch.eye(5))
    with pytest.raises(DimensionError):
        StageCost(Q=torch.eye(3), R=torch.ones((5, 4)))
    with pytest.raises(DimensionError):
        StageCost(Q=torch.eye(3), R=torch.eye(5), q=torch.ones(5))
    with pytest.raises(DimensionError):
        StageCost(Q=torch.eye(3), R=torch.eye(5), r=torch.ones(3))


def test_dtype():
    with pytest.raises(DataTypeError):
        StageCost(
            Q=torch.eye(4, dtype=torch.float32),
            R=torch.eye(4, dtype=torch.float64),
        )
