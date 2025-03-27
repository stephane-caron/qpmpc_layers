#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

import pytest
import torch

from qpmpc_layers.exceptions import DataTypeError, DimensionError
from qpmpc_layers.types import Constraint


def test_dimensions():
    with pytest.raises(DimensionError):
        Constraint(C=torch.ones((4, 4)), D=torch.eye(5))


def test_dtype():
    with pytest.raises(DataTypeError):
        Constraint(
            C=torch.eye(4, dtype=torch.float32),
            D=torch.eye(4, dtype=torch.float64),
        )
