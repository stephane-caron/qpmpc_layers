#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

import pytest
import torch

from qpmpc_layers.exceptions import DataTypeError, DimensionError
from qpmpc_layers.types import Dynamics


def test_dimensions():
    with pytest.raises(DimensionError):
        Dynamics(A=torch.ones((3, 4)), B=torch.eye(4))


def test_dtype():
    with pytest.raises(DataTypeError):
        Dynamics(
            A=torch.eye(4, dtype=torch.float32),
            B=torch.eye(4, dtype=torch.float64),
        )
