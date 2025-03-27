#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

import pytest
import torch

import qpmpc_layers.lti as lti
from qpmpc_layers import StageCost
from qpmpc_layers.exceptions import ShapeMismatch
from qpmpc_layers.plants import LinearizedWheeledInvertedPendulum


def test_shapes():
    with pytest.raises(ShapeMismatch):
        lti.OptimalControlProblem(
            nb_steps=12,
            stage_cost=StageCost(
                Q=torch.eye(4),
                R=torch.eye(3),
            ),
            dynamics=LinearizedWheeledInvertedPendulum(length=0.1, T=0.1),
        )
