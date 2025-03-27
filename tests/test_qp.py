#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

import torch

from qpmpc_layers.types import QP


def test_unpack():
    H, g, A, b, C, l, u = (
        torch.eye(3),
        torch.zeros(3),
        torch.eye(3),
        torch.ones(3),
        torch.eye(3),
        torch.ones(3),
        torch.ones(3),
    )
    qp = QP(H, g, A, b, C, l, u)
    assert (H, g, A, b, C, l, u) == qp.unpack()
