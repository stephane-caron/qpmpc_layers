#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

import casadi as cs

x = cs.MX.sym("x")
A = cs.vertcat(
    cs.horzcat(x, x**2, x**3),
    cs.horzcat(x + 1, x**2 + 1, x**3 + 1),
)
dA_dx = cs.jacobian(A, x)

A_func = cs.Function("A_func", [x], [A, A @ A.T])
dA_dx_func = cs.Function("dA_dx_func", [x], [dA_dx])

x_value = 2.0
A_evaluated = A_func(x_value)
dA_dx_evaluated = dA_dx_func(x_value)
print(f"A(x = {x_value}) = {A_evaluated}")
print(f"dA_dx(x = {x_value}) = {dA_dx_evaluated}")
