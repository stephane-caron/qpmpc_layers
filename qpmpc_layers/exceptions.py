#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Exceptions raised by the library."""

from typing import Sequence


class QPMPCLayersException(Exception):
    """Base class for exceptions from this library."""


class DimensionError(QPMPCLayersException):
    """Failed dimension check."""

    def __init__(
        self,
        what: str,
        shape: Sequence[int],
        expected_shape: Sequence[int],
    ):
        """Initialize exception.

        Args:
            what: Name of the matrix or vector whose check failed.
            shape: Evaluated shape.
            expected_shape: Expected shape.
        """
        super().__init__(
            f"Shape {shape} of {what} "
            f"does not match the expected {expected_shape}"
        )


class DataTypeError(QPMPCLayersException):
    """Unable to infer an intermediate tensor data type."""


class ShapeMismatch(QPMPCLayersException):
    """Unable to infer an intermediate tensor data type."""

    def __init__(
        self,
        what_1: str,
        shape_1: Sequence[int],
        what_2: str,
        shape_2: Sequence[int],
    ):
        """Initialize exception.

        Args:
            what_1: Name of the first matrix.
            shape_1: Shape of the first matrix.
            what_2: Name of the second matrix.
            shape_2: Shape of the second matrix.
        """
        super().__init__(
            f"Shape {shape_1} of {what_1} "
            f"is not compatible with the shape {shape_2} of {what_2}"
        )
