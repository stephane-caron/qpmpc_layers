#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Utility functions."""

from typing import List

from .exceptions import DataTypeError


def check_dtypes(obj: object, attributes: List[str]):
    """Check data type consistency of tensors in an object.

    Args:
        obj: Object instance containing tensors.
        attributes: List of tensor attribute names to check.

    Raises:
        DataTypeError:
    """
    dtypes = {
        attr: obj.__dict__[attr].dtype
        for attr in attributes
        if obj.__dict__[attr] is not None
    }
    otype = type(obj).__name__
    if len(set(dtypes.values())) > 1:
        raise DataTypeError(
            f"Inconsistent data types in {otype} object: {dtypes}"
        )
