""" Test error message string formatters."""

import numpy as np
import torch
from torch_nf.error_formatters import *
from pytest import raises


def test_format_type_err_msg():
    """Test that TypeError formatted strings are correct."""
    x = 20
    s1 = "foo"
    s2 = "bar"
    d = {"x": x, "s1": s1, "s2": s2}
    assert (
        format_type_err_msg(x, s1, s2, int) == "int argument foo must be int not str."
    )
    assert (
        format_type_err_msg(d, s2, x, str) == "dict argument bar must be str not int."
    )
    assert (
        format_type_err_msg(s1, s2, x, dict) == "str argument bar must be dict not int."
    )

    with raises(ValueError):
        format_type_err_msg(d, s1, s2, str)

    with raises(ValueError):
        format_type_err_msg(d, s1, x, int)

    return None

def test_dbg_check():
    M = 20
    N = 50
    D = 4
    y = torch.normal(0., 1., (M,N,D))
    assert not dbg_check(y, 'y')

    y[0,5,2] = np.nan
    assert dbg_check(y, 'y')

    y[3,1,2] = np.inf
    assert dbg_check(y, 'y')

    y[0,5,2] = 0.
    y[3,1,2] = 0.
    assert not dbg_check(y, 'y')
    return None


if __name__ == "__main__":
    test_format_type_err_msg()
    test_dbg_check()
