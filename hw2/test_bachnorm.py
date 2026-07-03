import sys
sys.path.append("./python")

import pytest
import needle as nd
import numpy as np

def test_empty_input():
    # Empty input (zero-dimensional)
    x = nd.init.zeros(0, device='cpu', dtype='float32')
    bn = nd.nn.BatchNorm1d(2)
    with pytest.raises(ValueError):
        bn(x)

def test_single_element():
    # Minimal input (single element)
    x = nd.init.ones((1, 2), device='cpu', dtype='float32')
    bn = nd.nn.BatchNorm1d(2)
    bn.train = False
    result = bn(x)
    assert result.shape == (1, 2)

def test_boundary_condition():
    # Large input causing overflow
    x = nd.init.ones((10**6, 2), device='cpu', dtype='float32') * 1e20
    bn = nd.nn.BatchNorm1d(2)
    bn.train = False
    result = bn(x)
    assert not np.isnan(result).any()

def test_invalid_input():
    # Input with wrong dimensions
    x = nd.init.ones((2, 2), device='cpu', dtype='float32')
    bn = nd.nn.BatchNorm1d(2)
    with pytest.raises(ValueError):
        bn(x)

def test_division_by_zero():
    # Division by zero in variance calculation
    x = nd.init.ones((2, 2), device='cpu', dtype='float32')
    bn = nd.nn.BatchNorm1d(2, eps=0.0)
    bn.train = True
    with pytest.raises(ZeroDivisionError):
        bn(x)
