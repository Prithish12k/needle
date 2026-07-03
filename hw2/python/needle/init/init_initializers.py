import math
from .init_basic import *
from typing import Any
from math import sqrt

def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    a = gain * sqrt(6/(fan_in+fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    raise NotImplementedError()
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    std = gain * sqrt(2/(fan_in+fan_out))
    return randn(fan_in, fan_out, std=std, **kwargs)
    raise NotImplementedError()
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    bound = sqrt(2) * sqrt(3/fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    raise NotImplementedError()
    ### END YOUR SOLUTION



def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    std = sqrt(2/fan_in)
    return randn(fan_in, fan_out, std=std, **kwargs)
    raise NotImplementedError()
    ### END YOUR SOLUTION
