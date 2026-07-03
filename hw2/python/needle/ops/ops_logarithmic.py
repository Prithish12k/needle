from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        S = Z - Z.max(axis=1, keepdims=True)
        return S - array_api.log(array_api.exp(S).sum(axis=1, keepdims=True))
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        S = exp(node)
        og_sum = out_grad.sum(axes=1).reshape((out_grad.shape[0], 1))
        return out_grad - og_sum*S
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return Z.max(axis=self.axes) + array_api.log(array_api.exp(Z - Z.max(axis=self.axes, keepdims=True)).sum(axis=self.axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        if self.axes is None:
            return out_grad.reshape([1]*len(a.shape)) * exp(a - node.reshape([1]*len(a.shape)))
        shape = list(a.shape)
        for x in self.axes:
            shape[x] = 1
        return out_grad.reshape(shape) * exp(a - node.reshape(shape))
        ### END YOUR SOLUTION

def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
