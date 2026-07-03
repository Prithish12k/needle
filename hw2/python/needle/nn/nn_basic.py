"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype).data, device=device, dtype=dtype)
        if bias:
                self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).transpose().data, device=device, dtype=dtype)        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        res = X @ self.weight
        if self.bias is not None:
            res = res + self.bias.broadcast_to((X.shape[0], self.out_features))
        return res
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        res = x
        for module in self.modules:
            res = module.forward(res)
        return res
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y_oh = init.one_hot(logits.shape[1], y)
        sft = ops.logsoftmax(logits)
        return -(y_oh*sft).sum() / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.constant(dim, device=device, dtype=dtype).data, device=device, dtype=dtype)
        self.bias = Parameter(init.constant(dim, c=0.0, device=device, dtype=dtype).data, device=device, dtype=dtype)
        self.running_mean = init.constant(dim, c=0.0, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.constant(dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        w = self.weight.reshape((1, x.shape[1])).broadcast_to(x.shape)
        b = self.bias.reshape((1, x.shape[1])).broadcast_to(x.shape)

        if self.training is False:
            norm = (x - self.running_mean.reshape((1, x.shape[1])).broadcast_to(x.shape)) / ((self.running_var.reshape((1, x.shape[1])).broadcast_to(x.shape) + self.eps)**0.5)
            return (w * norm) + b

        mu = x.sum(axes=0) / x.shape[0]
        self.running_mean = ((self.running_mean * (1 - self.momentum)) + (self.momentum * mu)).data
        mu = mu.reshape((1, x.shape[1])).broadcast_to(x.shape)

        sig = ((x - mu)**2).sum(axes=0) / x.shape[0]
        self.running_var = ((self.running_var * (1 - self.momentum)) + (self.momentum * sig)).data
        sig = sig.reshape((1, x.shape[1])).broadcast_to(x.shape)

        norm = (x - mu) / ((sig + self.eps)**0.5)
        return (w * norm) + b
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.constant(dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.bias = Parameter(init.constant(dim, c=0.0, device=device, dtype=dtype), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mu = x.sum(axes=1) / self.dim
        mu = mu.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        sig = ((x - mu)**2).sum(axes=1) / self.dim
        sig = (sig.reshape((x.shape[0], 1)).broadcast_to(x.shape) + self.eps)**0.5
        norm = (x - mu)/sig
        return (self.weight.reshape((1, self.dim)).broadcast_to(x.shape) * norm) + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training == False:
            return x

        mask = init.randb(*x.shape, p=1.0 - self.p, device=x.device, dtype="float32")

        return (mask * x) / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
