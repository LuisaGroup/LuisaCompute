from .types import BuiltinFuncBuilder, basic_dtypes, to_lctype
from .dylibs import lcapi


def autodiff():
    pass


@BuiltinFuncBuilder
def requires_grad(x):
    return None, lcapi.builder().call(lcapi.CallOp.REQUIRES_GRADIENT, [x.expr])


@BuiltinFuncBuilder
def grad(x):
    return x.dtype, lcapi.builder().call(to_lctype(x.dtype), lcapi.CallOp.GRADIENT, [x.expr])

@BuiltinFuncBuilder
def one(dtype):
    return dtype, lcapi.builder().call(to_lctype(dtype), lcapi.CallOp.ONE, [])

@BuiltinFuncBuilder
def zero(dtype):
    return dtype, lcapi.builder().call(to_lctype(dtype), lcapi.CallOp.ZERO, [])


@BuiltinFuncBuilder
def backward(x):
    x_grad = lcapi.builder().call(to_lctype(x.dtype), lcapi.CallOp.ONE, [])
    lcapi.builder().call(lcapi.CallOp.GRADIENT_MARKER, [x.expr, x_grad])
    return None, lcapi.builder().call(lcapi.CallOp.BACKWARD, [])


@BuiltinFuncBuilder
def detach(x):
    return x.dtype, lcapi.builder().call(lcapi.CallOp.DETACH, [x.expr])
