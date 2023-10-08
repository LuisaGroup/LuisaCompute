from .types import BuiltinFuncBuilder, basic_dtypes, to_lctype
from .dylibs import lcapi


def autodiff():
    pass


@BuiltinFuncBuilder
def requires_grad(*x):
    for xi in x:
        lcapi.builder().call(lcapi.CallOp.REQUIRES_GRADIENT, [xi.expr])
    return None, None


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
def backward(*x):
    if len(x) == 1:
        x_grad = lcapi.builder().call(to_lctype(x[0].dtype), lcapi.CallOp.ONE, [])
    else:
        assert len(x) == 2
        x_grad = x[1].expr
    x0 = x[0].expr
    lcapi.builder().call(lcapi.CallOp.GRADIENT_MARKER, [x0, x_grad])
    return None, lcapi.builder().call(lcapi.CallOp.BACKWARD, [x0])


@BuiltinFuncBuilder
def detach(x):
    return x.dtype, lcapi.builder().call(lcapi.CallOp.DETACH, [x.expr])
