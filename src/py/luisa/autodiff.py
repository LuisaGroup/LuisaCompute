from .types import BuiltinFuncBuilder, basic_dtypes, to_lctype
from .dylibs import lcapi


def autodiff():
    pass


@BuiltinFuncBuilder
def requires_grad(x):
    if not x.dtype in basic_dtypes:
        raise SyntaxError("auto-diff only allow basic types.")
    return None, lcapi.builder().call(lcapi.CallOp.REQUIRES_GRADIENT, [x.expr])


@BuiltinFuncBuilder
def grad(x):
    if not x.dtype in basic_dtypes:
        raise SyntaxError("auto-diff only allow basic types.")
    return x.dtype, lcapi.builder().call(to_lctype(x.dtype), lcapi.CallOp.GRADIENT, [x.expr])


@BuiltinFuncBuilder
def backward(x):
    if not x.dtype in basic_dtypes:
        raise SyntaxError("auto-diff only allow basic types.")
    grad = lcapi.builder().call(to_lctype(x.dtype),
                                lcapi.CallOp.GRADIENT, [x.expr])
    lcapi.builder().call(lcapi.CallOp.GRADIENT_MARKER, [x.expr, grad])


@BuiltinFuncBuilder
def detach(x):
    if not x.dtype in basic_dtypes:
        raise SyntaxError("auto-diff only allow basic types.")
    return x.dtype, lcapi.builder().call(lcapi.CallOp.DETACH, [x.expr])
