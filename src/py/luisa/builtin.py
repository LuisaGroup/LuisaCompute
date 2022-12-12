from . import lcapi
from .mathtypes import *
from .types import uint, to_lctype, coerce_map, dtype_of, BuiltinFuncBuilder, \
    scalar_dtypes, arithmetic_dtypes, vector_dtypes, matrix_dtypes, vector, length_of, element_of, nameof
import functools
from . import globalvars
from types import SimpleNamespace
import ast
from .array import ArrayType
from .struct import StructType
from .builtin_type_check import binary_type_infer
from .checkers import binary, all_arithmetic, broadcast, multi_param, length_eq, unary, all_float, all_integer, \
    all_bool, same_length, with_dim, ternary, with_inner_type, is_array, inner_type

def wrap_with_tmp_var(node):
    tmp = lcapi.builder().local(to_lctype(node.dtype))
    lcapi.builder().assign(tmp, node.expr)
    node.expr = tmp
    node.lr = 'l'


def upper_scalar_dtype(dtype0, dtype1):
    if float in [dtype0, dtype1]:
        return float
    elif int in [dtype0, dtype1]:
        return int
    else:
        return uint


def deduce_broadcast(dtype0, dtype1):
    if dtype0 in [float, int, uint, bool]:
        return dtype1  # Broadcast
    elif dtype1 in [float, int, uint, bool]:
        return dtype0  # Broadcast
    else:
        return dtype1  # same size || Matrix * Vector -> Vector


def to_bool(dtype):
    assert dtype in scalar_dtypes or dtype in vector_dtypes
    return vector(bool, length_of(dtype))


def to_float(dtype):
    if dtype in matrix_dtypes:
        return dtype
    assert dtype in scalar_dtypes or dtype in vector_dtypes
    return vector(float, length_of(dtype))


def to_int(dtype):
    assert dtype in scalar_dtypes or dtype in vector_dtypes
    return vector(int, length_of(dtype))


def to_uint(dtype):
    assert dtype in scalar_dtypes or dtype in vector_dtypes
    return vector(uint, length_of(dtype))


def builtin_unary_op(op, operand):
    lc_op = {
        ast.UAdd: lcapi.UnaryOp.PLUS,
        ast.USub: lcapi.UnaryOp.MINUS,
        ast.Not: lcapi.UnaryOp.NOT,
        ast.Invert: lcapi.UnaryOp.BIT_NOT
    }.get(op)
    dtype = operand.dtype
    length = length_of(operand.dtype)
    return dtype, lcapi.builder().unary(to_lctype(dtype), lc_op, operand.expr)


def builtin_bin_op(op, lhs, rhs):
    lc_op = {
        ast.Add: lcapi.BinaryOp.ADD,
        ast.Sub: lcapi.BinaryOp.SUB,
        ast.Mult: lcapi.BinaryOp.MUL,
        ast.Div: lcapi.BinaryOp.DIV,
        ast.FloorDiv: lcapi.BinaryOp.DIV,
        ast.Mod: lcapi.BinaryOp.MOD,
        ast.LShift: lcapi.BinaryOp.SHL,
        ast.RShift: lcapi.BinaryOp.SHR,
        ast.Pow: lcapi.CallOp.POW,
        ast.And: lcapi.BinaryOp.AND,
        ast.Or: lcapi.BinaryOp.OR,
        ast.BitOr: lcapi.BinaryOp.BIT_OR,
        ast.BitXor: lcapi.BinaryOp.BIT_XOR,
        ast.BitAnd: lcapi.BinaryOp.BIT_AND,
        ast.Eq: lcapi.BinaryOp.EQUAL,
        ast.NotEq: lcapi.BinaryOp.NOT_EQUAL,
        ast.Lt: lcapi.BinaryOp.LESS,
        ast.LtE: lcapi.BinaryOp.LESS_EQUAL,
        ast.Gt: lcapi.BinaryOp.GREATER,
        ast.GtE: lcapi.BinaryOp.GREATER_EQUAL
    }.get(op)
    if lc_op is None:
        raise TypeError(f'Unsupported binary operation: {op}')
    return_dtype = binary_type_infer(lhs.dtype, rhs.dtype, op)
    # power operation: a**b
    if op is ast.Pow:
        if type(rhs).__name__ == "Constant":
            exponential = rhs.value
            if type(exponential) is int:
                if exponential == 2:
                    return builtin_bin_op(ast.Mult, lhs, lhs)
                elif exponential == 3:
                    return builtin_bin_op(ast.Mult, lhs, builtin_bin_op(ast.Mult, lhs, lhs))
                elif exponential == 4:
                    return builtin_bin_op(ast.Mult, builtin_bin_op(ast.Mult, lhs, lhs),
                                          builtin_bin_op(ast.Mult, lhs, lhs))
        return builtin_func("pow", lhs, rhs)
    elif op is ast.Div:
        _, lhs_expr = builtin_type_cast(to_float(lhs.dtype), lhs)
        _, rhs_expr = builtin_type_cast(to_float(rhs.dtype), rhs)
        return return_dtype, lcapi.builder().binary(to_lctype(return_dtype), lc_op, lhs_expr, rhs_expr)
    else:
        return return_dtype, lcapi.builder().binary(to_lctype(return_dtype), lc_op, lhs.expr, rhs.expr)


builtin_func_names = {
    'set_block_size',
    'synchronize_block',
    'thread_id', 'block_id', 'dispatch_id', 'dispatch_size',
    'make_uint2', 'make_int2', 'make_float2', 'make_bool2',
    'make_uint3', 'make_int3', 'make_float3', 'make_bool3',
    'make_uint4', 'make_int4', 'make_float4', 'make_bool4',
    'make_float2x2', 'make_float3x3', 'make_float4x4',
    'isinf', 'isnan', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cos', 'cosh',
    'sin', 'sinh', 'tan', 'tanh', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10',
    'sqrt', 'rsqrt', 'ceil', 'floor', 'fract', 'trunc', 'round', 'abs', 'pow',
    'dot', 'cross',
    'length', 'length_squared', 'normalize',
    'copysign', 'fma',
    'min', 'max',
    'all', 'any',
    'select', 'clamp', 'step', 'lerp',
    'clz', 'ctz', 'popcount', 'reverse',
    'determinant', 'transpose', 'inverse',
    'array', 'struct',
    'make_ray', 'inf_ray', 'offset_ray_origin',
    'print',
    'len'
}


# each argument can be a dtype or list/tuple of dtype
def dtype_checked(*dtypes):
    signature = ', '.join([nameof(x) for x in dtypes])
    def wrapper(func_builder):
        @functools.wraps(func_builder)
        def decorated(*args):
            if len(args) != len(dtypes):
                raise TypeError(f"{nameof(func_builder)} takes exactly {len(dtypes)} arguments ({signature}), {len(args)} given.")
            for i in range(len(dtypes)):
                if args[i].dtype not in dtypes[i] if hasattr(dtypes[i], '__iter__') else args[i].dtype != dtypes[i]:
                    given = ', '.join([nameof(x.dtype) for x in args])
                    raise TypeError(f"{nameof(func_builder)} expects ({signature}). Calling with ({given})")
            return func_builder(*args)
        return decorated
    return wrapper


def check_exact_signature(signature, args, name):
    signature_repr = ','.join([nameof(x) for x in signature])
    giventype_repr = ','.join([nameof(x.dtype) for x in args])
    if len(signature) != len(args):
        raise TypeError(f"{name} takes exactly {len(signature)} arguments ({signature_repr}), {len(args)} given.")
    for idx in range(len(args)):
        if signature[idx] != args[idx].dtype:
            raise TypeError(f"{name} expects ({signature_repr}). Calling with ({giventype_repr})")


# type cast or initialization
# return dtype, expr
def builtin_type_cast(dtype, *args):
    # struct with constructor
    if type(dtype) is StructType and '__init__' in dtype.method_dict:
        obj = SimpleNamespace(dtype = dtype, expr = lcapi.builder().local(to_lctype(dtype)), lr = 'l')
        _rettype, _retexpr = callable_call(dtype.method_dict['__init__'], obj, *args)
        # if it's a constructor, make sure it doesn't return value
        if _rettype != None:
            raise TypeError(f'__init__() should return None, not {_rettype}')
        return dtype, obj.expr
    # default construct without arguments
    if len(args) == 0:
        # construct variable without initialization
        return dtype, lcapi.builder().local(to_lctype(dtype))
    # type cast of basic types
    # TODO may need temporary variable?
    if dtype in {int, float, bool}:
        if len(args) != 1:
            raise TypeError(f"Can't convert multiple values to {dtype.__name__}")
        if args[0].dtype not in {int, float, bool}:
            raise TypeError(f"Can't convert {args[0].dtype} to {dtype.__name__}")
        return dtype, lcapi.builder().cast(to_lctype(dtype), lcapi.CastOp.STATIC, args[0].expr)
    if dtype in vector_dtypes or dtype in matrix_dtypes:
        return builtin_func(f"make_{dtype.__name__}", *args)
    # TODO: vectors / matrices
    # TODO: array
    # TODO: struct
    raise NotImplementedError("only type cast to scalar types are currently supported")


def make_vector_call(dtype, op, args):
    # type check: must be corresponding scalar or vector of same element type
    assert dtype in {int, float, bool}
    dim = 1
    for arg in args:
        if not (arg.dtype == dtype or arg.dtype in vector_dtypes and element_of(arg.dtype) == dtype):
            print(arg.dtype, dtype)
            raise TypeError("arguments must be float or float vector")
        if arg.dtype in vector_dtypes:
            if dim != 1:
                if dim != to_lctype(arg.dtype).dimension():
                    raise TypeError("arguments can't contain vectors of different dimension")
            else:  # will upcast scalar to vector
                dim = to_lctype(arg.dtype).dimension()
    convtype = vector(dtype,dim)
    exprlist = []
    for arg in args:
        if arg.dtype == convtype:
            exprlist.append(arg.expr)
        else:
            dtype1, expr1 = builtin_type_cast(convtype, arg)
            exprlist.append(expr1)
    return convtype, lcapi.builder().call(to_lctype(convtype), op, exprlist)




@BuiltinFuncBuilder
def _bitwise_cast(*args):
    assert len(args)==2 and args[0].dtype == type
    dtype = args[0].expr
    assert dtype in (int, float)
    op = lcapi.CastOp.BITWISE
    # create temporary variable
    if args[1].lr == 'r':
        wrap_with_tmp_var(args[1])
    return dtype, lcapi.builder().cast(to_lctype(dtype), op, args[1].expr)


@BuiltinFuncBuilder
def _builtin_call(*args):
    if args[0].dtype == str: # void call
        op = getattr(lcapi.CallOp, args[0].expr)
        return None, lcapi.builder().call(op, [x.expr for x in args[1:]])
    else:
        check_exact_signature([type, str], args[0:2], "_builtin_call")
        dtype = args[0].expr
        op = getattr(lcapi.CallOp, args[1].expr)
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args[2:]])


@BuiltinFuncBuilder
@dtype_checked(int, int, int)
def set_block_size(x, y, z):
    for a in {x,y,z}:
        if type(a).__name__ != "Constant":
            raise TypeError("Because set_block_size is a compile-time instruction, arguments of set_block_size must be literal (constant).")
    return None, lcapi.builder().set_block_size(x.value, y.value, z.value)

@BuiltinFuncBuilder
@dtype_checked()
def synchronize_block():
    return None, lcapi.builder().call(lcapi.CallOp.SYNCHRONIZE_BLOCK, [])


def with_signature(*signatures):
    def wrapper(function_builder):
        def coerce_to(src_dtype, tgt_dtype) -> bool:
            try:
                return src_dtype in coerce_map[tgt_dtype]
            except KeyError:
                return src_dtype == tgt_dtype

        def match(func_name, signature, *args) -> (bool, str):
            signature_printed = ', '.join([nameof(x) for x in signature])
            if len(args) != len(signature):
                return False, f"{nameof(func_name)} takes {len(signature)} arguments ({signature_printed}), {len(args)} given."
            for arg, dtype in zip(args, signature):
                if not coerce_to(arg.dtype, dtype):
                    given = ', '.join([nameof(x.dtype) for x in args])
                    return False, f"{nameof(func_name)} expects ({signature_printed}). Calling with ({given})"
            return True, "Everything's alright."

        @functools.wraps(function_builder)
        def decorated(func_name, *args):
            reason = "Everything's alright."
            for signature in signatures:
                matched, reason = match(func_name, signature, *args)
                if matched:
                    return function_builder(func_name, *args)
            raise TypeError(reason)
        return decorated
    return wrapper


def with_checker(*checker_functions):
    def wrapper(function_builder):
        @functools.wraps(function_builder)
        def decorated(func_name, *args):
            for checker_function in checker_functions:
                if checker_function(*args) is False:
                    given = ', '.join([nameof(x.dtype) for x in args])
                    raise TypeError(f"type check failed for {func_name}, ({given}) is invalid because of "
                                    f"{checker_function.__name__}.")
            return function_builder(func_name, *args)
        return decorated
    return wrapper


class BuiltinFunctionCall:
    def __call__(self, name, *args, **kwargs):
        math_functions = (
            'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cos', 'cosh',
            'sin', 'sinh', 'tan', 'tanh', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10',
            'sqrt', 'rsqrt', 'ceil', 'floor', 'fract', 'trunc', 'round'
        )
        try:
            if name in math_functions:
                return self._invoke_math(name, *args)
            else:
                return getattr(self, f"invoke_{name}")(name, *args, **kwargs)
        except AttributeError:
            print(f"Error calling function {name}.")

    @staticmethod
    def _invoke_dispatch_related(name, *args, **kwargs):
        expr = getattr(lcapi.builder(), name)()
        dtype = int3
        expr1 = lcapi.builder().call(to_lctype(dtype), lcapi.CallOp.MAKE_INT3, [expr])
        tmp = lcapi.builder().local(to_lctype(dtype))
        lcapi.builder().assign(tmp, expr1)
        return dtype, tmp

    @staticmethod
    def _invoke_make_vector_n(data_type, vector_length, *args, **kwargs):
        op = getattr(lcapi.CallOp, f"make_{data_type.__name__}{vector_length}".upper())
        dtype = getattr(lcapi, f'{data_type.__name__}{vector_length}')
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_signature((float,), (float2,), (float3,), (float4,))
    def _invoke_math(name, *args):
        op = getattr(lcapi.CallOp, name.upper())
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_signature((int, int, int),)
    def invoke_set_block_size(_, *args, **kwargs):
        return set_block_size.builder(*args, **kwargs)

    @staticmethod
    @with_signature((),)
    def invoke_synchronize_block(_, *args, **kwargs):
        return synchronize_block.builder(*args, **kwargs)

    @staticmethod
    @with_signature((),)
    def invoke_thread_id(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_dispatch_related(name, *args, **kwargs)

    @staticmethod
    @with_signature((),)
    def invoke_block_id(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_dispatch_related(name, *args, **kwargs)

    @staticmethod
    @with_signature((),)
    def invoke_dispatch_id(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_dispatch_related(name, *args, **kwargs)

    @staticmethod
    @with_signature((),)
    def invoke_dispatch_size(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_dispatch_related(name, *args, **kwargs)

    @staticmethod
    @with_checker(multi_param, all_arithmetic, length_eq([1, 2]))
    def invoke_make_int2(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_make_vector_n(int, 2, *args, **kwargs)

    @staticmethod
    @with_checker(multi_param, all_arithmetic, length_eq([1, 3]))
    def invoke_make_int3(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_make_vector_n(int, 3, *args, **kwargs)

    @staticmethod
    @with_checker(multi_param, all_arithmetic, length_eq([1, 4]))
    def invoke_make_int4(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_make_vector_n(int, 4, *args, **kwargs)

    @staticmethod
    @with_checker(multi_param, all_arithmetic, length_eq([1, 2]))
    def invoke_make_float2(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_make_vector_n(float, 2, *args, **kwargs)

    @staticmethod
    @with_checker(multi_param, all_arithmetic, length_eq([1, 3]))
    def invoke_make_float3(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_make_vector_n(float, 3, *args, **kwargs)

    @staticmethod
    @with_checker(multi_param, all_arithmetic, length_eq([1, 4]))
    def invoke_make_float4(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_make_vector_n(float, 4, *args, **kwargs)

    @staticmethod
    @with_checker(multi_param, all_arithmetic, length_eq([1, 2]))
    def invoke_make_uint2(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_make_vector_n(uint, 2, *args, **kwargs)

    @staticmethod
    @with_checker(multi_param, all_arithmetic, length_eq([1, 3]))
    def invoke_make_uint3(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_make_vector_n(uint, 3, *args, **kwargs)

    @staticmethod
    @with_checker(multi_param, all_arithmetic, length_eq([1, 4]))
    def invoke_make_uint4(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_make_vector_n(uint, 4, *args, **kwargs)

    @staticmethod
    @with_checker(multi_param, all_bool, length_eq([1, 2]))
    def invoke_make_bool2(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_make_vector_n(bool, 2, *args, **kwargs)

    @staticmethod
    @with_checker(multi_param, all_bool, length_eq([1, 3]))
    def invoke_make_bool3(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_make_vector_n(bool, 3, *args, **kwargs)

    @staticmethod
    @with_checker(multi_param, all_bool, length_eq([1, 4]))
    def invoke_make_bool4(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_make_vector_n(bool, 4, *args, **kwargs)

    @staticmethod
    @with_signature((float,), (float2,), (float3,), (float4,))
    def invoke_isinf(name, *args, **kwargs):
        op = lcapi.CallOp.ISINF
        dtype = binary_type_infer.with_type(args[0].dtype, bool)
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_signature((float,), (float2,), (float3,), (float4,))
    def invoke_isnan(name, *args, **kwargs):
        op = lcapi.CallOp.ISNAN
        dtype = binary_type_infer.with_type(args[0].dtype, bool)
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_checker(unary, all_arithmetic)
    def invoke_abs(name, *args, **kwargs):
        op = lcapi.CallOp.ABS
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_signature((float2,), (float3,), (float4,))
    def invoke_length(name, *args, **kwargs):
        op = lcapi.CallOp.LENGTH
        dtype = float
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_signature((float2,), (float3,), (float4,))
    def invoke_length_squared(name, *args, **kwargs):
        op = lcapi.CallOp.LENGTH_SQUARED
        dtype = float
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_checker(binary, all_arithmetic, broadcast)
    def invoke_copysign(name, *args, **kwargs):
        op = lcapi.CallOp.COPYSIGN
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_checker(binary, all_arithmetic, broadcast)
    def invoke_min(name, *args, **kwargs):
        return make_vector_call(element_of(args[0].dtype), lcapi.CallOp.MIN, args)

    @staticmethod
    @with_checker(binary, all_arithmetic, broadcast)
    def invoke_max(name, *args, **kwargs):
        return make_vector_call(element_of(args[0].dtype), lcapi.CallOp.MAX, args)

    @staticmethod
    @with_signature((float2,), (float3,), (float4,))
    def invoke_normalize(name, *args, **kwargs):
        op = lcapi.CallOp.NORMALIZE
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_checker(binary, all_arithmetic, same_length)
    def invoke_dot(name, *args, **kwargs):
        op = lcapi.CallOp.DOT
        dtype = inner_type(args[0].dtype)
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_signature((float3, float3),)
    def invoke_cross(name, *args, **kwargs):
        op = lcapi.CallOp.CROSS
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_checker(ternary, all_float)
    def invoke_lerp(name, *args, **kwargs):
        return make_vector_call(float, lcapi.CallOp.LERP, args)

    @staticmethod
    @with_checker(ternary, with_inner_type((float, int), (float, int), (bool,)))
    def invoke_select(name, *args, **kwargs):
        op = lcapi.CallOp.SELECT
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    def invoke_print(name, *args, **kwargs):
        globalvars.printer.kernel_print(args)
        globalvars.current_context.uses_printer = True
        return None, None

    @staticmethod
    @with_checker(binary, all_arithmetic, broadcast)
    def invoke_pow(name, *args, **kwargs):
        for arg in args:
            if arg.dtype is not float:
                arg.dtype, arg.expr = builtin_type_cast(to_float(arg.dtype), arg)
        return make_vector_call(float, lcapi.CallOp.POW, args)

    @staticmethod
    @with_checker(unary, all_bool)
    def invoke_all(name, *args, **kwargs):
        op = lcapi.CallOp.ALL
        dtype = bool
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_checker(unary, all_bool)
    def invoke_any(name, *args, **kwargs):
        op = lcapi.CallOp.ANY
        dtype = bool
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_checker(ternary, all_arithmetic)
    def invoke_clamp(name, *args, **kwargs):
        op = lcapi.CallOp.CLAMP
        dtype = element_of(args[0].dtype)
        return make_vector_call(dtype, op, args)

    @staticmethod
    @with_checker(ternary, all_float)
    def invoke_fma(name, *args, **kwargs):
        op = lcapi.CallOp.FMA
        dtype = element_of(args[0].dtype)
        return make_vector_call(dtype, op, args)

    @staticmethod
    @with_checker(binary, all_arithmetic, same_length)
    def invoke_step(name, *args, **kwargs):
        op = lcapi.CallOp.STEP
        dtype = binary_type_infer.with_type(args[0].dtype, float)  # scalar -> float, vector<scalar> -> vector<float>
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_checker(unary, all_integer)
    def invoke_clz(name, *args, **kwargs):
        op = lcapi.CallOp.CLZ
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_checker(unary, all_integer)
    def invoke_ctz(name, *args, **kwargs):
        op = lcapi.CallOp.CTZ
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_checker(unary, all_integer)
    def invoke_popcount(name, *args, **kwargs):
        op = lcapi.CallOp.POPCOUNT
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_checker(unary, all_integer)
    def invoke_reverse(name, *args, **kwargs):
        op = lcapi.CallOp.REVERSE
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_signature((float3, float3, float3),)
    def invoke_faceforward(name, *args, **kwargs):
        op = lcapi.CallOp.FACEFORWARD
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_signature((float2x2,), (float3x3,), (float4x4,))
    def invoke_determinant(name, *args, **kwargs):
        op = lcapi.CallOp.DETERMINANT
        dtype = float
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [args[0].expr])

    @staticmethod
    @with_signature((float2x2,), (float3x3,), (float4x4,))
    def invoke_inverse(name, *args, **kwargs):
        op = lcapi.CallOp.INVERSE
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [args[0].expr])

    @staticmethod
    @with_signature((float2x2,), (float3x3,), (float4x4,))
    def invoke_transpose(name, *args, **kwargs):
        op = lcapi.CallOp.TRANSPOSE
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [args[0].expr])

    @staticmethod
    def invoke_array(name, *args, **kwargs):
        check_exact_signature([list], args, 'array')
        # deduce array dtype & length
        nodes = args[0].elts
        size = len(nodes)
        if size == 0:
            raise TypeError("Can't create empty array")
        dtype = nodes[0].dtype
        for x in nodes:
            if x.dtype != dtype:
                raise TypeError("all elements of array must be of same type")
        arrtype = ArrayType(dtype=dtype, size=size)
        # create & fill array
        arrexpr = lcapi.builder().local(to_lctype(arrtype))
        for idx in range(size):
            sliceexpr = lcapi.builder().literal(to_lctype(int), idx)
            lhs = lcapi.builder().access(to_lctype(dtype), arrexpr, sliceexpr)
            lcapi.builder().assign(lhs, nodes[idx].expr)
        return arrtype, arrexpr

    @staticmethod
    def invoke_struct(name, *args, **kwargs):
        # get alignment
        alignment = 1
        if len(args) > 0:
            if len(args) > 1 or args[0].dtype != int:
                raise TypeError("struct only takes an optional positional argument 'alignment' (int)")
            if type(args[0]).__name__ != "Constant":
                raise TypeError("alignment must be compile-time constant (literal).")
            alignment = args[0].value
        if 'alignment' in kwargs:
            if type(kwargs['alignment']).__name__ != "Constant":
                raise TypeError("alignment must be compile-time constant (literal).")
            alignment = kwargs.pop('alignment').value
        # deduce struct type
        strtype = StructType(alignment=alignment, **{name:kwargs[name].dtype for name in kwargs})
        # create & fill struct
        strexpr = lcapi.builder().local(to_lctype(strtype))
        for name in kwargs:
            idx = strtype.idx_dict[name]
            dtype = strtype.membertype[idx]
            lhs = lcapi.builder().member(to_lctype(dtype), strexpr, idx)
            lcapi.builder().assign(lhs, kwargs[name].expr)
        return strtype, strexpr

    @staticmethod
    @with_checker(unary, is_array)
    def invoke_len(name, *args, **kwargs):
        return int, lcapi.builder().literal(to_lctype(int), length_of(args[0].dtype))

    @staticmethod
    def invoke_make_ray(name, *args, **kwargs):
        from .accel import make_ray
        return callable_call(make_ray, *args)

    @staticmethod
    def invoke_inf_ray(name, *args, **kwargs):
        from .accel import inf_ray
        return callable_call(inf_ray, *args)

    @staticmethod
    def invoke_offset_ray_origin(name, *args, **kwargs):
        from .accel import offset_ray_origin
        return callable_call(offset_ray_origin, *args)


builtin_func = BuiltinFunctionCall()


def callable_call(func, *args):
    # get function instance by argtypes
    if func is globalvars.current_context.func and tuple(a.dtype for a in args) == globalvars.current_context.argtypes:
        raise Exception("Recursion is not supported")
    f = func.get_compiled(call_from_host=False, argtypes=tuple(a.dtype for a in args))
    globalvars.current_context.uses_printer |= f.uses_printer
    # create temporary var for each r-value argument
    for node in args:
        if node.lr == "r":
            wrap_with_tmp_var(node)
    # call
    if getattr(f, "return_type", None) == None:
        return None, lcapi.builder().call(f.function, [x.expr for x in args])
    else:
        dtype = f.return_type
        return dtype, lcapi.builder().call(to_lctype(dtype), f.function, [x.expr for x in args])
