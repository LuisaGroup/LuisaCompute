from .dylibs import lcapi
from .mathtypes import *
from .types import uint, uint3, float2, float3, float4, short, ushort, half, half2, half3, half4, long, ulong, to_lctype, is_bit16_types, is_bit64_types, BuiltinFuncBuilder, arithmetic_dtypes, vector_dtypes, scalar_and_vector_dtypes, matrix_dtypes, vector_and_matrix_dtypes, \
    vector, length_of, element_of, nameof, implicit_covertable, basic_dtypes
import functools
from . import globalvars
from types import SimpleNamespace
import ast
from .struct import StructType


def wrap_with_tmp_var(node):
    tmp = lcapi.builder().local(to_lctype(node.dtype))
    lcapi.builder().assign(tmp, node.expr)
    node.expr = tmp
    node.lr = 'l'


def upper_scalar_dtype(dtype0, dtype1):
    set = {dtype0, dtype1}
    if float in set:
        return float
    elif int in set:
        return int
    elif half in set:
        return half
    elif short in set:
        return short
    elif ushort in set:
        return ushort
    else:
        return uint


def deduce_broadcast(dtype0, dtype1):
    if dtype0 in {int, uint, float, short, ushort, half, bool}:
        return dtype1  # Broadcast
    elif dtype1 in {int, uint, float, short, ushort, half, bool}:
        return dtype0  # Broadcast
    else:
        return dtype1  # same size || Matrix * Vector -> Vector


def to_bool(dtype):
    assert dtype in scalar_and_vector_dtypes
    return vector(bool, length_of(dtype))


def to_float(dtype):
    if dtype in matrix_dtypes:
        return dtype
    assert dtype in scalar_and_vector_dtypes
    if is_bit16_types(dtype):
        return vector(half, length_of(dtype))
    return vector(float, length_of(dtype))


def to_int(dtype):
    assert dtype in scalar_and_vector_dtypes
    if is_bit16_types(dtype):
        return vector(short, length_of(dtype))
    if is_bit64_types(dtype):
        return vector(long, length_of(dtype))
    return vector(int, length_of(dtype))


def to_uint(dtype):
    assert dtype in scalar_and_vector_dtypes
    if is_bit16_types(dtype):
        return vector(ushort, length_of(dtype))
    if is_bit64_types(dtype):
        return vector(ulong, length_of(dtype))
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
    dtype0, dtype1 = lhs.dtype, rhs.dtype
    length0, length1 = length_of(dtype0), length_of(dtype1)
    lhs_expr, rhs_expr = lhs.expr, rhs.expr
    if op != ast.Mult:
        assert implicit_covertable(dtype0, dtype1) or \
               (length0 == 1 or length1 == 1 and implicit_covertable(element_of(dtype0), element_of(dtype1))), \
            f'Binary operation between ({dtype0} and {dtype1}) is not supported'
    else:
        assert implicit_covertable(dtype0, dtype1) or \
               (length0 == 1 or length1 == 1 and implicit_covertable(element_of(dtype0), element_of(dtype1))) or \
               (dtype0 == float2x2 and dtype1 in {float2, half2}) or \
               (dtype0 == float3x3 and dtype1 in {float3, half3}) or \
               (dtype0 == float4x4 and dtype1 in {float4, half4}), \
            f'Binary operation between ({dtype0} and {dtype1}) is not supported'
    scalar_operation = length0 == length1 == 1
    dtype = None

    if op in {ast.Mod, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift}:
        inner_type_0 = element_of(lhs.dtype)
        assert inner_type_0 in {int, uint, short, ushort, long, ulong}, \
            f'operator `{op}` only supports `int` and `uint` types.'
        if scalar_operation:
            inner_type_1 = element_of(rhs.dtype)
            assert inner_type_1 in {int, uint, short, ushort, long, ulong}, \
                f'operator `{op}` only supports `int` and `uint` types.'
            dtype = upper_scalar_dtype(dtype0, dtype1)
        else:
            assert implicit_covertable(element_of(rhs.dtype), inner_type_0), \
                'operation between vectors of different types not supported.'
            dtype = deduce_broadcast(dtype0, dtype1)
        # and / or: bool allowed
    elif op in {ast.And, ast.Or}:
        assert element_of(lhs.dtype) == element_of(rhs.dtype) == bool, f'operator `{op}` only supports `bool` type.'
        dtype = deduce_broadcast(dtype0, dtype1)
        # add / sub / div: int, uint and float allowed
        # relational: int, uint and float allowed
    elif op in {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq}:
        inner_type_0 = element_of(lhs.dtype)
        assert inner_type_0 in {int, uint, float, short, ushort, half, long, ulong}, \
            f'operator `{op}` only supports `int`, `uint` and `float` types.'
        if scalar_operation:
            # allow implicit type conversion
            # so check rhs's type, ensure it also satisfies the constraints.
            inner_type_1 = element_of(rhs.dtype)
            assert inner_type_1 in {int, uint, float, short, ushort, half, long, ulong}, \
                f'operator `{op}` only supports `int`, `uint` and `float` types.'
            dtype = upper_scalar_dtype(dtype0, dtype1)
        else:
            # forbid implicit type conversion
            # so check rhs's type, ensure it is the same with lhs
            assert implicit_covertable(element_of(rhs.dtype), inner_type_0), \
                'operation between vectors of different types not supported.'
            dtype = deduce_broadcast(dtype0, dtype1)
        if op in {ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq}:
            dtype = to_bool(dtype)
        elif op is ast.Div:
            dtype = to_float(dtype)
            _, lhs_expr = builtin_type_cast(to_float(lhs.dtype), lhs)
            _, rhs_expr = builtin_type_cast(to_float(rhs.dtype), rhs)
    return dtype, lcapi.builder().binary(to_lctype(dtype), lc_op, lhs_expr, rhs_expr)


builtin_func_names = {
    'set_block_size',
    'sync_block',
    'thread_id', 'block_id', 'dispatch_id', 'dispatch_size',
    'kernel_id', 'object_id',
    'make_uint2', 'make_int2', 'make_float2', 'make_bool2',
    'make_uint3', 'make_int3', 'make_float3', 'make_bool3',
    'make_uint4', 'make_int4', 'make_float4', 'make_bool4',
    'make_ushort2', 'make_short2', 'make_half2',
    'make_ushort3', 'make_short3', 'make_half3',
    'make_ushort4', 'make_short4', 'make_half4',
    'make_ulong2', 'make_long2',
    'make_ulong3', 'make_long3',
    'make_ulong4', 'make_long4',
    'make_float2x2', 'make_float3x3', 'make_float4x4',
    'isinf', 'isnan', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'atan2', 'cos', 'cosh',
    'sin', 'sinh', 'tan', 'tanh', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10',
    'sqrt', 'rsqrt', 'ceil', 'floor', 'fract', 'trunc', 'round', 'abs', 'pow',
    'dot', 'cross',
    'length', 'length_squared', 'normalize',
    'copysign', 'fma',
    'min', 'max',
    'all', 'any',
    'select', 'clamp', 'saturate', 'step', 'lerp',
    'clz', 'ctz', 'popcount', 'reverse',
    'determinant', 'transpose', 'inverse', "faceforward", "reflect",
    'print',
    'len', 'ddx', 'ddy'
}


# each argument can be a dtype or list/tuple of dtype
def dtype_checked(*dtypes):
    signature = ', '.join([nameof(x) for x in dtypes])

    def wrapper(func_builder):
        @functools.wraps(func_builder)
        def decorated(*args):
            if len(args) != len(dtypes):
                raise TypeError(
                    f"{nameof(func_builder)} takes exactly {len(dtypes)} arguments ({signature}), {len(args)} given.")
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
        if not implicit_covertable(signature[idx], args[idx].dtype):
            raise TypeError(f"{name} expects ({signature_repr}). Calling with ({giventype_repr})")


# type cast or initialization
# return dtype, expr
def builtin_type_cast(dtype, *args):
    # struct with constructor
    if type(dtype) is StructType and '__init__' in dtype.method_dict:
        obj = SimpleNamespace(dtype=dtype, expr=lcapi.builder().local(to_lctype(dtype)), lr='l')
        _rettype, _retexpr = callable_call(dtype.method_dict['__init__'], obj, *args)
        # if it's a constructor, make sure it doesn't return value
        if _rettype is not None:
            raise TypeError(f'__init__() should return None, not {_rettype}')
        return dtype, obj.expr
    # default construct without arguments
    if len(args) == 0:
        # construct variable without initialization
        if type(dtype).__name__ == "SharedArrayType":
            return dtype, lcapi.builder().shared(to_lctype(dtype))
        else:
            return dtype, lcapi.builder().local(to_lctype(dtype))
    # type cast of basic types
    # TODO may need temporary variable?
    if dtype in {int, uint, float, short, ushort, half, bool, long, ulong}:
        if len(args) != 1:
            raise TypeError(f"Can't convert multiple values to {dtype.__name__}")
        if args[0].dtype not in {int, uint, float, short, ushort, half, uint, long, ulong}:
            raise TypeError(f"Can't convert {args[0].dtype} to {dtype.__name__}")
        return dtype, lcapi.builder().cast(to_lctype(dtype), lcapi.CastOp.STATIC, args[0].expr)
    if dtype in vector_and_matrix_dtypes:
        return builtin_func(f"make_{dtype.__name__}", *args)
    # TODO: vectors / matrices
    # TODO: array
    # TODO: struct
    raise NotImplementedError("only type cast to scalar types are currently supported")


def make_vector_call(dtype, op, args):
    # type check: must be corresponding scalar or vector of same element type
    assert dtype in {int, uint, float, short, ushort, half, bool, long, ulong}
    dim = 1
    for arg in args:
        if not (implicit_covertable(arg.dtype, dtype) or arg.dtype in vector_dtypes and implicit_covertable(
                element_of(arg.dtype), dtype)):
            raise TypeError("arguments must be float or float vector")
        if arg.dtype in vector_dtypes:
            if dim != 1:
                if dim != to_lctype(arg.dtype).dimension():
                    raise TypeError("arguments can't contain vectors of different dimension")
            else:  # will upcast scalar to vector
                dim = to_lctype(arg.dtype).dimension()
    convtype = vector(dtype, dim)
    exprlist = []
    for arg in args:
        if implicit_covertable(arg.dtype, convtype):
            exprlist.append(arg.expr)
        else:
            dtype1, expr1 = builtin_type_cast(convtype, arg)
            exprlist.append(expr1)
    return convtype, lcapi.builder().call(to_lctype(convtype), op, exprlist)


@BuiltinFuncBuilder
def discard():
    return None, lcapi.builder().call(lcapi.CallOp.RASTER_DISCARD, [])


@BuiltinFuncBuilder
def bitwise_cast(*args):
    assert len(args) == 2 and args[0].dtype == type
    dtype = args[0].expr
    assert dtype in {int, uint, float, short, ushort, half, long, ulong}
    op = lcapi.CastOp.BITWISE
    return dtype, lcapi.builder().cast(to_lctype(dtype), op, args[1].expr)


@BuiltinFuncBuilder
def _builtin_call(*args):
    if args[0].dtype == str:  # void call
        op = getattr(lcapi.CallOp, args[0].expr)
        return None, lcapi.builder().call(op, [x.expr for x in args[1:]])
    else:
        check_exact_signature([type, str], args[0:2], "_builtin_call")
        dtype = args[0].expr
        op = getattr(lcapi.CallOp, args[1].expr)
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args[2:]])


@BuiltinFuncBuilder
def set_block_size(x, y, z):
    values = []
    for a in [x, y, z]:
        if type(a).__name__ != "Constant":
            eval = lcapi.builder().try_eval_int(a.expr)
            if eval.exist():
                values.append(eval.value())
            else:
                raise TypeError(
                    "Because set_block_size is a compile-time instruction, arguments of set_block_size must be literal (constant).")
        else:
            values.append(a.value)
    for i in range(3):
        if not type(values[i]) in {int, uint}:
            raise TypeError(f"set_block_size argument {i} must be int or uint")
        elif values[i] == 0:
            raise ValueError(f"block size can not be 0")
    accum = values[0] * values[1] * values[2]
    if accum > 1024:
        raise ValueError(f"block size should be less than or equal to 1024")
    return None, lcapi.builder().set_block_size(values[0], values[1], values[2])


@BuiltinFuncBuilder
@dtype_checked()
def sync_block():
    return None, lcapi.builder().call(lcapi.CallOp.SYNCHRONIZE_BLOCK, [])


_func_map = {}


def _compute_xx_id(name, *args):
    assert len(args) == 0
    # NOTE: cast to signed int by default
    expr = getattr(lcapi.builder(), name)()
    dtype = uint3
    # expr1 = lcapi.builder().call(to_lctype(dtype), lcapi.CallOp.MAKE_INT3, [expr])
    # tmp = lcapi.builder().local(to_lctype(dtype))
    # lcapi.builder().assign(tmp, expr1)
    return dtype, expr


for _func in 'thread_id', 'block_id', 'dispatch_id', 'dispatch_size':
    _func_map[_func] = _compute_xx_id


def _custom_xx_id(name, *args):
    assert len(args) == 0
    # NOTE: cast to signed int by default
    expr = getattr(lcapi.builder(), name)()
    dtype = uint
    # tmp = lcapi.builder().local(to_lctype(dtype))
    # lcapi.builder().assign(tmp, expr)
    return dtype, expr


for _func in 'kernel_id', 'object_id':
    _func_map[_func] = _custom_xx_id


# e.g. make_float4(...)
def _make_vec(name, *args):
    N = int(name[len(name) - 1:len(name)])
    T = name[5:len(name) - 1]
    if sum([length_of(x.dtype) for x in args]) not in {1, N}:
        raise ValueError(
            f"Argument length incorrect, expected 1 or {N}, found {sum([length_of(x.dtype) for x in args])}")
    op = getattr(lcapi.CallOp, name.upper())
    dtype = getattr(lcapi, f'{T}{N}')
    return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])


for T in 'uint', 'int', 'float', 'bool':
    for N in 2, 3, 4:
        _func_map[f'make_{T}{N}'] = _make_vec


def _make_vec_16(name, *args):
    N = int(name[len(name) - 1:len(name)])
    T = name[5:len(name)]
    if sum([length_of(x.dtype) for x in args]) not in {1, N}:
        raise ValueError(
            f"Argument length incorrect, expected 1 or {N}, found {sum([length_of(x.dtype) for x in args])}")
    dtype = eval(T)
    upper_name = name.upper()
    op = getattr(lcapi.CallOp, upper_name)
    return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])


for T in 'ushort', 'short', 'half', 'long', 'ulong':
    for N in 2, 3, 4:
        _func_map[f'make_{T}{N}'] = _make_vec_16

def _make_matrices(name, *args):
    for N in 2, 3, 4:
        if name == f'make_float{N}x{N}':
            try:
                if len(args) == 1:
                    assert args[0].dtype in {float2x2, float3x3, float4x4}
                elif len(args) == N:
                    for arg in args:
                        assert arg.dtype == vector(float, N)
                elif len(args) == N * N:
                    for arg in args:
                        assert arg.dtype in {int, uint, float, short, ushort, half}
            except AssertionError:
                raise TypeError(f"Can't make {T}{N}x{N} from {[x.dtype for x in args]}")
            op = getattr(lcapi.CallOp, name.upper())
            dtype = getattr(lcapi, f'float{N}x{N}')
            return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])


for N in 2, 3, 4:
    _func_map[f'make_float{N}x{N}'] = _make_matrices


def _atan2(name, *args):
    assert len(args) == 2
    assert (args[0].dtype in {float, float2, float3, float4, half, half2, half3, half4}) and (
                args[0].dtype == args[1].dtype)
    op = getattr(lcapi.CallOp, name.upper())
    dtype = args[0].dtype
    return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])


_func_map["atan2"] = _atan2


def _one_float_arg(name, *args):
    assert len(args) == 1
    assert args[0].dtype in {float, float2, float3, float4, half, half2, half3, half4}
    op = getattr(lcapi.CallOp, name.upper())
    dtype = args[0].dtype
    return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])


for name in (
'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cos', 'cosh', 'sin', 'sinh', 'tan', 'tanh', 'exp', 'exp2', 'exp10',
'log', 'log2', 'log10', 'sqrt', 'rsqrt', 'ceil', 'floor', 'fract', 'trunc', 'round', 'saturate'):
    _func_map[name] = _one_float_arg
    # type check: arg must be float / float vector


def _is(name, *args):
    assert len(args) == 1
    assert args[0].dtype in {float, float2, float3, float4, half, half2, half3, half4}
    op = getattr(lcapi.CallOp, name.upper())
    dtype = to_bool(args[0].dtype)
    return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])


for name in ('isinf', 'isnan'):
    _func_map[name] = _is


def _abs(name, *args):
    assert len(args) == 1
    assert args[0].dtype in arithmetic_dtypes
    op = getattr(lcapi.CallOp, name.upper())
    dtype = args[0].dtype
    return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])


_func_map["abs"] = _abs


def _copysign(name, *args):
    assert len(args) == 2
    return make_vector_call(element_of(args[0].dtype), lcapi.CallOp.COPYSIGN, args)


_func_map["copysign"] = _copysign


def _minmax(name, *args):
    assert len(args) == 2
    op = getattr(lcapi.CallOp, name.upper())
    return make_vector_call(element_of(args[0].dtype), op, args)


for name in ('min', 'max'):
    _func_map[name] = _minmax


def _len(name, *args):
    assert len(args) == 1
    assert args[0].dtype in {float2, float3, float4, half2, half3, half4}
    op = getattr(lcapi.CallOp, name.upper())
    return element_of(args[0].dtype), lcapi.builder().call(to_lctype(float), op, [x.expr for x in args])


for name in ('length', 'length_squared'):
    _func_map[name] = _len


def _normalize(name, *args):
    assert len(args) == 1
    assert args[0].dtype in {float2, float3, float4, half2, half3, half4}
    op = getattr(lcapi.CallOp, name.upper())
    return args[0].dtype, lcapi.builder().call(to_lctype(args[0].dtype), op, [x.expr for x in args])


_func_map["normalize"] = _normalize


def _dot(name, *args):
    assert len(args) == 2
    assert args[0].dtype in {float2, float3, float4, half2, half3, half4}
    assert args[0].dtype == args[1].dtype
    op = getattr(lcapi.CallOp, name.upper())
    return element_of(args[0].dtype), lcapi.builder().call(to_lctype(element_of(args[0].dtype)), op, [x.expr for x in args])


_func_map["dot"] = _dot


def _dd(name, *args):
    assert len(args) == 1
    assert args[0].dtype in basic_dtypes
    op = getattr(lcapi.CallOp, name.upper())
    return args[0].dtype, lcapi.builder().call(to_lctype(args[0].dtype), op, [args[0].expr])


_func_map["ddx"] = _dd
_func_map["ddy"] = _dd


def _cross(name, *args):
    assert len(args) == 2
    assert args[0].dtype in {float3, half3}
    assert args[1].dtype in {float3, half3}
    op = getattr(lcapi.CallOp, name.upper())
    return args[0].dtype, lcapi.builder().call(to_lctype(args[0].dtype), op, [x.expr for x in args])


_func_map["cross"] = _cross


def _lerp(name, *args):
    t_len = length_of(args[2].dtype)
    assert len(args) == 3 and (args[0].dtype == args[1].dtype) and (length_of(args[0].dtype) == t_len or t_len == 1)
    return make_vector_call(element_of(args[0].dtype), lcapi.CallOp.LERP, args)


_func_map["lerp"] = _lerp


def _select(name, *args):
    bool_vec_len = length_of(args[2].dtype)
    assert len(args) == 3 and \
           args[2].dtype in {bool, bool2, bool3, bool4} and \
           args[0].dtype == args[1].dtype and \
           args[0].dtype in scalar_and_vector_dtypes and \
           (length_of(args[0].dtype) == bool_vec_len or bool_vec_len == 1)
    return args[0].dtype, lcapi.builder().call(to_lctype(args[0].dtype), lcapi.CallOp.SELECT, [x.expr for x in args])


_func_map["select"] = _select


def _print(name, *args):
    globalvars.printer.kernel_print(args)
    globalvars.current_context.uses_printer = True
    return None, None


_func_map["print"] = _print


def _pow(name, *args):
    assert len(args) == 2
    for arg in args:
        if not (arg.dtype in {float, half}):
            arg.dtype, arg.expr = builtin_type_cast(to_float(arg.dtype), arg)
    return make_vector_call(element_of(arg.dtype), lcapi.CallOp.POW, args)


_func_map["pow"] = _pow


def _aa(name, *args):
    op = getattr(lcapi.CallOp, name.upper())
    assert len(args) == 1
    assert args[0].dtype in vector_dtypes and element_of(args[0].dtype) is bool
    return bool, lcapi.builder().call(to_lctype(bool), op, [args[0].expr])


for name in ('all', 'any'):
    _func_map[name] = _aa


def _tri_arg(name, *args):
    op = getattr(lcapi.CallOp, name.upper())
    assert len(args) == 3
    e = element_of(args[0].dtype)
    if name == 'clamp':
        assert e in {int, uint, float, short, ushort, half}
    else:
        assert e in {float, half}
    return make_vector_call(e, op, args)


for name in ('clamp', 'fma'):
    _func_map[name] = _tri_arg


def _step(name, *args):
    op = lcapi.CallOp.STEP
    assert len(args) == 2
    assert implicit_covertable(args[0].dtype, args[1].dtype) and args[0].dtype in arithmetic_dtypes, \
        "invalid parameter"
    if args[0].dtype in {int, uint, float}:
        # step(scalar, scalar) -> float
        dtype = float
    elif args[0].dtype in {short, ushort, half}:
        dtype = half
    else:
        # step(vector<scalar>, vector<scalar>) -> vector(float)
        dtype = vector(float, length_of(args[0].dtype))
    return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])


_func_map["step"] = _step


def _int_func(name, *args):
    op = getattr(lcapi.CallOp, name.upper())
    assert len(args) == 1
    assert args[0].dtype == int or \
           args[0].dtype in vector_dtypes and element_of(args[0].dtype) is int, \
        "invalid parameter"
    # clz(uint) -> uint
    # clz(vector<uint>) -> vector<uint>
    dtype = args[0].dtype
    return dtype, lcapi.builder().call(to_lctype(dtype), op, [args[0].expr])


for name in ('clz', 'ctz', 'popcount', 'reverse'):
    _func_map[name] = _int_func


def _faceforward(name, *args):
    op = getattr(lcapi.CallOp, name.upper())
    f3_map = {float3, half3}
    assert len(args) == 3 and \
           args[0].dtype in f3_map and args[1].dtype in f3_map and args[2].dtype in f3_map, \
        "invalid parameter"
    dtype = args[0].dtype
    return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])


_func_map["faceforward"] = _faceforward


def _reflect(name, *args):
    op = getattr(lcapi.CallOp, name.upper())
    f3_map = {float3, half3}
    assert len(args) == 2
    assert args[0].dtype in f3_map and args[1].dtype in f3_map, "invalid parameter"
    dtype = args[0].dtype
    return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])


_func_map["reflect"] = _reflect


def _det(name, *args):
    op = getattr(lcapi.CallOp, name.upper())
    assert len(args) == 1
    assert to_lctype(args[0].dtype).is_matrix()
    dtype = float
    return dtype, lcapi.builder().call(to_lctype(dtype), op, [args[0].expr])


_func_map["determinant"] = _det


def _mats(name, *args):
    op = getattr(lcapi.CallOp, name.upper())
    assert len(args) == 1
    assert to_lctype(args[0].dtype).is_matrix()
    dtype = args[0].dtype
    return dtype, lcapi.builder().call(to_lctype(dtype), op, [args[0].expr])


for name in ('transpose', 'inverse'):
    _func_map[name] = _mats


def _len(name, *args):
    assert len(args) == 1
    if type(args[0].dtype).__name__ == "ArrayType" or args[0].dtype in vector_and_matrix_dtypes:
        return int, lcapi.builder().literal(to_lctype(int), length_of(args[0].dtype))
    raise TypeError(f"{nameof(args[0].dtype)} object has no len()")


_func_map["len"] = _len


# return dtype, expr
def builtin_func(name, *args, **kwargs):
    for f in [set_block_size, sync_block]:
        if name == f.__name__:
            return f.builder(*args, **kwargs)
    func = _func_map.get(name)
    if func is None:
        raise NameError(f'unrecognized function call {name}')
    return func(name, *args)


def callable_call(func, *args):
    shared_dict = {}
    exprs = []
    idx = 0
    for node in args:
        if node.lr == "r":
            lctype = to_lctype(node.dtype)
            if lctype.is_array() or lctype.is_structure() or lctype.is_custom():
                wrap_with_tmp_var(node)
    for i in args:
        if hasattr(i, "id") and str(i.dtype).find("SharedArrayType") == 0:
            shared_dict[idx] = {"var": globalvars.current_context.local_variable[i.id]}
        else:
            exprs.append(i.expr)
        idx += 1
    idx += 1
    # get function instance by argtypes
    arg_list = tuple(a.dtype for a in args)
    if func is globalvars.current_context.func and arg_list == globalvars.current_context.argtypes:
        raise Exception("Recursion is not supported")
    f = func.get_compiled(func_type=1, allow_ref=True, argtypes=arg_list, arg_info=shared_dict)
    globalvars.current_context.uses_printer |= f.uses_printer
    # create temporary var for each r-value argument
    # call
    if getattr(f, "return_type", None) is None:
        return None, lcapi.builder().call(f.function, exprs)
    else:
        dtype = f.return_type
        return dtype, lcapi.builder().call(to_lctype(dtype), f.function, exprs)
