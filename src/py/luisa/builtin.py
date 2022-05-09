import lcapi

from .types import to_lctype, from_lctype, basic_type_dict, dtype_of, is_vector_type, basic_lctype_dict, scalar_dtypes, \
    scalar_lctypes, arithmetic_dtypes, arithmetic_lctypes
from functools import reduce
from . import globalvars
from types import SimpleNamespace
import ast
from .types import BuiltinFuncBuilder, ref as ref_type
from .array import ArrayType
from .struct import StructType


def wrap_with_tmp_var(node):
    tmp = lcapi.builder().local(to_lctype(node.dtype))
    lcapi.builder().assign(tmp, node.expr)
    node.expr = tmp
    node.lr = 'l'

def get_length(arg) -> int:
    lc_type = to_lctype(arg.dtype)
    if lc_type.is_scalar():
        return 1
    elif lc_type.is_array() or lc_type.is_vector() or lc_type.is_matrix() or lc_type.is_texture():
        return lc_type.dimension()
    else:
        assert False, f'Unknown argument type {arg.dtype}'


def get_inner_type(lc_type):
    if lc_type.is_scalar():
        return lc_type
    elif lc_type.is_array() or lc_type.is_vector() or lc_type.is_matrix() or lc_type.is_texture():
        return lc_type.element()
    else:
        assert False, 'Unknown argument type'


def check_type(lc_type):
    return lambda argument: to_lctype(argument.dtype) == lc_type


def check_types(dtype, arguments):
    return reduce(lambda x, y: x and y, map(check_type(to_lctype(dtype)), arguments))


def check_inner_type(lc_type):
    def check(argument):
        arg_type = to_lctype(argument.dtype)
        if arg_type.is_array() or arg_type.is_vector() or arg_type.is_matrix() or arg_type.is_texture():
            return arg_type.element() == lc_type
        else:
            return arg_type == lc_type

    return check


def check_inner_types(lc_type, arguments):
    return reduce(lambda x, y: x and y, map(check_inner_type(lc_type), arguments))


def check_type_in(dtypes, argument):
    lc_types = [to_lctype(dtype) for dtype in dtypes]
    return to_lctype(argument.dtype) in lc_types


def get_arg_length(arguments) -> int:
    return sum(map(get_length, arguments))


def implicit_coersion(dtype0, dtype1):
    if float in [dtype0, dtype1]:
        return float
    elif int in [dtype0, dtype1]:
        return int
    else:
        return int  # TODO: 怎么表示一个 uint？


def deduce_broadcast(dtype0, dtype1):
    if dtype0 in [float, int, int, bool]:  # TODO: 在 dtype 里添加 uint 的表示
        return dtype1  # Broadcast
    elif dtype1 in [float, int, int, bool]:
        return dtype0  # Broadcast
    else:
        return dtype1  # same size || Matrix * Vector -> Vector


def to(target_dtype, dtype):
    func = {
        int: to_int,
        float: to_float,
        bool: to_bool
        # int: to_uint
    }[target_dtype]
    return func(dtype)


def to_bool(dtype):
    if dtype in [float, int, int, bool]:
        return bool
    elif dtype in [lcapi.float2, lcapi.int2, lcapi.uint2, lcapi.bool2]:
        return lcapi.bool2
    elif dtype in [lcapi.float3, lcapi.int3, lcapi.uint3, lcapi.bool3]:
        return lcapi.bool3
    elif dtype in [lcapi.float4, lcapi.int4, lcapi.uint4, lcapi.bool4]:
        return lcapi.bool4


def to_float(dtype):
    if dtype in [float, int, int, bool]:
        return float
    elif dtype in [lcapi.float2, lcapi.int2, lcapi.uint2, lcapi.bool2]:
        return lcapi.float2
    elif dtype in [lcapi.float3, lcapi.int3, lcapi.uint3, lcapi.bool3]:
        return lcapi.float3
    elif dtype in [lcapi.float4, lcapi.int4, lcapi.uint4, lcapi.bool4]:
        return lcapi.float4


def to_int(dtype):
    if dtype in [float, int, int, bool]:
        return int
    elif dtype in [lcapi.float2, lcapi.int2, lcapi.uint2, lcapi.bool2]:
        return lcapi.int2
    elif dtype in [lcapi.float3, lcapi.int3, lcapi.uint3, lcapi.bool3]:
        return lcapi.int3
    elif dtype in [lcapi.float4, lcapi.int4, lcapi.uint4, lcapi.bool4]:
        return lcapi.int4


def to_uint(dtype):
    if dtype in [float, int, int, bool]:
        return int
    elif dtype in [lcapi.float2, lcapi.int2, lcapi.uint2, lcapi.bool2]:
        return lcapi.uint2
    elif dtype in [lcapi.float3, lcapi.int3, lcapi.uint3, lcapi.bool3]:
        return lcapi.uint3
    elif dtype in [lcapi.float4, lcapi.int4, lcapi.uint4, lcapi.bool4]:
        return lcapi.uint4


def builtin_unary_op(op, operand):
    lc_op = {
        ast.UAdd: lcapi.UnaryOp.PLUS,
        ast.USub: lcapi.UnaryOp.MINUS,
        ast.Not: lcapi.UnaryOp.NOT,
        ast.Invert: lcapi.UnaryOp.BIT_NOT
    }.get(op)
    dtype = operand.dtype
    length = get_length(operand)
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
    length0, length1 = get_length(lhs), get_length(rhs)
    lhs_expr, rhs_expr = lhs.expr, rhs.expr
    if op != ast.Mult:
        assert (dtype0 == dtype1) or \
               (length0 == 1 or length1 == 1), \
            'Broadcast operations between different sized vectors not supported'
    else:
        assert (dtype0 == dtype1) or \
               (length0 == 1 or length1 == 1) or \
               (dtype0 == lcapi.float2x2 and dtype1 == lcapi.float2) or \
               (dtype0 == lcapi.float3x3 and dtype1 == lcapi.float3) or \
               (dtype0 == lcapi.float4x4 and dtype1 == lcapi.float4), \
            'Broadcast operations between different sized vectors not supported'
    scalar_operation = length0 == length1 == 1
    dtype = None

    if op in (ast.Mod, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift):
        inner_lc_type_0 = get_inner_type(to_lctype(lhs.dtype))
        assert inner_lc_type_0 in [basic_type_dict[int], lcapi.Type.from_('uint')], \
            f'operator `{op}` only supports `int` and `uint` types.'
        if scalar_operation:
            inner_lc_type_1 = get_inner_type(to_lctype(rhs.dtype))
            assert inner_lc_type_1 in [basic_type_dict[int], lcapi.Type.from_('uint')], \
                f'operator `{op}` only supports `int` and `uint` types.'
            dtype = implicit_coersion(dtype0, dtype1)
        else:
            assert check_inner_type(inner_lc_type_0)(rhs), \
                'operation between vectors of different types not supported.'
            dtype = deduce_broadcast(dtype0, dtype1)
        # and / or: bool allowed
    elif op in (ast.And, ast.Or):
        assert check_inner_types(to_lctype(bool), [lhs, rhs]), f'operator `{op}` only supports `bool` type.'
        dtype = deduce_broadcast(dtype0, dtype1)
        # add / sub / div: int, uint and float allowed
        # relational: int, uint and float allowed
    elif op in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq):
        inner_lc_type_0 = get_inner_type(to_lctype(lhs.dtype))
        assert inner_lc_type_0 in [basic_type_dict[int], basic_type_dict[float], lcapi.Type.from_('uint')], \
            f'operator `{op}` only supports `int`, `uint` and `float` types.'
        if scalar_operation:
            # allow implicit type conversion
            # so check rhs's type, ensure it also satisfies the constraints.
            inner_lc_type_1 = get_inner_type(to_lctype(rhs.dtype))
            assert inner_lc_type_1 in [basic_type_dict[int], basic_type_dict[float], lcapi.Type.from_('uint')], \
                f'operator `{op}` only supports `int`, `uint` and `float` types.'
            dtype = implicit_coersion(dtype0, dtype1)
        else:
            # forbid implicit type conversion
            # so check rhs's type, ensure it is the same with lhs
            assert check_inner_type(inner_lc_type_0)(rhs), \
                'operation between vectors of different types not supported.'
            dtype = deduce_broadcast(dtype0, dtype1)
        if op in (ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq):
            dtype = to_bool(dtype)
        elif op is ast.Div:
            dtype = to_float(dtype)
            _, lhs_expr = builtin_type_cast(to_float(lhs.dtype), lhs)
            _, rhs_expr = builtin_type_cast(to_float(rhs.dtype), rhs)
    return dtype, lcapi.builder().binary(to_lctype(dtype), lc_op, lhs_expr, rhs_expr)


builtin_func_names = {
    'set_block_size',
    'thread_id', 'block_id', 'dispatch_id', 'dispatch_size',
    'make_uint2', 'make_int2', 'make_float2', 'make_bool2',
    'make_uint3', 'make_int3', 'make_float3', 'make_bool3',
    'make_uint4', 'make_int4', 'make_float4', 'make_bool4',
    'make_float2x2', 'make_float3x3', 'make_float4x4',
    'isinf', 'isnan', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cos', 'cosh',
    'sin', 'sinh', 'tan', 'tanh', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10',
    'sqrt', 'rsqrt', 'ceil', 'floor', 'fract', 'trunc', 'round',
    'abs', 'copysign',
    'dot', 'cross',
    'length', 'length_squared', 'normalize',
    'lerp',
    'print',
    'min', 'max',
    'all', 'any',
    'select', 'clamp', 'step',
    'clz', 'ctz', 'popcount', 'reverse',
    'fma', 'copysign',
    'determinant', 'transpose', 'inverse',
    'synchronize_block',
    'array', 'struct'
}


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
        assert len(args) == 1 and args[0].dtype in {int, float, bool}
        return dtype, lcapi.builder().cast(to_lctype(dtype), lcapi.CastOp.STATIC, args[0].expr)
    lctype = to_lctype(dtype)
    if lctype.is_vector() or lctype.is_matrix():
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
        if not (arg.dtype == dtype or is_vector_type(arg.dtype) and to_lctype(arg.dtype).element() == to_lctype(dtype)):
            print(arg.dtype, dtype)
            raise TypeError("arguments must be float or float vector")
        if is_vector_type(arg.dtype):
            if dim != 1:
                if dim != to_lctype(arg.dtype).dimension():
                    raise TypeError("arguments can't contain vectors of different dimension")
            else:  # will upcast scalar to vector
                dim = to_lctype(arg.dtype).dimension()
    convtype = getattr(lcapi, f'{dtype.__name__}{dim}') if dim > 1 else dtype
    exprlist = []
    for arg in args:
        if arg.dtype == convtype:
            exprlist.append(arg.expr)
        else:
            dtype1, expr1 = builtin_type_cast(convtype, arg)
            exprlist.append(expr1)
    return convtype, lcapi.builder().call(to_lctype(convtype), op, exprlist)


def check_exact_signature(signature, args, name):
    signature_repr = ','.join([getattr(x, '__name__', None) or repr(x) for x in signature])
    giventype_repr = ','.join([getattr(x.dtype, '__name__', None) or repr(x.dtype) for x in args])
    if len(signature) != len(args):
        raise TypeError(f"{name} takes exactly {len(signature)} arguments ({signature_repr}), {len(args)} given.")
    for idx in range(len(args)):
        if signature[idx] != args[idx].dtype:
            raise TypeError(f"{name} expects ({signature_repr}). Calling with ({giventype_repr})")



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


# @BuiltinFuncBuilder
# def set_block_size(x, y, z):
#     check_exact_signature([int, int, int], (x,y,z), "set_block_size")
#     for a in args:
#         if type(a).__name__ != "Constant":
#             raise TypeError("Because set_block_size is a compile-time instruction, arguments of set_block_size must be literal (constant).")
#     lcapi.builder().set_block_size(x.value, y.value, z.value)

# @BuiltinFuncBuilder
# def synchronize_block():
#     return None, lcapi.builder.call(lcapi.CallOp.SYNCHRONIZE_BLOCK, [])


# return dtype, expr
def builtin_func(name, *args, **kwargs):
    if name == "set_block_size":
        check_exact_signature([int, int, int], args, "set_block_size")
        for a in args:
            if type(a).__name__ != "Constant":
                raise TypeError(
                    "Because set_block_size is a compile-time instruction, arguments of set_block_size must be literal (constant).")
        lcapi.builder().set_block_size(*[a.value for a in args])
        return None, None

    if name == 'synchronize_block':
        assert len(args) == 0
        return None, lcapi.builder.call(None, lcapi.CallOp.SYNCHRONIZE_BLOCK, [])

    # e.g. dispatch_id()
    for func in 'thread_id', 'block_id', 'dispatch_id', 'dispatch_size':
        if name == func:
            assert len(args) == 0
            # NOTE: cast to signed int by default
            expr = getattr(lcapi.builder(), func)()
            dtype = lcapi.int3
            expr1 = lcapi.builder().call(to_lctype(dtype), lcapi.CallOp.MAKE_INT3, [expr])
            tmp = lcapi.builder().local(to_lctype(dtype))
            lcapi.builder().assign(tmp, expr1)
            return dtype, tmp

    # e.g. make_float4(...)
    for T in 'uint', 'int', 'float', 'bool':
        for N in 2, 3, 4:
            if name == f'make_{T}{N}':
                if get_arg_length(args) not in [1, N]:
                    raise ValueError(f"Argument length incorrect, expected 1 or {N}, found {get_arg_length(args)}")
                op = getattr(lcapi.CallOp, name.upper())
                dtype = getattr(lcapi, f'{T}{N}')
                return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    # e.g. make_float2x2(...)
    for N in 2, 3, 4:
        if name == f'make_float{N}x{N}':
            assert (len(args) == 1 and check_type_in([float, lcapi.float2x2, lcapi.float3x3, lcapi.float4x4], args[0])) \
                   or (len(args) == N and check_types(basic_lctype_dict[lcapi.Type.from_(f"vector<float,{N}>")], args)) \
                   or (len(args) == N * N and check_types(float, args)), 'type check failed'
            op = getattr(lcapi.CallOp, name.upper())
            dtype = getattr(lcapi, f'float{N}x{N}')
            return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    # TODO: atan2

    def element_type(_dtype):
        if _dtype in {int, float, bool}:
            return _dtype
        return from_lctype(to_lctype(_dtype).element())

    # e.g. sin(x)
    if name in ('acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cos', 'cosh',
                'sin', 'sinh', 'tan', 'tanh', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10',
                'sqrt', 'rsqrt', 'ceil', 'floor', 'fract', 'trunc', 'round'):
        # type check: arg must be float / float vector
        assert len(args) == 1
        assert args[0].dtype == float or \
               (to_lctype(args[0].dtype).is_vector() and to_lctype(args[0].dtype).element() == to_lctype(float))
        op = getattr(lcapi.CallOp, name.upper())
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])
        
    if name in ('isinf','isnan'):
        # type check: arg must be float / float vector
        assert len(args) == 1
        assert args[0].dtype == float or \
               (to_lctype(args[0].dtype).is_vector() and to_lctype(args[0].dtype).element() == to_lctype(float))
        op = getattr(lcapi.CallOp, name.upper())
        dtype = to_bool(args[0].dtype)
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    if name in ('abs',):
        assert len(args) == 1
        assert args[0].dtype in (int, float) or to_lctype(args[0].dtype).is_vector() and to_lctype(
            args[0].dtype).element() in (to_lctype(int), to_lctype(float))
        op = getattr(lcapi.CallOp, name.upper())
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    if name in ('copysign',):
        assert len(args) == 2
        return make_vector_call(float, lcapi.CallOp.COPYSIGN, args)

    if name in ('min', 'max'):
        assert len(args) == 2
        op = getattr(lcapi.CallOp, name.upper())
        return make_vector_call(element_type(args[0].dtype), op, args)

    if name in ('length', 'length_squared'):
        assert len(args) == 1
        assert to_lctype(args[0].dtype).is_vector() and to_lctype(args[0].dtype).element() == to_lctype(float)
        op = getattr(lcapi.CallOp, name.upper())
        return float, lcapi.builder().call(to_lctype(float), op, [x.expr for x in args])

    if name in ('normalize',):
        assert len(args) == 1
        assert to_lctype(args[0].dtype).is_vector() and to_lctype(args[0].dtype).element() == to_lctype(float)
        op = getattr(lcapi.CallOp, name.upper())
        return args[0].dtype, lcapi.builder().call(to_lctype(args[0].dtype), op, [x.expr for x in args])

    if name in ('dot',):
        assert len(args) == 2
        assert to_lctype(args[0].dtype).is_vector() and to_lctype(args[0].dtype).element() == to_lctype(float)
        assert to_lctype(args[1].dtype).is_vector() and to_lctype(args[1].dtype).element() == to_lctype(float)
        assert to_lctype(args[0].dtype).dimension() == to_lctype(args[1].dtype).dimension()
        op = getattr(lcapi.CallOp, name.upper())
        return float, lcapi.builder().call(to_lctype(float), op, [x.expr for x in args])

    if name in ('cross',):
        assert len(args) == 2
        assert args[0].dtype == lcapi.float3
        assert args[1].dtype == lcapi.float3
        op = getattr(lcapi.CallOp, name.upper())
        return lcapi.float3, lcapi.builder().call(to_lctype(lcapi.float3), op, [x.expr for x in args])

    if name in ('lerp',):
        assert len(args) == 3
        return make_vector_call(float, lcapi.CallOp.LERP, args)

    if name in ('select',):
        assert len(args) == 3
        assert args[2].dtype in [bool, lcapi.bool2, lcapi.bool3, lcapi.bool4]
        assert args[0].dtype == args[1].dtype
        assert args[2].dtype == bool or to_lctype(args[0].dtype).is_scalar() or \
            to_lctype(args[0].dtype).is_vector() and to_lctype(args[0].dtype).dimension() == to_lctype(args[2].dtype).dimension()
        return args[0].dtype, lcapi.builder().call(to_lctype(args[0].dtype), lcapi.CallOp.SELECT, [x.expr for x in args])

    if name == 'print':
        globalvars.printer.kernel_print(args)
        globalvars.current_context.uses_printer = True
        return None, None

    # buffer
    if name == "buffer_read":
        op = lcapi.CallOp.BUFFER_READ
        dtype = args[0].dtype.dtype
        check_exact_signature([int], args[1:], "Buffer.read")
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    if name == "buffer_write":
        op = lcapi.CallOp.BUFFER_WRITE
        dtype = args[0].dtype.dtype
        check_exact_signature([int, dtype], args[1:], "Buffer.write")
        lcapi.builder().call(op, [x.expr for x in args])
        return None, None

    if name == "texture2d_read":
        op = lcapi.CallOp.TEXTURE_READ
        dtype = getattr(lcapi, args[0].dtype.dtype.__name__ + "4")
        check_exact_signature([lcapi.int2], args[1:], "Texture2D.read")
        args[1].dtype, args[1].expr = builtin_type_cast(lcapi.uint2, args[1])  # convert int2 to uint2
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    if name == "texture2d_write":
        op = lcapi.CallOp.TEXTURE_WRITE
        dtype = getattr(lcapi, args[0].dtype.dtype.__name__ + "4")
        check_exact_signature([lcapi.int2, dtype], args[1:], "Texture2D.write")
        args[1].dtype, args[1].expr = builtin_type_cast(lcapi.uint2, args[1])  # convert int2 to uint2
        lcapi.builder().call(op, [x.expr for x in args])
        return None, None

    for N in (2, 3):
        if name == f'bindless_texture{N}d_sample':
            op = getattr(lcapi.CallOp, name.upper())
            uv_dtype = getattr(lcapi, f"float{N}")
            check_exact_signature([int, uv_dtype], args[1:], f'BindlessTexture{N}D.sample')
            # TODO: convert args[1] to uint
            dtype = lcapi.float4
            return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])

        if name == f'bindless_texture{N}d_sample_level':
            op = getattr(lcapi.CallOp, name.upper())
            uv_dtype = getattr(lcapi, f"float{N}")
            check_exact_signature([int, uv_dtype, float], args[1:], f'BindlessTexture{N}D.sample_level')
            # TODO: convert args[1] to uint
            dtype = lcapi.float4
            return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])

        if name == f'bindless_texture{N}d_sample_grad':
            op = getattr(lcapi.CallOp, name.upper())
            uv_dtype = getattr(lcapi, f"float{N}")
            check_exact_signature([int, uv_dtype, uv_dtype, uv_dtype], args[1:], f'BindlessTexture{N}D.sample_grad')
            # TODO: convert args[1] to uint
            dtype = lcapi.float4
            return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])

        if name == f'bindless_texture{N}d_read':
            op = getattr(lcapi.CallOp, name.upper())
            coord_dtype = getattr(lcapi, f"uint{N}")
            check_exact_signature([int, coord_dtype], args[1:], f'BindlessTexture{N}d.read')
            # TODO: convert args[1] to uint
            dtype = lcapi.float4
            return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])

        if name == f'bindless_texture{N}d_read_level':
            op = getattr(lcapi.CallOp, name.upper())
            coord_dtype = getattr(lcapi, f"uint{N}")
            check_exact_signature([int, coord_dtype, int], args[1:], f'BindlessTexture{N}d.read_level')
            # TODO: convert args[1] to uint
            dtype = lcapi.float4
            return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])

        if name == f'bindless_texture{N}d_size':
            op = getattr(lcapi.CallOp, name.upper())
            coord_dtype = getattr(lcapi, f"uint{N}")
            check_exact_signature([int, coord_dtype], args[1:], f'BindlessTexture{N}d.size')
            # TODO: convert args[1] to uint
            dtype = getattr(lcapi, f"uint{N}")
            return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])

        if name == f'bindless_texture{N}d_size_level':
            op = getattr(lcapi.CallOp, name.upper())
            coord_dtype = getattr(lcapi, f"uint{N}")
            check_exact_signature([int, coord_dtype, int], args[1:], f'BindlessTexture{N}d.size_level')
            # TODO: convert args[1] & args[3] to uint
            dtype = getattr(lcapi, f"uint{N}")
            return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])

    if name == 'pow':
        assert len(args) == 2
        for arg in args:
            if arg.dtype is not float:
                arg.dtype, arg.expr = builtin_type_cast(to_float(arg.dtype), arg)
        return make_vector_call(float, lcapi.CallOp.POW, args)

    if name in ('all', 'any'):
        op = getattr(lcapi.CallOp, name.upper())
        assert len(args) == 1
        assert to_lctype(args[0].dtype).is_vector() and to_lctype(args[0].dtype).element() == to_lctype(bool)
        return bool, lcapi.builder().call(to_lctype(bool), op, [args[0].expr])

    # if name == 'select':
    #     op = lcapi.CallOp.SELECT
    #     assert len(args) == 3
    #     lctypes = [to_lctype(arg.dtype) for arg in args]
    #     assert args[0].dtype == bool and args[1].dtype in scalar_dtypes and args[2].dtype in scalar_dtypes and \
    #            args[1].dtype == args[2].dtype or \
    #            lctypes[0].is_vector() and lctypes[0].element() == to_lctype(bool) and \
    #            lctypes[1].is_vector() and lctypes[1].element() in scalar_lctypes and \
    #            lctypes[2].is_vector() and lctypes[2].element() in scalar_lctypes and \
    #            lctypes[0].dimension() == lctypes[1].dimension() == lctypes[2].dimension and \
    #            lctypes[1].element() == lctypes[2].element(), "invalid parameter"
    #     dtype = args[1].dtype
    #     return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])

    if name in ('clamp', 'fma'):
        op = getattr(lcapi.CallOp, name.upper())
        assert len(args) == 3
        e = element_type(args[0].dtype)
        if name == 'clamp':
            assert e in arithmetic_dtypes
        else:
            assert e == 'float'
        return make_vector_call(e, op, args)

    if name == 'step':
        op = lcapi.CallOp.STEP
        assert len(args) == 2
        lctypes = [to_lctype(arg.dtype) for arg in args]
        assert args[0].dtype == args[1].dtype and args[0].dtype in arithmetic_dtypes or \
               lctypes[0] == lctypes[1] and lctypes[0].is_vector() and lctypes[0].element() in arithmetic_lctypes, \
               "invalid parameter"
        if args[0].dtype in arithmetic_dtypes:
            # step(scalar, scalar) -> float
            dtype = float
        else:
            # step(vector<scalar>, vector<scalar>) -> vector(float)
            dtype = lcapi.Type.from_(f"vector<float,{lctypes[0].dimension()}>")
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])

    if name in ('clz', 'ctz', 'popcount', 'reverse'):
        op = getattr(lcapi.CallOp, name.upper())
        assert len(args) == 1
        assert args[0].dtype == int or \
               to_lctype(args[0].dtype).is_vector() and to_lctype(args[0].dtype) == lcapi.Type.from_('int'), \
               "invalid parameter"
        # clz(uint) -> uint
        # clz(vector<uint>) -> vector<uint>
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [args[0].expr])

    if name == 'copysign':
        op = getattr(lcapi.CallOp, name.upper())
        assert len(args) == 2
        lctypes = [to_lctype(arg.dtype) for arg in args]
        assert args[0].dtype == args[1].dtype and args[0].dtype in arithmetic_dtypes or \
               lctypes[0] == lctypes[1] and lctypes[0].is_vector() and lctypes[0].element() in arithmetic_lctypes, \
               "invalid parameter"
        # copysign(scalar, scalar) -> scalar
        # copysign(vector<scalar>, vector<scalar>) -> vector<scalar>
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])

    if name == 'faceforward':
        op = getattr(lcapi.CallOp, name.upper())
        assert len(args) == 3
        assert args[0].dtype == lcapi.float3 and args[1].dtype == lcapi.float3 and args[2].dtype == lcapi.float3, \
               "invalid parameter"
        dtype = lcapi.float3
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])

    if name == 'determinant':
        op = getattr(lcapi.CallOp, name.upper())
        assert len(args) == 1
        assert to_lctype(args[0].dtype).is_matrix()
        dtype = float
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [args[0].expr])

    if name in ('transpose', 'inverse'):
        op = getattr(lcapi.CallOp, name.upper())
        assert len(args) == 1
        assert to_lctype(args[0].dtype).is_matrix()
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [args[0].expr])

    # UNUSABLE YET
    if name in ('atomic_exchange', 'atomic_fetch_add', 'atomic_fetch_sub', 'atomic_fetch_and', 'atomic_fetch_or',
                'atomic_fetch_xor', 'atomic_fetch_min', 'atomic_fetch_max'):
        op = getattr(lcapi.CallOp, name.upper())
        assert len(args) == 2
        assert args[0] is ref_type and args[0].dtype == args[1].dtype
        # TODO: Finish type check for atomic operations
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [args[0].expr])

    if name == 'atomic_compare_exchange':
        pass

    if name == 'array': # create array from list
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

    if name == 'struct': # create struct from kwargs
        # deduce struct type
        strtype = StructType(**{name:kwargs[name].dtype for name in kwargs})
        # create & fill struct
        strexpr = lcapi.builder().local(to_lctype(strtype))
        for name in kwargs:
            idx = strtype.idx_dict[name]
            dtype = strtype.membertype[idx]
            lhs = lcapi.builder().member(to_lctype(dtype), strexpr, idx)
            lcapi.builder().assign(lhs, kwargs[name].expr)
        return strtype, strexpr

    raise NameError(f'unrecognized function call {name}')


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
