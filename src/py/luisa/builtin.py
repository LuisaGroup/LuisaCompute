import lcapi
from .mathtypes import *
from .types import uint, to_lctype, BuiltinFuncBuilder, \
    scalar_dtypes, arithmetic_dtypes, vector_dtypes, matrix_dtypes, vector, length_of, element_of, nameof
import functools
from . import globalvars
from types import SimpleNamespace
import ast
from .array import ArrayType, SharedArrayType
from .struct import StructType


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
        return int  # TODO: 怎么表示一个 uint？


def deduce_broadcast(dtype0, dtype1):
    if dtype0 in [float, int, int, bool]:  # TODO: 在 dtype 里添加 uint 的表示
        return dtype1  # Broadcast
    elif dtype1 in [float, int, int, bool]:
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
        assert (dtype0 == dtype1) or \
               (length0 == 1 or length1 == 1 and element_of(dtype0) == element_of(dtype1)), \
            f'Binary operation between ({dtype0} and {dtype1}) is not supported'
    else:
        assert (dtype0 == dtype1) or \
               (length0 == 1 or length1 == 1 and element_of(dtype0) == element_of(dtype1)) or \
               (dtype0 == float2x2 and dtype1 == float2) or \
               (dtype0 == float3x3 and dtype1 == float3) or \
               (dtype0 == float4x4 and dtype1 == float4), \
            f'Binary operation between ({dtype0} and {dtype1}) is not supported'
    scalar_operation = length0 == length1 == 1
    dtype = None

    if op in (ast.Mod, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift):
        inner_type_0 = element_of(lhs.dtype)
        assert inner_type_0 in [int, uint], \
            f'operator `{op}` only supports `int` and `uint` types.'
        if scalar_operation:
            inner_type_1 = element_of(rhs.dtype)
            assert inner_type_1 in [int, uint], \
                f'operator `{op}` only supports `int` and `uint` types.'
            dtype = upper_scalar_dtype(dtype0, dtype1)
        else:
            assert element_of(rhs.dtype) == inner_type_0, \
                'operation between vectors of different types not supported.'
            dtype = deduce_broadcast(dtype0, dtype1)
        # and / or: bool allowed
    elif op in (ast.And, ast.Or):
        assert element_of(lhs.dtype) == element_of(rhs.dtype) == bool, f'operator `{op}` only supports `bool` type.'
        dtype = deduce_broadcast(dtype0, dtype1)
        # add / sub / div: int, uint and float allowed
        # relational: int, uint and float allowed
    elif op in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq):
        inner_type_0 = element_of(lhs.dtype)
        assert inner_type_0 in [int, float, uint], \
            f'operator `{op}` only supports `int`, `uint` and `float` types.'
        if scalar_operation:
            # allow implicit type conversion
            # so check rhs's type, ensure it also satisfies the constraints.
            inner_type_1 = element_of(rhs.dtype)
            assert inner_type_1 in [int, float, uint], \
                f'operator `{op}` only supports `int`, `uint` and `float` types.'
            dtype = upper_scalar_dtype(dtype0, dtype1)
        else:
            # forbid implicit type conversion
            # so check rhs's type, ensure it is the same with lhs
            assert element_of(rhs.dtype) == inner_type_0, \
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
    'sync_block',
    'thread_id', 'block_id', 'dispatch_id', 'dispatch_size',
    'kernel_id', 'object_id',
    'make_uint2', 'make_int2', 'make_float2', 'make_bool2',
    'make_uint3', 'make_int3', 'make_float3', 'make_bool3',
    'make_uint4', 'make_int4', 'make_float4', 'make_bool4',
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
        if type(dtype) == SharedArrayType:
            return dtype, lcapi.builder().shared(to_lctype(dtype))
        else:
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
def discard():
    return None, lcapi.builder().call(lcapi.CallOp.RASTER_DISCARD, [])
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
def set_block_size(x, y, z):
    values = []
    for a in [x,y,z]:
        if type(a).__name__ != "Constant":
            eval = lcapi.builder().try_eval_int(a.expr)
            if eval.exist():
                values.append(eval.value())
            else:
                raise TypeError("Because set_block_size is a compile-time instruction, arguments of set_block_size must be literal (constant).")
        else:
            values.append(a.value)
    for i in range(3):
        if type(values[i]) != int and type(values[i]) != uint:
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

# return dtype, expr
def builtin_func(name, *args, **kwargs):

    for f in [set_block_size, sync_block]:
        if name == f.__name__:
            return f.builder(*args, **kwargs)

    # e.g. dispatch_id()
    for func in 'thread_id', 'block_id', 'dispatch_id', 'dispatch_size':
        if name == func:
            assert len(args) == 0
            # NOTE: cast to signed int by default
            expr = getattr(lcapi.builder(), func)()
            dtype = int3
            expr1 = lcapi.builder().call(to_lctype(dtype), lcapi.CallOp.MAKE_INT3, [expr])
            tmp = lcapi.builder().local(to_lctype(dtype))
            lcapi.builder().assign(tmp, expr1)
            return dtype, tmp

    for func in 'kernel_id', 'object_id':
        if name == func:
            assert len(args) == 0
            # NOTE: cast to signed int by default
            expr = getattr(lcapi.builder(), func)()
            dtype = int
            tmp = lcapi.builder().local(to_lctype(dtype))
            lcapi.builder().assign(tmp, expr)
            return dtype, tmp

    # e.g. make_float4(...)
    for T in 'uint', 'int', 'float', 'bool':
        for N in 2, 3, 4:
            if name == f'make_{T}{N}':
                if sum([length_of(x.dtype) for x in args]) not in {1, N}:
                    raise ValueError(f"Argument length incorrect, expected 1 or {N}, found {sum([length_of(x.dtype) for x in args])}")
                # for x in args:
                #     if element_of(x.dtype) != {'int':int, 'float':float, 'bool':bool, 'uint':uint}[T] and \
                #         not (T == "float" and x.dtype == int):
                #         raise TypeError(f"Can't make {T}{N} from {x.dtype} (must be of same element type)")
                op = getattr(lcapi.CallOp, name.upper())
                dtype = getattr(lcapi, f'{T}{N}')
                return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    # e.g. make_float2x2(...)
    for N in 2, 3, 4:
        if name == f'make_float{N}x{N}':
            # for x in args:
            #     if element_of(x.dtype) != float and x.dtype != int:
            #         raise TypeError(f"Can't make {T}{N}x{N} from {x.dtype} (must be of same element type)")
            try:
                if len(args) == 1:
                    assert args[0].dtype in {float, int, float2x2, float3x3, float4x4}
                elif len(args) == N:
                    for arg in args:
                        assert arg.dtype == vector(float,N)
                elif len(args) == N*N:
                    for arg in args:
                        assert arg.dtype in {float, int}
            except AssertionError:
                raise TypeError(f"Can't make {T}{N}x{N} from {[x.dtype for x in args]}")
            op = getattr(lcapi.CallOp, name.upper())
            dtype = getattr(lcapi, f'float{N}x{N}')
            return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    # e.g. sin(x)
    if name == 'atan2':
        assert len(args) == 2
        assert (args[0].dtype in {float, float2, float3, float4}) and (args[0].dtype == args[1].dtype)
        op = getattr(lcapi.CallOp, name.upper())
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    if name in ('acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cos', 'cosh',
                'sin', 'sinh', 'tan', 'tanh', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10',
                'sqrt', 'rsqrt', 'ceil', 'floor', 'fract', 'trunc', 'round'):
        # type check: arg must be float / float vector
        assert len(args) == 1
        assert args[0].dtype in {float, float2, float3, float4}
        op = getattr(lcapi.CallOp, name.upper())
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])
        
    if name in ('isinf','isnan'):
        # type check: arg must be float / float vector
        assert len(args) == 1
        assert args[0].dtype in {float, float2, float3, float4}
        op = getattr(lcapi.CallOp, name.upper())
        dtype = to_bool(args[0].dtype)
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    if name in ('abs'):
        assert len(args) == 1
        assert args[0].dtype in arithmetic_dtypes
        op = getattr(lcapi.CallOp, name.upper())
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    if name in ('copysign'):
        assert len(args) == 2
        return make_vector_call(float, lcapi.CallOp.COPYSIGN, args)

    if name in ('min', 'max'):
        assert len(args) == 2
        op = getattr(lcapi.CallOp, name.upper())
        return make_vector_call(element_of(args[0].dtype), op, args)

    if name in ('length', 'length_squared'):
        assert len(args) == 1
        assert args[0].dtype in {float2, float3, float4}
        op = getattr(lcapi.CallOp, name.upper())
        return float, lcapi.builder().call(to_lctype(float), op, [x.expr for x in args])

    if name in ('normalize'):
        assert len(args) == 1
        assert args[0].dtype in {float2, float3, float4}
        op = getattr(lcapi.CallOp, name.upper())
        return args[0].dtype, lcapi.builder().call(to_lctype(args[0].dtype), op, [x.expr for x in args])

    if name in ('dot'):
        assert len(args) == 2
        assert args[0].dtype in {float2, float3, float4}
        assert args[0].dtype == args[1].dtype
        op = getattr(lcapi.CallOp, name.upper())
        return float, lcapi.builder().call(to_lctype(float), op, [x.expr for x in args])

    if name in ('cross'):
        assert len(args) == 2
        assert args[0].dtype == float3
        assert args[1].dtype == float3
        op = getattr(lcapi.CallOp, name.upper())
        return float3, lcapi.builder().call(to_lctype(float3), op, [x.expr for x in args])

    if name in ('lerp'):
        assert len(args) == 3
        return make_vector_call(float, lcapi.CallOp.LERP, args)

    if name in ('select'):
        assert len(args) == 3
        assert args[2].dtype in [bool, bool2, bool3, bool4]
        assert args[0].dtype == args[1].dtype
        assert args[2].dtype == bool or args[0].dtype in scalar_dtypes or \
            args[0].dtype in vector_dtypes and length_of(args[0].dtype) == length_of(args[2].dtype)
        return args[0].dtype, lcapi.builder().call(to_lctype(args[0].dtype), lcapi.CallOp.SELECT, [x.expr for x in args])

    if name == 'print':
        globalvars.printer.kernel_print(args)
        globalvars.current_context.uses_printer = True
        return None, None

    if name == 'pow':
        assert len(args) == 2
        for arg in args:
            if arg.dtype is not float:
                arg.dtype, arg.expr = builtin_type_cast(to_float(arg.dtype), arg)
        return make_vector_call(float, lcapi.CallOp.POW, args)

    if name in ('all', 'any'):
        op = getattr(lcapi.CallOp, name.upper())
        assert len(args) == 1
        assert args[0].dtype in vector_dtypes and element_of(args[0].dtype) is bool
        return bool, lcapi.builder().call(to_lctype(bool), op, [args[0].expr])
    if name == 'saturate':
        assert len(args) == 1
        e = element_of(args[0].dtype)
        assert e in {int, float}
        op = getattr(lcapi.CallOp, name.upper())
        return make_vector_call(e, op, args)

    if name in ('clamp', 'fma'):
        op = getattr(lcapi.CallOp, name.upper())
        assert len(args) == 3
        e = element_of(args[0].dtype)
        if name == 'clamp':
            assert e in {int, float}
        else:
            assert e == float
        return make_vector_call(e, op, args)

    if name == 'step':
        op = lcapi.CallOp.STEP
        assert len(args) == 2
        assert args[0].dtype == args[1].dtype and args[0].dtype in arithmetic_dtypes, \
               "invalid parameter"
        if args[0].dtype in {int, float}:
            # step(scalar, scalar) -> float
            dtype = float
        else:
            # step(vector<scalar>, vector<scalar>) -> vector(float)
            dtype = vector(float, length_of(args[0].dtype))
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])

    if name in ('clz', 'ctz', 'popcount', 'reverse'):
        op = getattr(lcapi.CallOp, name.upper())
        assert len(args) == 1
        assert args[0].dtype == int or \
                args[0].dtype in vector_dtypes and element_of(args[0].dtype) is int, \
               "invalid parameter"
        # clz(uint) -> uint
        # clz(vector<uint>) -> vector<uint>
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [args[0].expr])

    if name == 'faceforward':
        op = getattr(lcapi.CallOp, name.upper())
        assert len(args) == 3
        assert args[0].dtype == float3 and args[1].dtype == float3 and args[2].dtype == float3, \
               "invalid parameter"
        dtype = float3
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [arg.expr for arg in args])

    if name == 'reflect':
        op = getattr(lcapi.CallOp, name.upper())
        assert len(args) == 2
        assert args[0].dtype == float3 and args[1].dtype == float3, "invalid parameter"
        dtype = float3
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
        # get alignment
        alignment = 1
        if len(args) > 0:
            if len(args) > 1 or args[0].dtype != int:
                raise TypeError("struct only takes an optional positional argument 'alignment' (int)")
            if type(args[0]).__name__ != "Constant":
                eval_value = lcapi.builder().try_eval_int(args[0].expr)
                if eval_value.exist():
                    alignment = eval_value.value()
                else:
                    raise TypeError("alignment must be compile-time constant (literal).")
            else:
                alignment = args[0].value
        if 'alignment' in kwargs:
            if type(kwargs['alignment']).__name__ != "Constant":
                eval_value = lcapi.builder().try_eval_int(kwargs['alignment'].expr)
                if eval_value.exist():
                    alignment = eval_value.value()
                else:
                    raise TypeError("alignment must be compile-time constant (literal).")
            else:
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

    if name == 'len':
        assert len(args) == 1
        if type(args[0].dtype) is ArrayType or args[0].dtype in {*vector_dtypes, *matrix_dtypes}:
            return int, lcapi.builder().literal(to_lctype(int), length_of(args[0].dtype))
        raise TypeError(f"{nameof(args[0].dtype)} object has no len()")

    if name == 'make_ray':
        from .accel import make_ray
        return callable_call(make_ray, *args)

    if name == 'inf_ray':
        from .accel import inf_ray
        return callable_call(inf_ray, *args)

    if name == 'offset_ray_origin':
        from .accel import offset_ray_origin
        return callable_call(offset_ray_origin, *args)

    raise NameError(f'unrecognized function call {name}')


def callable_call(func, *args):
    shared_dict={}
    exprs = []
    idx = 0
    for i in args:
        if hasattr(i, "id") and str(i.dtype).find("SharedArrayType") == 0:
            shared_dict[idx]={"var":globalvars.current_context.local_variable[i.id]}
        else:
            exprs.append(i.expr)
        idx += 1
    idx += 1
    # get function instance by argtypes
    arg_list = tuple(a.dtype for a in args)
    if func is globalvars.current_context.func and arg_list == globalvars.current_context.argtypes:
        raise Exception("Recursion is not supported")
    f = func.get_compiled(call_from_host=False, allow_ref=True, argtypes=arg_list, arg_info=shared_dict)
    globalvars.current_context.uses_printer |= f.uses_printer
    # create temporary var for each r-value argument
    for node in args:
        if node.lr == "r":
            wrap_with_tmp_var(node)
    # call
    if getattr(f, "return_type", None) == None:
        return None, lcapi.builder().call(f.function, exprs)
    else:
        dtype = f.return_type
        return dtype, lcapi.builder().call(to_lctype(dtype), f.function, exprs)
