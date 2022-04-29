import lcapi
from .types import to_lctype, from_lctype, basic_type_dict, dtype_of, is_vector_type
from functools import reduce
from . import globalvars
from .structtype import StructType
from types import SimpleNamespace
import ast


def get_length(arg) -> int:
    lc_type = to_lctype(arg.dtype)
    if lc_type.is_scalar():
        return 1
    elif lc_type.is_array() or lc_type.is_vector() or lc_type.is_matrix() or lc_type.is_texture():
        return lc_type.dimension()
    else:
        assert False, 'Unknown argument type'


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
        raise Exception(f'Unsupported compare operation: {op}')
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
            _, lhs_expr = builtin_type_cast(to_float(lhs.dtype), [lhs])
            _, rhs_expr = builtin_type_cast(to_float(rhs.dtype), [rhs])
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
    'length', 'normalize',
    'lerp',
    'print',
    'min', 'max'
}


# type cast or initialization
# return dtype, expr
def builtin_type_cast(dtype, args):
    # struct with constructor
    if type(dtype) is StructType and '__init__' in dtype.method_dict:
        obj = SimpleNamespace()
        obj.dtype = dtype
        obj.expr = lcapi.builder().local(to_lctype(dtype))
        callable_call(dtype.method_dict['__init__'], [obj] + args)
        return dtype, obj.expr
    # default construct without arguments
    if len(args) == 0:
        # construct variable without initialization
        return dtype, lcapi.builder().local(to_lctype(dtype))
    # type cast of basic types
    if dtype in {int, float, bool}:
        assert len(args) == 1 and args[0].dtype in {int, float, bool}
        return dtype, lcapi.builder().cast(to_lctype(dtype), lcapi.CastOp.STATIC, args[0].expr)
    lctype = to_lctype(dtype)
    if lctype.is_vector() or lctype.is_matrix():
        return builtin_func(f"make_{dtype.__name__}", args)
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
            raise TypeError("arguments must be float or float vector")
        if is_vector_type(arg.dtype):
            if dim != 1:
                if dim != to_lctype(arg.dtype).dimension():
                    raise TypeError("arguments can't contain vectors of different dimension")
            else: # will upcast scalar to vector
                dim = to_lctype(arg.dtype).dimension()
    convtype = getattr(lcapi, f'{dtype.__name__}{dim}') if dim>1 else dtype
    exprlist = []
    for arg in args:
        if arg.dtype == convtype:
            exprlist.append(arg.expr)
        else:
            dtype1, expr1 = builtin_type_cast(convtype, [arg])
            exprlist.append(expr1)
    return convtype, lcapi.builder().call(to_lctype(convtype), op, exprlist)


def check_exact_signature(signature, args, name):
    if len(signature) != len(args):
        raise TypeError(f"{name} takes exactly {len(signature)} arguments, {len(args)} given.")
    for idx in range(len(args)):
        if signature[idx] != args[idx].dtype:
            raise TypeError(f"{name} expects ({','.join([getattr(x,'__name__',None) or repr(x) for x in signature])}). Calling with ({','.join([getattr(x.dtype,'__name__',None) or repr(x.dtype) for x in args])})")


# return dtype, expr
def builtin_func(name, args):

    if name == "set_block_size":
        check_exact_signature([int,int,int], args, "set_block_size")
        for a in args:
            if type(a).__name__ != "Constant":
                raise TypeError("Because set_block_size is a compile-time instruction, arguments of set_block_size must be literal (constant).")
        lcapi.builder().set_block_size(*[a.value for a in args])
        return None, None

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
                   or (len(args) == N and check_types(lcapi.Type.from_(f"vector<float,{N}>"), args)) \
                   or (len(args) == N * N and check_types(float, args)), 'type check failed'
            op = getattr(lcapi.CallOp, name.upper())
            dtype = getattr(lcapi, f'float{N}x{N}')
            return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    # TODO: atan2

    # e.g. sin(x)
    if name in ('isinf', 'isnan', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cos', 'cosh',
                'sin', 'sinh', 'tan', 'tanh', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10',
                'sqrt', 'rsqrt', 'ceil', 'floor', 'fract', 'trunc', 'round'):
        # type check: arg must be float / float vector
        assert len(args) == 1
        assert args[0].dtype == float or \
               (to_lctype(args[0].dtype).is_vector() and to_lctype(args[0].dtype).element() == to_lctype(float))
        op = getattr(lcapi.CallOp, name.upper())
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    if name in ('abs',):
        assert len(args) == 1
        assert args[0].dtype in (int, float) or to_lctype(args[0].dtype).is_vector() and to_lctype(args[0].dtype).element() in (to_lctype(int), to_lctype(float))
        op = getattr(lcapi.CallOp, name.upper())
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    if name in ('copysign',):
        assert len(args) == 2
        return make_vector_call(float, lcapi.CallOp.COPYSIGN, args)

    if name in ('min', 'max'):
        assert len(args) == 2

        def element_type(_dtype):
            if _dtype in {int, float, bool}:
                return _dtype
            return from_lctype(to_lctype(_dtype).element())
        op = getattr(lcapi.CallOp, name.upper())
        return make_vector_call(element_type(args[0].dtype), op, args)

    if name in ('length',):
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

    if name == 'print':
        globalvars.printer.kernel_print(args)
        globalvars.current_kernel.uses_printer = True
        return None, None

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
        args[1].dtype, args[1].expr = builtin_type_cast(lcapi.uint2, [args[1]]) # convert int2 to uint2
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])
    if name == "texture2d_write":
        op = lcapi.CallOp.TEXTURE_WRITE
        dtype = getattr(lcapi, args[0].dtype.dtype.__name__ + "4")
        check_exact_signature([lcapi.int2, dtype], args[1:], "Texture2D.write")
        args[1].dtype, args[1].expr = builtin_type_cast(lcapi.uint2, [args[1]]) # convert int2 to uint2
        lcapi.builder().call(op, [x.expr for x in args])
        return None, None

    raise Exception(f'unrecognized function call {name}')



def callable_call(func, args):
    globalvars.current_kernel.uses_printer |= func.uses_printer
    check_exact_signature([x[1] for x in func.params], args, f"(callable){func.funcname}")
    # call
    if not hasattr(func, "return_type") or func.return_type == None:
        return None, lcapi.builder().call(func.func, [x.expr for x in args])
    else:
        dtype = func.return_type
        return dtype, lcapi.builder().call(to_lctype(dtype), func.func, [x.expr for x in args])
