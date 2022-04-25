import lcapi
from .types import to_lctype, basic_type_dict, dtype_of
from functools import reduce
from . import globalvars


def deduce_unary_type(op, dtype):
    # TODO: Type check
    return dtype


def deduce_binary_type(op, dtype1, dtype2):
    # TODO: Type check
    # TODO: upcast
    return dtype1


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


def check_arg_length(length):
    return lambda arguments: sum(map(get_length, arguments)) == length


def implicit_coersion(dtype0, dtype1):
    if float in [dtype0, dtype1]:
        return float
    elif int in [dtype0, dtype1]:
        return int
    else:
        return int  # 怎么表示一个 uint？


def deduce_broadcast(dtype0, dtype1):
    if dtype0 in [float, int, int, bool]:  # 同样的问题，uint 如何表示为 dtype？
        return dtype1  # Broadcast
    elif dtype1 in [float, int, int, bool]:
        return dtype0  # Broadcast
    else:
        return dtype1  # same size || Matrix * Vector -> Vector


def to_bool(dtype):
    if dtype in [float, int, int, bool]:
        return bool
    elif dtype in [lcapi.float2, lcapi.int2, lcapi.uint2, lcapi.bool2]:
        return lcapi.bool2
    elif dtype in [lcapi.float3, lcapi.int3, lcapi.uint3, lcapi.bool3]:
        return lcapi.bool3
    elif dtype in [lcapi.float4, lcapi.int4, lcapi.uint4, lcapi.bool4]:
        return lcapi.bool4


def builtin_bin_op(op, lhs, rhs):
    assert op in (
        lcapi.BinaryOp.ADD,
        lcapi.BinaryOp.SUB,
        lcapi.BinaryOp.MUL,
        lcapi.BinaryOp.DIV,
        lcapi.BinaryOp.MOD,
        lcapi.BinaryOp.SHL,
        lcapi.BinaryOp.SHR,
        lcapi.BinaryOp.BIT_OR,
        lcapi.BinaryOp.BIT_XOR,
        lcapi.BinaryOp.BIT_AND,
        lcapi.BinaryOp.GREATER,
        lcapi.BinaryOp.LESS,
        lcapi.BinaryOp.GREATER_EQUAL,
        lcapi.BinaryOp.LESS_EQUAL,
        lcapi.BinaryOp.EQUAL,
        lcapi.BinaryOp.NOT_EQUAL
    ), 'Illegal op'
    dtype0, dtype1 = lhs.dtype, rhs.dtype
    length0, length1 = get_length(lhs), get_length(rhs)
    if op != lcapi.BinaryOp.MUL:
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

    if op in (lcapi.BinaryOp.MOD, lcapi.BinaryOp.BIT_AND, lcapi.BinaryOp.BIT_OR, lcapi.BinaryOp.BIT_XOR,
              lcapi.BinaryOp.SHL, lcapi.BinaryOp.SHR):
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
    elif op in (lcapi.BinaryOp.AND, lcapi.BinaryOp.OR):
        assert check_inner_types(to_lctype(bool), [lhs, rhs]), f'operator `{op}` only supports `bool` type.'
        dtype = deduce_broadcast(dtype0, dtype1)
        # add / sub / div: int, uint and float allowed
        # relational: int, uint and float allowed
    elif op in (
        lcapi.BinaryOp.ADD, lcapi.BinaryOp.SUB, lcapi.BinaryOp.DIV, lcapi.BinaryOp.MUL,
        lcapi.BinaryOp.LESS, lcapi.BinaryOp.GREATER, lcapi.BinaryOp.LESS_EQUAL, lcapi.BinaryOp.GREATER_EQUAL,
        lcapi.BinaryOp.EQUAL, lcapi.BinaryOp.NOT_EQUAL
    ):
        is_compare = op in (
            lcapi.BinaryOp.LESS, lcapi.BinaryOp.GREATER, lcapi.BinaryOp.LESS_EQUAL, lcapi.BinaryOp.GREATER_EQUAL,
            lcapi.BinaryOp.EQUAL, lcapi.BinaryOp.NOT_EQUAL
        )
        inner_lc_type_0 = get_inner_type(to_lctype(lhs.dtype))
        assert inner_lc_type_0 in [basic_type_dict[int], basic_type_dict[float], lcapi.Type.from_('uint')], \
            f'operator `{op}` only supports `int`, `uint` and `float` types.'
        if scalar_operation:
            # allow implicit type conversion
            # so check arg[1]'s type, ensure it also satisfies the constraints.
            inner_lc_type_1 = get_inner_type(to_lctype(rhs.dtype))
            assert inner_lc_type_1 in [basic_type_dict[int], basic_type_dict[float], lcapi.Type.from_('uint')], \
                f'operator `{op}` only supports `int`, `uint` and `float` types.'
            dtype = implicit_coersion(dtype0, dtype1)
        else:
            # forbid implicit type conversion
            # so check arg[1]'s type, ensure it is the same with arg[0]
            assert check_inner_type(inner_lc_type_0)(rhs), \
                'operation between vectors of different types not supported.'
            dtype = deduce_broadcast(dtype0, dtype1)
        if is_compare:
            dtype = to_bool(dtype)
    return dtype, lcapi.builder().binary(to_lctype(dtype), op, lhs.expr, rhs.expr)


def builtin_func(name, args):
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
                lc_type = lcapi.Type.from_(T)
                assert check_arg_length(N)(args), 'Check arg length failed'
                assert check_inner_types(lc_type, args), 'check arg type failed'
                # TODO: check args
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

    if name == 'print':
        globalvars.printer.kernel_print(args)
        return None, None

    bin_ops = {
        'add': '+',
        'sub': '-',
        'mul': '*',
        'div': '/',
        'mod': '%',
        'bit_and': '&',
        'bit_or': '|',
        'bit_xor': '^',
        'shl': '<<',
        'shr': '>>',
        'and': '&&',
        'or': '||',
        'less': '<',
        'greater': '>',
        'less_equal': '<=',
        'greater_equal': '>=',
        'equal': '==',
        'not_equal': '!='
    }

    raise Exception('unrecognized function call')
