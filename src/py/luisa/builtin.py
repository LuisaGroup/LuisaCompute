from functools import reduce

import lcapi
from .types import basic_types


def deduce_unary_type(op, dtype):
    # TODO: Type check
    return dtype


def deduce_binary_type(op, dtype1, dtype2):
    # TODO: Type check
    # TODO: upcast
    return dtype1


def get_length(arg) -> int:
    type_ = arg.dtype
    if type_.is_scalar():
        return 1
    elif type_.is_array() or type_.is_vector() or type_.is_matrix() or type_.is_texture():
        return type_.dimension()
    else:
        assert False


def check_type(type_):
    def check(argument):
        return argument.dtype == basic_types[type_]

    return check


def check_types(type_):
    return lambda arguments: reduce(lambda x, y: x and y, map(check_type(type_), arguments))


def check_inner_type(type_):
    def check(argument):
        type__ = argument.dtype
        if type__.is_array() or type__.is_vector() or type__.is_matrix() or type__.is_texture():
            return argument.dtype.element() == type_
        else:
            return argument.dtype == type_

    return check


def check_inner_types(type_):
    return lambda arguments: reduce(lambda x, y: x and y, map(check_inner_type(type_), arguments))


def check_type_in(types_):
    lc_types = [basic_types[type_] for type_ in types_]

    def check(argument):
        return argument.dtype in lc_types

    return check


def check_arg_length(length):
    return lambda arguments: sum(map(get_length, arguments)) == length


def builtin_func(name, args):
    # e.g. dispatch_id()
    for func in 'thread_id', 'block_id', 'dispatch_id', 'dispatch_size':
        if name == func:
            assert len(args) == 0
            return lcapi.Type.from_("vector<uint,3>"), getattr(lcapi.builder(), func)()

    # e.g. make_float4(...)
    for T in 'uint', 'int', 'float', 'bool':
        for N in 2, 3, 4:
            if name == f'make_{T}{N}':
                type_ = lcapi.Type.from_(T)
                # if not check_arg_length(N)(args):
                #     print('Check arg length failed')
                #     raise FileExistsError
                # if not check_inner_types(type_)(args):
                #     print('check arg type failed')
                #     raise FileNotFoundError
                assert check_arg_length(N)(args), 'Check arg length failed'
                assert check_inner_types(type_)(args), 'check arg type failed'
                # TODO: check args
                op = getattr(lcapi.CallOp, name.upper())
                rettype = lcapi.Type.from_(f'vector<{T},{N}>')
                return rettype, lcapi.builder().call(rettype, op, [arg.expr for arg in args])

    # e.g. make_float2x2(...)
    for N in 2, 3, 4:
        if name == f'make_float{N}x{N}':
            assert (len(args) == 1 and check_type_in([float, lcapi.float2x2, lcapi.float3x3, lcapi.float4x4])(args[0]))\
                or (len(args) == N and check_types(lcapi.Type.from_(f"vector<float,{N}>"))(args)) \
                or (len(args) == N * N and check_types(float)(args)), 'type check failed'
            # if not checked:
            #     raise FileNotFoundError
            op = getattr(lcapi.CallOp, name.upper())
            rettype = lcapi.Type.from_(f'matrix<{N}>')
            return rettype, lcapi.builder().call(rettype, op, [arg.expr for arg in args])

    # TODO: atan2

    # e.g. sin(x)
    if name in ('isinf', 'isnan', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cos', 'cosh',
                'sin', 'sinh', 'tan', 'tanh', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10',
                'sqrt', 'rsqrt', 'ceil', 'floor', 'fract', 'trunc', 'round'):
        # type check: arg must be float / float vector
        assert len(args) == 1
        assert args[0].dtype == lcapi.Type.from_('float') or args[0].dtype.is_vector() and args[
            0].dtype.element() == lcapi.Type.from_('float')
        op = getattr(lcapi.CallOp, name.upper())
        rettype = args[0].type
        return rettype, lcapi.builder().call(rettype, op, [arg.expr for arg in args])

    raise Exception('unrecognized function call')
