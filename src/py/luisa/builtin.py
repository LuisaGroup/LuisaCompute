# from .types import types
import lcapi

def deduce_unary_type(op, dtype):
    # TODO: Type check
    return dtype
    
def deduce_binary_type(op, dtype1, dtype2):
    # TODO: Type check
    # TODO: upcast
    return dtype1


def builtin_func(name, args):
    # e.g. dispatch_id()
    for func in 'thread_id', 'block_id', 'dispatch_id', 'dispatch_size':
        if name == func:
            assert len(args) == 0
            return lcapi.Type.from_("vector<uint,3>"), getattr(lcapi.builder(), func)()

    # e.g. make_float4(...)
    for T in 'uint','int','float','bool':
        for N in 2,3,4:
            if name == f'make_{T}{N}':
                # TODO: check args
                op = getattr(lcapi.CallOp, name.upper())
                rettype = lcapi.Type.from_(f'vector<{T},{N}>')
                return rettype, lcapi.builder().call(rettype, op, [x.ptr for x in args])

    # e.g. make_float2x2(...)
    for N in 2,3,4:
        if name == f'make_float{N}x{N}':
            # TODO: check args
            op = getattr(lcapi.CallOp, name.upper())
            # NOTE: OP only supports from vectors;
            # TODO: from scalar / matrix
            rettype = lcapi.Type.from_(f'matrix<{N}>')
            return rettype, lcapi.builder().call(rettype, op, [x.ptr for x in args])

    # TODO: atan2

    # e.g. sin(x)
    if name in ('isinf', 'isnan', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cos', 'cosh',
                'sin', 'sinh', 'tan', 'tanh', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10',
                'sqrt', 'rsqrt', 'ceil', 'floor', 'fract', 'trunc', 'round'):
        # type check: arg must be float / float vector
        assert len(args) == 1
        assert args[0].dtype == lcapi.Type.from_('float') or args[0].dtype.is_vector() and args[0].dtype.element() == lcapi.Type.from_('float')
        op = getattr(lcapi.CallOp, name.upper())
        rettype = args[0].type
        return rettype, lcapi.builder().call(rettype, op, [x.ptr for x in args])

    raise Exception('unrecognized function call')

