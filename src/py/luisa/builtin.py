import lcapi
from .types import to_lctype, is_vector_type
from functools import reduce
from . import globalvars



def deduce_unary_type(op, dtype):
    # TODO: Type check
    return dtype


def deduce_binary_type(op, dtype1, dtype2):
    # TODO: Type check
    # TODO: upcast
    if dtype1 == dtype2:
        return dtype1
    if dtype1 == int and dtype2 == float:
        return float
    if dtype1 == float and dtype2 == int:
        return float
    if is_vector_type(dtype1) and dtype2 in {int,float,bool}:
        if to_lctype(dtype1).element() == to_lctype(dtype2):
            return dtype1
    if is_vector_type(dtype2) and dtype1 in {int,float,bool}:
        if to_lctype(dtype2).element() == to_lctype(dtype1):
            return dtype2
    raise NotImplementedError(f"deduce_binary_type not implemented for {dtype1}, {dtype2}")


# def get_length(arg) -> int:
#     type_ = arg.dtype
#     if type_.is_scalar():
#         return 1
#     elif type_.is_array() or type_.is_vector() or type_.is_matrix() or type_.is_texture():
#         return type_.dimension()
#     else:
#         assert False


# def check_type(type_):
#     def check(argument):
#         return argument.dtype == basic_types[type_]

#     return check


# def check_types(type_):
#     return lambda arguments: reduce(lambda x, y: x and y, map(check_type(type_), arguments))


# def check_inner_type(type_):
#     def check(argument):
#         type__ = argument.dtype
#         if type__.is_array() or type__.is_vector() or type__.is_matrix() or type__.is_texture():
#             return argument.dtype.element() == type_
#         else:
#             return argument.dtype == type_

#     return check


# def check_inner_types(type_):
#     return lambda arguments: reduce(lambda x, y: x and y, map(check_inner_type(type_), arguments))


# def check_type_in(types_):
#     lc_types = [basic_types[type_] for type_ in types_]

#     def check(argument):
#         return argument.dtype in lc_types

#     return check


# def check_arg_length(length):
#     return lambda arguments: sum(map(get_length, arguments)) == length

builtin_func_names = {
    'thread_id', 'block_id', 'dispatch_id', 'dispatch_size',
    'make_uint2', 'make_int2', 'make_float2', 'make_bool2',
    'make_uint3', 'make_int3', 'make_float3', 'make_bool3',
    'make_uint4', 'make_int4', 'make_float4', 'make_bool4',
    'make_float2x2', 'make_float3x3', 'make_float4x4',
    'isinf', 'isnan', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cos', 'cosh',
    'sin', 'sinh', 'tan', 'tanh', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10',
    'sqrt', 'rsqrt', 'ceil', 'floor', 'fract', 'trunc', 'round',
    'abs', 'copysign',
    'dot',
    'cross',
    'length', 
    'normalize',
    'lerp',
    'print'
}


# type cast or initialization
# return dtype, expr
def builtin_type_cast(dtype, args):
    if len(args) == 0:
        # construct variable without initialization
        return dtype, lcapi.builder().local(to_lctype(dtype))
    if dtype in {int, float, bool}:
        assert len(args)==1 and args[0].dtype in {int, float, bool}
        return dtype, lcapi.builder().cast(to_lctype(dtype), lcapi.CastOp.STATIC, args[0].expr)
    lctype = to_lctype(dtype)
    if lctype.is_basic():
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
        assert arg.dtype == dtype or is_vector_type(arg.dtype) and to_lctype(arg.dtype).element() == to_lctype(dtype)
        if is_vector_type(arg.dtype):
            if dim != 1:
                assert dim == to_lctype(arg.dtype).dimension()
            else:
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
    


# return dtype, expr
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
                # type_ = lcapi.Type.from_(T)
                # if not check_arg_length(N)(args):
                #     print('Check arg length failed')
                #     raise FileExistsError
                # if not check_inner_types(type_)(args):
                #     print('check arg type failed')
                #     raise FileNotFoundError
                # assert check_arg_length(N)(args), 'Check arg length failed'
                # assert check_inner_types(type_)(args), 'check arg type failed'
                # TODO: check args
                op = getattr(lcapi.CallOp, name.upper())
                dtype = getattr(lcapi, f'{T}{N}')
                return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    # e.g. make_float2x2(...)
    for N in 2, 3, 4:
        if name == f'make_float{N}x{N}':
            # assert (len(args) == 1 and check_type_in([float, lcapi.float2x2, lcapi.float3x3, lcapi.float4x4])(args[0]))\
            #     or (len(args) == N and check_types(lcapi.Type.from_(f"vector<float,{N}>"))(args)) \
            #     or (len(args) == N * N and check_types(float)(args)), 'type check failed'
            # if not checked:
            #     raise FileNotFoundError
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
        assert args[0].dtype == float or to_lctype(args[0].dtype).is_vector() and to_lctype(args[0].dtype).element() == to_lctype(float)
        op = getattr(lcapi.CallOp, name.upper())
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    if name in ('abs'):
        assert len(args) == 1
        assert args[0].dtype in (int, float) or to_lctype(args[0].dtype).is_vector() and to_lctype(args[0].dtype).element() in (to_lctype(int), to_lctype(float))
        op = getattr(lcapi.CallOp, name.upper())
        dtype = args[0].dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    if name in ('copysign'):
        assert len(args) == 2
        return make_vector_call(float, lcapi.CallOp.COPYSIGN, args)


    if name in ('length'):
        assert len(args) == 1
        assert to_lctype(args[0].dtype).is_vector() and to_lctype(args[0].dtype).element() == to_lctype(float)
        op = getattr(lcapi.CallOp, name.upper())
        return float, lcapi.builder().call(to_lctype(float), op, [x.expr for x in args])

    if name in ('normalize'):
        assert len(args) == 1
        assert to_lctype(args[0].dtype).is_vector() and to_lctype(args[0].dtype).element() == to_lctype(float)
        op = getattr(lcapi.CallOp, name.upper())
        return args[0].dtype, lcapi.builder().call(to_lctype(args[0].dtype), op, [x.expr for x in args])

    if name in ('dot'):
        assert len(args) == 2
        assert to_lctype(args[0].dtype).is_vector() and to_lctype(args[0].dtype).element() == to_lctype(float)
        assert to_lctype(args[1].dtype).is_vector() and to_lctype(args[1].dtype).element() == to_lctype(float)
        assert to_lctype(args[0].dtype).dimension() == to_lctype(args[1].dtype).dimension()
        op = getattr(lcapi.CallOp, name.upper())
        return float, lcapi.builder().call(to_lctype(float), op, [x.expr for x in args])

    if name in ('cross'):
        assert len(args) == 2
        assert args[0].dtype == lcapi.float3
        assert args[1].dtype == lcapi.float3
        op = getattr(lcapi.CallOp, name.upper())
        return lcapi.float3, lcapi.builder().call(to_lctype(lcapi.float3), op, [x.expr for x in args])
        
    if name in ('lerp'):
        assert len(args) == 3
        return make_vector_call(float, lcapi.CallOp.LERP, args)

    if name == 'print':
        globalvars.printer.kernel_print(args)
        return None, None


    if name == "buffer_read":
        builtin_op = lcapi.CallOp.BUFFER_READ
        dtype = args[0].dtype.dtype
        return dtype, lcapi.builder().call(to_lctype(dtype), builtin_op, [x.expr for x in args])
    if name == "buffer_write":
        builtin_op = lcapi.CallOp.BUFFER_WRITE
        lcapi.builder().call(builtin_op, [x.expr for x in args])
        return None, None





    raise Exception(f'unrecognized function call {name}')


