from .types import element_of, length_of, scalar_dtypes
from .builtin_type_check import TC
from functools import reduce


def no_param(*args) -> bool:
    return len(args) == 0


def unary(*args) -> bool:
    return len(args) == 1


def binary(*args) -> bool:
    return len(args) == 2


def ternary(*args) -> bool:
    return len(args) == 3


def multi_param(*args) -> bool:
    return True


def all_arithmetic(*args) -> bool:
    return reduce(lambda x, y: x and y, map(lambda x: TC.is_arithmetic(inner_type(x.dtype)), args))


def all_integer(*args) -> bool:
    return reduce(lambda x, y: x and y, map(lambda x: TC.is_integer(inner_type(x.dtype)), args))


def all_bool(*args) -> bool:
    return reduce(lambda x, y: x and y, map(lambda x: TC.is_bool(inner_type(x.dtype)), args))


def all_float(*args) -> bool:
    return reduce(lambda x, y: x and y, map(lambda x: TC.is_float(inner_type(x.dtype)), args))


def with_inner_type(*inner_types):
    def with_inner_type(*args) -> bool:
        return reduce(lambda x, y: x and y, map(lambda x: inner_type(x[0].dtype) in x[1], zip(args, inner_types)))
    return with_inner_type


def length_eq(total_lengths):
    def length_eq(*args) -> bool:
        return sum(map(lambda x: length(x.dtype), args)) in total_lengths
    return length_eq


def length_leq(total_length):
    def length_leq(*args) -> bool:
        return sum(map(lambda x: length(x.dtype), args)) <= total_length
    return length_leq


def broadcast(*args) -> bool:
    return length(args[0].dtype) == length(args[1].dtype) or length(args[0].dtype) == 1 or length(args[1].dtype) == 1


def same_length(*args) -> bool:
    for i in range(len(args) - 1):
        if length(args[i].dtype) != length(args[i + 1].dtype):
            return False
    return True


def with_dim(dimension):
    def check(*args) -> bool:
        return length(args[0].dtype) == dimension and length(args[1].dtype) == dimension
    return check


def is_array(*args):
    from .array import ArrayType
    from .types import vector_dtypes, matrix_dtypes
    return type(args[0].dtype) is ArrayType or args[0].dtype in {*vector_dtypes, *matrix_dtypes}


def inner_type(dtype):
    if dtype in scalar_dtypes:
        return dtype
    else:
        return element_of(dtype)


def length(dtype):
    if dtype in scalar_dtypes:
        return 1
    else:
        return length_of(dtype)
