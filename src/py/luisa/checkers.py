from .types import element_of, length_of, scalar_dtypes
from .builtin_type_check import TC
from functools import reduce


def binary(*args) -> bool:
    return len(args) == 2


def all_arithmetic(*args) -> bool:
    return reduce(lambda x, y: x and y, map(lambda x: TC.is_arithmetic(inner_type(x.dtype)), args))


def all_integer(*args) -> bool:
    return reduce(lambda x, y: x and y, map(lambda x: TC.is_integer(inner_type(x.dtype)), args))


def all_bool(*args) -> bool:
    return reduce(lambda x, y: x and y, map(lambda x: TC.is_bool(inner_type(x.dtype)), args))


def broadcast_binary(*args) -> bool:
    return length(args[0].dtype) == length(args[1].dtype) or length(args[0].dtype) == 1 or length(args[1].dtype) == 1


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
