from .types import element_of, length_of, scalar_dtypes
from .builtin_type_check import TC
from functools import reduce


def no_param(*args) -> bool:
    return len(args) == 0

def unary(*args) -> bool:
    return len(args) == 1

def binary(*args) -> bool:
    return len(args) == 2

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


def length_eq(total_lengths):
    def check(*args) -> bool:
        return sum(map(lambda x: length(x.dtype), args)) in total_lengths
    return check

def length_leq(total_length):
    def check(*args) -> bool:
        return sum(map(lambda x: length(x.dtype), args)) <= total_length
    return check

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
