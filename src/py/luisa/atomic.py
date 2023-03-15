import lcapi
from .types import to_lctype
from .types import BuiltinFuncBuilder
from .func import func
from .builtin import _builtin_call

# because indexed access of buffer isn't officially supported (by astbuilder),
# here we provide the buffer access function for atomic operations

# ======================= int buffer atomic operations ========================

@func
def atomic_exchange(self, idx: int, desired: int):
    ''' stores desired, returns old. '''
    return _builtin_call(int, "ATOMIC_EXCHANGE", self, idx, desired)

@func
def atomic_compare_exchange(self, idx: int, expected: int, desired: int):
    ''' stores (old == expected ? desired : old), returns old. '''
    return _builtin_call(int, "ATOMIC_COMPARE_EXCHANGE", self, idx, expected, desired)

@func
def atomic_fetch_add(self, idx: int, val: int):
    ''' stores (old + val), returns old. '''
    return _builtin_call(int, "ATOMIC_FETCH_ADD", self, idx, val)

@func
def atomic_fetch_sub(self, idx: int, val: int):
    ''' stores (old - val), returns old. '''
    return _builtin_call(int, "ATOMIC_FETCH_SUB", self, idx, val)

@func
def atomic_fetch_and(self, idx: int, val: int):
    ''' stores (old & val), returns old. '''
    return _builtin_call(int, "ATOMIC_FETCH_AND", self, idx, val)

@func
def atomic_fetch_or(self, idx: int, val: int):
    ''' stores (old | val), returns old. '''
    return _builtin_call(int, "ATOMIC_FETCH_OR", self, idx, val)

@func
def atomic_fetch_xor(self, idx: int, val: int):
    ''' stores (old ^ val), returns old. '''
    return _builtin_call(int, "ATOMIC_FETCH_XOR", self, idx, val)

@func
def atomic_fetch_min(self, idx: int, val: int):
    ''' stores min(old, val), returns old. '''
    return _builtin_call(int, "ATOMIC_FETCH_MIN", self, idx, val)

@func
def atomic_fetch_max(self, idx: int, val: int):
    ''' stores max(old, val), returns old. '''
    return _builtin_call(int, "ATOMIC_FETCH_MAX", self, idx, val)

int_atomic_functions = [
    atomic_exchange,
    atomic_compare_exchange,
    atomic_fetch_add,
    atomic_fetch_sub,
    atomic_fetch_and,
    atomic_fetch_or,
    atomic_fetch_xor,
    atomic_fetch_min,
    atomic_fetch_max
]

# ======================= float buffer atomic operations ========================

@func
def atomic_exchange(self, idx: int, desired: float):
    ''' stores desired, returns old. '''
    return _builtin_call(float, "ATOMIC_EXCHANGE", self, idx, desired)

@func
def atomic_compare_exchange(self, idx: int, expected: float, desired: float):
    ''' stores (old == expected ? desired : old), returns old. '''
    return _builtin_call(float, "ATOMIC_COMPARE_EXCHANGE", self, idx, expected, desired)

@func
def atomic_fetch_add(self, idx: int, val: float):
    ''' stores (old + val), returns old. '''
    return _builtin_call(float, "ATOMIC_FETCH_ADD", self, idx, val)

@func
def atomic_fetch_sub(self, idx: int, val: float):
    ''' stores (old - val), returns old. '''
    return _builtin_call(float, "ATOMIC_FETCH_SUB", self, idx, val)

@func
def atomic_fetch_and(self, idx: int, val: float):
    ''' stores (old & val), returns old. '''
    return _builtin_call(float, "ATOMIC_FETCH_AND", self, idx, val)

@func
def atomic_fetch_or(self, idx: int, val: float):
    ''' stores (old | val), returns old. '''
    return _builtin_call(float, "ATOMIC_FETCH_OR", self, idx, val)

@func
def atomic_fetch_xor(self, idx: int, val: float):
    ''' stores (old ^ val), returns old. '''
    return _builtin_call(float, "ATOMIC_FETCH_XOR", self, idx, val)

@func
def atomic_fetch_min(self, idx: int, val: float):
    ''' stores min(old, val), returns old. '''
    return _builtin_call(float, "ATOMIC_FETCH_MIN", self, idx, val)

@func
def atomic_fetch_max(self, idx: int, val: float):
    ''' stores max(old, val), returns old. '''
    return _builtin_call(float, "ATOMIC_FETCH_MAX", self, idx, val)

float_atomic_functions = [
    atomic_exchange,
    atomic_compare_exchange,
    atomic_fetch_add,
    atomic_fetch_sub,
    atomic_fetch_and,
    atomic_fetch_or,
    atomic_fetch_xor,
    atomic_fetch_min,
    atomic_fetch_max
]

__all__ = ["int_atomic_functions", "float_atomic_functions"]
