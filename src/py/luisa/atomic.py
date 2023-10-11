from .dylibs import lcapi
from .types import BuiltinFuncBuilder
from .func import func
from .builtin import check_exact_signature


# because indexed access of buffer isn't officially supported (by astbuilder),
# here we provide the buffer access function for atomic operations

# ======================= int buffer atomic operations ========================
def _atomic_call(*args):
    op = getattr(lcapi.CallOp, args[1])
    chain = lcapi.AtomicAccessChain()
    chain.create(args[2].expr)
    chain.access(args[3].expr)
    return args[0], chain.operate(op, [x.expr for x in args[4:]])


@BuiltinFuncBuilder
def atomic_exchange(*argnodes):#(self, idx: int, desired: int):
    ''' stores desired, returns old. '''
    check_exact_signature([int, int], argnodes[1:], "atomic_exchange")
    return _atomic_call(int, "ATOMIC_EXCHANGE", argnodes[0], argnodes[1], argnodes[2])


@BuiltinFuncBuilder
def atomic_compare_exchange(*argnodes):#(self, idx: int, expected: int, desired: int):
    ''' stores (old == expected ? desired : old), returns old. '''
    check_exact_signature([int, int, int], argnodes[1:], "atomic_compare_exchange")
    return _atomic_call(int, "ATOMIC_COMPARE_EXCHANGE", argnodes[0], argnodes[1], argnodes[2], argnodes[3])


@BuiltinFuncBuilder
def atomic_fetch_add(*argnodes):#(self, idx: int, val: int):
    ''' stores (old + val), returns old. '''
    check_exact_signature([int, int], argnodes[1:], "atomic_fetch_add")
    return _atomic_call(int, "ATOMIC_FETCH_ADD", argnodes[0], argnodes[1], argnodes[2])


@BuiltinFuncBuilder
def atomic_fetch_sub(*argnodes):#(self, idx: int, val: int):
    ''' stores (old - val), returns old. '''
    check_exact_signature([int, int], argnodes[1:], "atomic_fetch_sub")
    return _atomic_call(int, "ATOMIC_FETCH_SUB", argnodes[0], argnodes[1], argnodes[2])


@BuiltinFuncBuilder
def atomic_fetch_and(*argnodes):#(self, idx: int, val: int):
    ''' stores (old & val), returns old. '''
    check_exact_signature([int, int], argnodes[1:], "atomic_fetch_and")
    return _atomic_call(int, "ATOMIC_FETCH_AND", argnodes[0], argnodes[1], argnodes[2])


@BuiltinFuncBuilder
def atomic_fetch_or(*argnodes):#(self, idx: int, val: int):
    ''' stores (old | val), returns old. '''
    check_exact_signature([int, int], argnodes[1:], "atomic_fetch_or")
    return _atomic_call(int, "ATOMIC_FETCH_OR", argnodes[0], argnodes[1], argnodes[2])


@BuiltinFuncBuilder
def atomic_fetch_xor(*argnodes):#(self, idx: int, val: int):
    ''' stores (old ^ val), returns old. '''
    check_exact_signature([int, int], argnodes[1:], "atomic_fetch_xor")
    return _atomic_call(int, "ATOMIC_FETCH_XOR", argnodes[0], argnodes[1], argnodes[2])


@BuiltinFuncBuilder
def atomic_fetch_min(*argnodes):#(self, idx: int, val: int):
    ''' stores min(old, val), returns old. '''
    check_exact_signature([int, int], argnodes[1:], "atomic_fetch_min")
    return _atomic_call(int, "ATOMIC_FETCH_MIN", argnodes[0], argnodes[1], argnodes[2])


@BuiltinFuncBuilder
def atomic_fetch_max(*argnodes):#(self, idx: int, val: int):
    ''' stores max(old, val), returns old. '''
    check_exact_signature([int, int], argnodes[1:], "check_exact_signature")
    return _atomic_call(int, "ATOMIC_FETCH_MAX", argnodes[0], argnodes[1], argnodes[2])


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
@BuiltinFuncBuilder
def atomic_exchange(*argnodes):#(self, idx: int, desired: int):
    ''' stores desired, returns old. '''
    check_exact_signature([int, float], argnodes[1:], "atomic_exchange")
    return _atomic_call(float, "ATOMIC_EXCHANGE", argnodes[0], argnodes[1], argnodes[2])


@BuiltinFuncBuilder
def atomic_compare_exchange(*argnodes):#(self, idx: int, expected: int, desired: int):
    ''' stores (old == expected ? desired : old), returns old. '''
    check_exact_signature([int, float, float], argnodes[1:], "atomic_compare_exchange")
    return _atomic_call(float, "ATOMIC_COMPARE_EXCHANGE", argnodes[0], argnodes[1], argnodes[2], argnodes[3])


@BuiltinFuncBuilder
def atomic_fetch_add(*argnodes):#(self, idx: int, val: int):
    ''' stores (old + val), returns old. '''
    check_exact_signature([int, float], argnodes[1:], "atomic_fetch_add")
    return _atomic_call(float, "ATOMIC_FETCH_ADD", argnodes[0], argnodes[1], argnodes[2])


@BuiltinFuncBuilder
def atomic_fetch_sub(*argnodes):#(self, idx: int, val: int):
    ''' stores (old - val), returns old. '''
    check_exact_signature([int, float], argnodes[1:], "atomic_fetch_sub")
    return _atomic_call(float, "ATOMIC_FETCH_SUB", argnodes[0], argnodes[1], argnodes[2])


@BuiltinFuncBuilder
def atomic_fetch_min(*argnodes):#(self, idx: int, val: int):
    ''' stores min(old, val), returns old. '''
    check_exact_signature([int, float], argnodes[1:], "atomic_fetch_min")
    return _atomic_call(float, "ATOMIC_FETCH_MIN", argnodes[0], argnodes[1], argnodes[2])


@BuiltinFuncBuilder
def atomic_fetch_max(*argnodes):#(self, idx: int, val: int):
    ''' stores max(old, val), returns old. '''
    check_exact_signature([int, float], argnodes[1:], "check_exact_signature")
    return _atomic_call(float, "ATOMIC_FETCH_MAX", argnodes[0], argnodes[1], argnodes[2])



float_atomic_functions = [
    atomic_exchange,
    atomic_compare_exchange,
    atomic_fetch_add,
    atomic_fetch_sub,
    atomic_fetch_min,
    atomic_fetch_max
]

__all__ = ["int_atomic_functions", "float_atomic_functions"]
