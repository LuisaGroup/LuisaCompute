from .dylibs import lcapi
from .types import BuiltinFuncBuilder
from .func import func
from .builtin import check_exact_signature


# because indexed access of buffer isn't officially supported (by astbuilder),
# here we provide the buffer access function for atomic operations

def _atomic_call(dtype, op_name, *args):
    op = getattr(lcapi.CallOp, op_name)
    chain = lcapi.AtomicAccessChain()
    chain.create(args[0].expr)
    chain.access(args[1].expr)
    return dtype, chain.operate(op, [x.expr for x in args[2:]])
# ======================= int buffer atomic operations ========================


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


@BuiltinFuncBuilder
def _atomic_access_call(n_dtype, n_op_name, buf, idx, member_nest_level, *args):
    """
    example:
        buf = luisa.Buffer(10000, luisa.StructType(k=float, a=float4))
        # buf[i].a.w += k (atomic)
        _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", buf, idx, 2, 1, 3, k)
    """
    assert n_dtype.dtype is type
    assert n_op_name.dtype is str
    assert type(buf.dtype).__name__ == "BufferType"
    assert idx.dtype is int
    dtype = n_dtype.expr
    op_name = n_op_name.expr
    op = getattr(lcapi.CallOp, op_name)
    chain = lcapi.AtomicAccessChain()
    chain.create(buf.expr)
    chain.access(idx.expr)
    assert type(member_nest_level).__name__ == "Constant"
    for l in range(member_nest_level.value):
        assert type(args[l]).__name__ == "Constant"
        chain.member(args[l].value)
    return dtype, chain.operate(op, [x.expr for x in args[member_nest_level.value:]])
    

__all__ = ["int_atomic_functions", "float_atomic_functions", "_atomic_access_call"]
