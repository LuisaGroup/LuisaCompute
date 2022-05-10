import lcapi
from . import globalvars
from .globalvars import get_global_device
from .types import to_lctype, basic_dtypes, dtype_of
from functools import cache
from .func import func
from .builtin import _builtin_call
from .mathtypes import *
from .types import BuiltinFuncBuilder
from .builtin import check_exact_signature
from types import SimpleNamespace

class Buffer:
    def __init__(self, size, dtype):
        if not (dtype in basic_dtypes or hasattr(dtype, 'to_bytes')):
            raise TypeError('Invalid buffer element type')
        self.bufferType = BufferType(dtype)
        self.read = self.bufferType.read
        self.write = self.bufferType.write
        self.dtype = dtype
        self.size = size
        self.bytesize = size * to_lctype(self.dtype).size()
        # instantiate buffer on device
        self.handle = get_global_device().impl().create_buffer(self.bytesize)

    @staticmethod
    def buffer(arr):
        if type(arr).__name__ == "ndarray":
            return Buffer.from_array(arr)
        elif type(arr) == list:
            return Buffer.from_list(arr)
        else:
            raise TypeError(f"buffer from unrecognized type: {type(arr)}")

    @staticmethod
    def empty(size, dtype):
        return Buffer(size, dtype)

    @func
    def fill_kernel(buf, value):
        buf.write(dispatch_id().x, value)

    @staticmethod
    def zeros(size, dtype):
        buf = Buffer.empty(size, dtype)
        try:
            val = dtype(0)
        except Exception:
            raise TypeError(f"can't deduce zero value of {dtype} type")
        Buffer.fill_kernel(buf, val, dispatch_size=(size,1,1))
        return buf

    @staticmethod
    def ones(size, dtype):
        buf = Buffer.empty(size, dtype)
        try:
            val = dtype(1)
        except Exception:
            raise TypeError(f"can't deduce zero value of {dtype} type")
        Buffer.fill_kernel(buf, val, dispatch_size=(size,1,1))
        return buf

    @staticmethod
    def filled(size, val, dtype=None):
        buf = Buffer.empty(size, dtype)
        if dtype is None:
            dtype = dtype_of(val)
        else:
            assert dtype_of(val) == dtype
        Buffer.fill_kernel(buf, val, dispatch_size=(size,1,1))
        return buf

    @staticmethod
    def from_list(arr):
        assert len(arr) > 0
        buf = Buffer(len(arr), dtype_of(arr[0]))
        buf.copy_from_list(arr)
        return buf

    @staticmethod
    def from_array(arr):
        assert len(arr) > 0
        buf = Buffer(len(arr), dtype_of(arr[0].item()))
        buf.copy_from_array(arr)
        return buf

    def copy_from_list(self, arr, sync = False, stream = None):
        if stream is None:
            stream = globalvars.stream
        assert len(arr) == self.size
        lctype = to_lctype(self.dtype)
        packed_bytes = b''
        for x in arr:
            assert dtype_of(x) == self.dtype
            if lctype.is_basic():
                packed_bytes += lcapi.to_bytes(x)
            else:
                packed_bytes += x.to_bytes()
        assert len(packed_bytes) == self.bytesize
        ulcmd = lcapi.BufferUploadCommand.create(self.handle, 0, self.bytesize, packed_bytes)
        stream.add(ulcmd)
        if sync:
            stream.synchronize()
        
    def copy_from_array(self, arr, sync = False, stream = None): # arr: numpy array or list
        if stream is None:
            stream = globalvars.stream
        # numpy array of same data layout
        assert arr.size * arr.itemsize == self.bytesize
        ulcmd = lcapi.BufferUploadCommand.create(self.handle, 0, self.bytesize, arr)
        stream.add(ulcmd)
        if sync:
            stream.synchronize()

    def copy_from(self, arr, sync = False, stream = None): # arr: numpy array or list
        if type(arr).__name__ == "ndarray":
            self.copy_from_array(arr, sync, stream)
        elif type(arr) == list:
            self.copy_from_list(arr, sync, stream)
        else:
            raise TypeError(f"copy from unrecognized type: {type(arr)}")

    def copy_to(self, arr, sync = True, stream = None): # arr: numpy array; user is resposible for data layout
        if stream is None:
            stream = globalvars.stream
        assert arr.size * arr.itemsize == self.bytesize
        dlcmd = lcapi.BufferDownloadCommand.create(self.handle, 0, self.bytesize, arr)
        stream.add(dlcmd)
        if sync:
            stream.synchronize()

    def numpy(self): # only supports scalar
        import numpy as np
        npf = {int:np.int32, float:np.float32, bool:bool}[self.dtype]
        arr = np.empty(self.size, dtype=npf)
        self.copy_to(arr, sync=True)
        return arr

buffer = Buffer.buffer


# because indexed access of buffer isn't officially supported (by astbuilder),
# here we provide the int buffer access function for atomic operations
@BuiltinFuncBuilder
def _iaccess(self, idx): # all arguments are AST nodes
    return int, lcapi.builder().access(to_lctype(int), self.expr, idx.expr)


class BufferType:
    def __init__(self, dtype):
        self.dtype = dtype
        self.luisa_type = lcapi.Type.from_("buffer<" + to_lctype(dtype).description() + ">")
        self.read = self.get_read_method(self.dtype)
        self.write = self.get_write_method(self.dtype)
        # disable atomic operations if it's not an int buffer
        if dtype != int:
            self.atomic_exchange = None
            self.atomic_compare_exchange = None
            self.atomic_fetch_add = None
            self.atomic_fetch_sub = None
            self.atomic_fetch_and = None
            self.atomic_fetch_or = None
            self.atomic_fetch_xor = None
            self.atomic_fetch_min = None
            self.atomic_fetch_max = None
            self.lock = None
            self.unlock = None

    def __eq__(self, other):
        return type(other) is BufferType and self.dtype == other.dtype

    def __hash__(self):
        return hash(self.dtype) ^ 8965828349193294

    @staticmethod
    @cache
    def get_read_method(dtype):
        @func
        def read(self, idx: int):
            return _builtin_call(dtype, "BUFFER_READ", self, idx)
        return read
    
    @staticmethod
    @cache
    def get_write_method(dtype):
        @func
        def write(self, idx: int, value: dtype):
            _builtin_call("BUFFER_WRITE", self, idx, value)
        return write


    # ========== atomic operations (int buffer only) ==========

    @func
    def atomic_exchange(self, idx: int, desired: int):
        ''' stores desired, returns old. '''
        return _builtin_call(int, "ATOMIC_EXCHANGE", _iaccess(self, idx), desired)

    @func
    def atomic_compare_exchange(self, idx: int, expected: int, desired: int):
        ''' stores (old == expected ? desired : old), returns old. '''
        return _builtin_call(int, "ATOMIC_COMPARE_EXCHANGE", _iaccess(self, idx), expected, desired)

    @func
    def atomic_fetch_add(self, idx: int, val: int):
        ''' stores (old + val), returns old. '''
        return _builtin_call(int, "ATOMIC_FETCH_ADD", _iaccess(self, idx), val)
 
    @func
    def atomic_fetch_sub(self, idx: int, val: int):
        ''' stores (old - val), returns old. '''
        return _builtin_call(int, "ATOMIC_FETCH_SUB", _iaccess(self, idx), val)

    @func
    def atomic_fetch_and(self, idx: int, val: int):
        ''' stores (old & val), returns old. '''
        return _builtin_call(int, "ATOMIC_FETCH_AND", _iaccess(self, idx), val)

    @func
    def atomic_fetch_or(self, idx: int, val: int):
        ''' stores (old | val), returns old. '''
        return _builtin_call(int, "ATOMIC_FETCH_OR", _iaccess(self, idx), val)

    @func
    def atomic_fetch_xor(self, idx: int, val: int):
        ''' stores (old ^ val), returns old. '''
        return _builtin_call(int, "ATOMIC_FETCH_XOR", _iaccess(self, idx), val)

    @func
    def atomic_fetch_min(self, idx: int, val: int):
        ''' stores min(old, val), returns old. '''
        return _builtin_call(int, "ATOMIC_FETCH_MIN", _iaccess(self, idx), val)

    @func
    def atomic_fetch_max(self, idx: int, val: int):
        ''' stores max(old, val), returns old. '''
        return _builtin_call(int, "ATOMIC_FETCH_MAX", _iaccess(self, idx), val)

    # ========== atomic utilities (int buffer only) ==========

    @func
    def lock(self, idx: int):
        ''' wait while the element is non-zero; then set to 1 '''
        while self.atomic_compare_exchange(idx, 0, 1) != 0:
            pass

    @func
    def unlock(self, idx: int):
        ''' set the element to 0 '''
        tmp = self.atomic_exchange(idx, 0)


