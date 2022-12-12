from . import lcapi
from . import globalvars
from .globalvars import get_global_device
from .types import to_lctype, basic_dtypes, dtype_of
from .types import vector_dtypes, matrix_dtypes, element_of, length_of
from functools import cache
from .func import func
from .builtin import _builtin_call
from .mathtypes import *
from .builtin import check_exact_signature
from types import SimpleNamespace
from .atomic import int_atomic_functions, float_atomic_functions

class Buffer:
    def __init__(self, size, dtype):
        if dtype not in basic_dtypes and type(dtype).__name__ not in {'StructType','ArrayType'}:
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
    def filled(size, val):
        buf = Buffer.empty(size, dtype)
        dtype = dtype_of(val)
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
        packed_bytes = bytearray()
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

    def numpy(self, stream = None): # only supports scalar
        import numpy as np
        npf = {int:np.int32, float:np.float32, bool:bool}[self.dtype]
        arr = np.empty(self.size, dtype=npf)
        self.copy_to(arr, sync=True, stream=stream)
        return arr

    def to_list(self, stream = None):
        packed_bytes = bytes(self.bytesize)
        dlcmd = lcapi.BufferDownloadCommand.create(self.handle, 0, self.bytesize, packed_bytes)
        if stream is None:
            stream = globalvars.stream
        stream.add(dlcmd)
        stream.synchronize()
        elsize = to_lctype(self.dtype).size()
        return [from_bytes(self.dtype, packed_bytes[elsize*i: elsize*(i+1)]) for i in range(self.size)]

buffer = Buffer.buffer


class BufferType:
    def __init__(self, dtype):
        self.dtype = dtype
        self.luisa_type = lcapi.Type.from_("buffer<" + to_lctype(dtype).description() + ">")
        self.read = self.get_read_method(self.dtype)
        self.write = self.get_write_method(self.dtype)
        # disable atomic operations if it's not an int buffer
        if dtype == int:
            for f in int_atomic_functions:
                setattr(self, f.__name__, f)
        if dtype == float:
            for f in float_atomic_functions:
                setattr(self, f.__name__, f)

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


def from_bytes(dtype, packed):
    import struct
    if dtype == int:
        return struct.unpack('i', packed)[0]
    if dtype == float:
        return struct.unpack('f', packed)[0]
    if dtype == bool:
        return struct.unpack('?', packed)[0]
    if dtype in vector_dtypes or dtype in matrix_dtypes:
        el = element_of(dtype)
        elsize = to_lctype(el).size()
        return dtype(*[from_bytes(el, packed[i*elsize: (i+1)*elsize]) for i in range(0, length_of(dtype))])
    if hasattr(dtype, 'membertype'): # struct
        values = []
        offset = 0
        for el in dtype.membertype:
            elsize = to_lctype(el).size()
            curr_align = to_lctype(el).alignment()
            offset = (offset + curr_align - 1) // curr_align * curr_align
            values.append(from_bytes(el, packed[offset: offset + elsize]))
            offset += elsize

        return dtype(**{name: values[dtype.idx_dict[name]] for name in dtype.idx_dict})
    if hasattr(dtype, 'size'): # array
        el = dtype.dtype
        elsize = to_lctype(el).size()
        return dtype([from_bytes(el, packed[i*elsize: (i+1)*elsize]) for i in range(0, dtype.size)])
    assert False
