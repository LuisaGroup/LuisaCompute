import lcapi
from . import globalvars
from .globalvars import get_global_device
from .types import to_lctype, is_vector_type, BuiltinFuncEntry
from functools import cache
from . import callable

class Buffer:
    def __init__(self, size, dtype):
        if not (dtype in {int, float, bool} or is_vector_type(dtype)):
            raise Exception('buffer only supports scalar / vector yet')
        self.bufferType = BufferType(dtype)
        self.dtype = dtype
        self.size = size
        self.bytesize = size * to_lctype(self.dtype).size()
        # instantiate buffer on device
        self.handle = get_global_device().impl().create_buffer(self.bytesize)

    def copy_from(self, arr, sync = False, stream = None): # arr: numpy array
        if stream is None:
            stream = globalvars.stream
        assert arr.size * arr.itemsize == self.bytesize
        ulcmd = lcapi.BufferUploadCommand.create(self.handle, 0, self.bytesize, arr)
        stream.add(ulcmd)
        if sync:
            stream.synchronize()

    def copy_to(self, arr, sync = True, stream = None): # arr: numpy array
        if stream is None:
            stream = globalvars.stream
        assert arr.size * arr.itemsize == self.bytesize
        dlcmd = lcapi.BufferDownloadCommand.create(self.handle, 0, self.bytesize, arr)
        stream.add(dlcmd)
        if sync:
            stream.synchronize()


class BufferType:
    def __init__(self, dtype):
        self.dtype = dtype
        self.luisa_type = lcapi.Type.from_("buffer<" + to_lctype(dtype).description() + ">")
        # self.write = self.get_write_method(self.dtype)
        # print("write:", self.write)

    def __eq__(self, other):
        return type(other) is BufferType and self.dtype == other.dtype

    def __hash__(self):
        return hash(self.dtype) ^ 8965828349193294

    @callable
    def read(self, idx):
        return _builtin_call(dtype, "BUFFER_READ", [self, idx])
    
    @staticmethod
    @cache
    def get_write_method(dtype):
        @callable
        def write(self, idx, value: dtype):
            _builtin_call(dtype, "BUFFER_WRITE", [self, idx, value])
        return write

