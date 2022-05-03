import lcapi
from . import globalvars
from .globalvars import get_global_device
from .types import to_lctype, is_vector_type, BuiltinFuncEntry
from functools import cache

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

    def __eq__(self, other):
        return type(other) is BufferType and self.dtype == other.dtype
        
    @callable
    def read(self, idx):
        return _builtin_call(dtype, "BUFFER_READ", [self, idx])

    @callable
    def write(self, idx, value):
        _builtin_call(dtype, "BUFFER_WRITE", [self, idx, value])

