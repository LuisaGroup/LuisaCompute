import lcapi
from . import globalvars

class Buffer:
    def __init__(self, size, dtype):
        if not dtype in {int, float, bool}:
            raise Exception('invalid buffer dtype')
        self.dtype = dtype
        self.size = size
        self.bytesize = size * lcapi.Type.from_(dtype.__name__).size()
        self.handle = globalvars.device.impl().create_buffer(self.bytesize)

    def async_copy_from(self, arr): # arr: numpy array
        assert arr.size * arr.itemsize == self.bytesize
        ulcmd = lcapi.BufferUploadCommand.create(self.handle, 0, self.bytesize, arr)
        globalvars.stream.add(ulcmd)

    def async_copy_to(self, arr): # arr: numpy array
        assert arr.size * arr.itemsize == self.bytesize
        dlcmd = lcapi.BufferDownloadCommand.create(self.handle, 0, self.bytesize, arr)
        globalvars.stream.add(dlcmd)

    def read(self, idx):
        raise Exception("Method can only be called in Luisa kernel / callable")

    def write(self, idx):
        raise Exception("Method can only be called in Luisa kernel / callable")
