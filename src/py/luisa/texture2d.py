import lcapi
from . import globalvars
from .globalvars import get_global_device
from .types import dtype_of
from functools import cache
from .func import func
from .builtin import _builtin_call
from .mathtypes import *


class Texture2D:
    def __init__(self, width, height, channel, dtype, storage = None):
        if not dtype in {int, float}:
            raise Exception('Texture2D only supports int / float')
        if not channel in (1,2,4):
            raise Exception('Texture2D can only have 1/2/4 channels')
        self.width = width
        self.height = height
        self.channel = channel
        self.dtype = dtype
        self.vectype = getattr(lcapi, dtype.__name__ + str(channel))
        # default storage type: max precision
        if storage is None:
            storage = getattr(lcapi.PixelStorage, dtype.__name__.upper() + str(channel))
        self.storage = storage
        if lcapi.pixel_storage_channel_count(storage) != channel:
            raise TypeError("pixel storage inconsistent with channel count")
        self.format = getattr(lcapi, "pixel_storage_to_format_" + dtype.__name__)(storage)

        self.bytesize = lcapi.pixel_storage_size(storage) * width * height;
        self.texture2DType = Texture2DType(dtype, channel)
        self.read = self.texture2DType.read
        self.write = self.texture2DType.write
        # instantiate texture on device
        self.handle = get_global_device().impl().create_texture(self.format, 2, width, height, 1, 1)

    @staticmethod
    def empty(width, height, channel, dtype, storage = None):
        return Texture2D(width, height, channel, dtype, storage)

    @staticmethod
    @cache
    def get_fill_kernel(value):
        @func
        def fill(tex):
            tex.write(dispatch_id().xy, value)
        return fill

    @staticmethod
    def zeros(width, height, channel, dtype, storage = None):
        tex = Texture2D.empty(width, height, channel, dtype, storage)
        val = tex.vectype(0)
        Texture2D.get_fill_kernel(val)(tex, dispatch_size=(width,height,1))
        return tex

    @staticmethod
    def ones(width, height, channel, dtype, storage = None):
        tex = Texture2D.empty(width, height, channel, dtype, storage)
        val = tex.vectype(1)
        Texture2D.get_fill_kernel(val)(tex, dispatch_size=(width,height,1))
        return tex

    @staticmethod
    def filled(width, height, channel, dtype, val, storage = None): # TODO deduce dtype
        tex = Texture2D.empty(width, height, channel, dtype, storage)
        assert dtype_of(val) == tex.vectype
        Texture2D.get_fill_kernel(val)(tex, dispatch_size=(width,height,1))
        return tex

    @staticmethod
    def from_array(arr):
        # TODO deduce dtype & storage
        assert len(arr.shape) == 3 and arr.shape[0]>0 and arr.shape[1]>0 and arr.shape[2] in (1,2,4)
        tex = Texture2D.empty(arr.shape[0], arr.shape[1], arr.shape[2], dtype_of(arr[0][0][0].item()))
        tex.copy_from_array(arr)
        return tex

    def copy_from(self, arr, sync = False, stream = None):
        return copy_from_array(self, arr, sync, stream)

    def copy_from_array(self, arr, sync = False, stream = None): # arr: numpy array
        if stream is None:
            stream = globalvars.stream
        assert arr.size * arr.itemsize == self.bytesize
        ulcmd = lcapi.TextureUploadCommand.create(self.handle, self.storage, 0, lcapi.uint3(self.width,self.height,1), arr)
        stream.add(ulcmd)
        if sync:
            stream.synchronize()

    def copy_to(self, arr, sync = True, stream = None): # arr: numpy array
        if stream is None:
            stream = globalvars.stream
        assert arr.size * arr.itemsize == self.bytesize
        dlcmd = lcapi.TextureDownloadCommand.create(self.handle, self.storage, 0, lcapi.uint3(self.width,self.height,1), arr)
        stream.add(dlcmd)
        if sync:
            stream.synchronize()


def texture2d(arr):
    if type(arr).__name__ == "ndarray":
        return Texture2D.from_array(arr)
    else:
        raise TypeError(f"Texture2D from unrecognized type: {type(arr)}")


class Texture2DType:
    def __init__(self, dtype, channel):
        self.dtype = dtype
        self.channel = channel
        self.vectype = getattr(lcapi, dtype.__name__ + str(channel))
        self.luisa_type = lcapi.Type.from_("texture<2," + dtype.__name__ + ">")
        self.read = self.get_read_method(self.vectype)
        self.write = self.get_write_method(self.vectype)

    def __eq__(self, other):
        return type(other) is Texture2DType and self.dtype == other.dtype and self.channel == other.channel

    def __hash__(self):
        return hash(self.dtype) ^ hash(self.channel) ^ 127858794396757894

    @staticmethod
    @cache
    def get_read_method(dtype):
        @func
        def read(self, coord: int2):
            return _builtin_call(dtype, "TEXTURE_READ", self, make_uint2(coord))
        return read
    
    @staticmethod
    @cache
    def get_write_method(dtype):
        @func
        def write(self, coord: int2, value: dtype):
            _builtin_call("TEXTURE_WRITE", self, make_uint2(coord), value)
        return write
