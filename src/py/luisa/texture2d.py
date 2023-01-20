from turtle import width
import lcapi
from . import globalvars
from .globalvars import get_global_device
from .types import dtype_of, length_of, element_of, vector
from functools import cache
from .func import func
from .builtin import _builtin_call
from .mathtypes import *
from .types import uint, uint2


def _check_storage(storage_name, dtype):
    compatible = { float: {'byte','short','half','float'}, int: {'byte','short','int','uint'}, uint: {'byte','short','int','uint'}}
    if storage_name.lower() not in compatible[dtype]:
        raise TypeError(f"{dtype} texture is only compatible with storage: {compatible[dtype]}")

class Texture2D:
    def __init__(self, width, height, channel, dtype, mip=1, storage = None):
        if not dtype in {int, uint, float}:
            raise Exception('Texture2D only supports int / uint / float')
        if not channel in (1,2,4):
            raise Exception('Texture2D can only have 1/2/4 channels')
        self.width = width
        self.height = height
        self.channel = channel
        self.dtype = dtype
        self.mip = mip
        self.vectype = dtype if channel == 1 else getattr(lcapi, dtype.__name__ + str(channel))
        # default storage type: max precision
        self.storage_name = storage.upper() if storage is not None else dtype.__name__.upper()
        _check_storage(self.storage_name, dtype)
        self.storage = getattr(lcapi.PixelStorage, self.storage_name + str(channel))
        self.format = getattr(lcapi, "pixel_storage_to_format_" + dtype.__name__)(self.storage)

        self.bytesize = lcapi.pixel_storage_size(self.storage) * width * height
        self.texture2DType = Texture2DType(dtype, channel)
        self.read = self.texture2DType.read
        self.write = self.texture2DType.write
        # instantiate texture on device
        self.handle = get_global_device().impl().create_texture(self.format, 2, width, height, 1, mip)
    def __del__(self):
        if self.handle != None:
            device = get_global_device()
            if device != None:
                device.impl().destroy_texture(self.handle)    
    @staticmethod
    def texture2d(arr):
        if type(arr).__name__ == "ndarray":
            return Texture2D.from_array(arr)
        else:
            raise TypeError(f"Texture2D from unrecognized type: {type(arr)}")

    @staticmethod
    def empty(width, height, channel, dtype, storage = None):
        return Texture2D(width, height, channel, dtype, 1, storage)

    @staticmethod
    def from_array(arr): # arr: numpy array
        # TODO deduce dtype & storage
        assert len(arr.shape) == 3 and arr.shape[0]>0 and arr.shape[1]>0 and arr.shape[2] in (1,2,4)
        tex = Texture2D.empty(arr.shape[0], arr.shape[1], arr.shape[2], dtype_of(arr[0][0][0].item()))
        tex.copy_from_array(arr)
        return tex

    # use manually load temporarily

    @staticmethod
    def from_hdr_image(path:str):
        # load 32-bit 4-channel image from file
        import numpy as np
        arr = lcapi.load_hdr_image(path)
        assert len(arr.shape) == 3 and arr.shape[2] == 4
        assert arr.dtype == np.float32
        # save as SRGB 8-bit texture
        tex = Texture2D.empty(arr.shape[0], arr.shape[1], 4, float)
        tex.copy_from_array(arr)
        return tex

    @staticmethod
    def from_ldr_image(path:str):
        # load 8-bit 4-channel image from file
        import numpy as np
        arr = lcapi.load_ldr_image(path)
        assert len(arr.shape) == 3 and arr.shape[2] == 4
        assert arr.dtype == np.ubyte
        # save as SRGB 8-bit texture
        tex = Texture2D.empty(arr.shape[0], arr.shape[1], 4, float, storage="byte")
        tex.copy_from_array(arr)
        return tex

    def to_image(self, path:str):
        import numpy as np
        if self.format == lcapi.PixelFormat.RGBA32F:
            arr = np.empty([self.width, self.height, 4], dtype=np.float32)
            self.copy_to(arr)
            lcapi.save_hdr_image(path, arr, self.width, self.height)
            del arr
        elif self.format == lcapi.PixelFormat.RGBA8UNorm:
            arr = np.empty([self.width, self.height, 4], dtype=np.ubyte)
            self.copy_to(arr)
            lcapi.save_ldr_image(path, arr, self.width, self.height)
            del(arr)
        else:
            raise "Illegal export image format!"

    def copy_from(self, arr, sync = False, stream = None):
        return self.copy_from_array(arr, sync, stream)

    def copy_from_array(self, arr, sync = False, stream = None): # arr: numpy array
        if stream is None:
            stream = globalvars.stream
        assert arr.size * arr.itemsize == self.bytesize
        ulcmd = lcapi.TextureUploadCommand.create(self.handle, self.storage, 0, lcapi.uint3(self.width,self.height,1), arr)
        stream.add(ulcmd)
        # stream.add_upload_buffer(arr)
        if sync:
            stream.synchronize()

    def copy_to(self, arr, sync = True, stream = None): # arr: numpy array
        if stream is None:
            stream = globalvars.stream
        assert arr.size * arr.itemsize == self.bytesize
        dlcmd = lcapi.TextureDownloadCommand.create(self.handle, self.storage, 0, lcapi.uint3(self.width,self.height,1), arr)
        stream.add(dlcmd)
        # stream.add_readback_buffer(arr)
        if sync:
            stream.synchronize()

    def numpy(self):
        import numpy as np
        if self.dtype == float:
            npf = {'BYTE': np.uint8, 'SHORT': np.uint16, 'HALF': np.half, 'FLOAT': np.float32}[self.storage_name]
        else:
            npf = {'BYTE': np.int8, 'SHORT': np.int16, 'INT': np.int32}[self.storage_name]
        arr = np.empty((self.height, self.width, self.channel), dtype=npf)
        self.copy_to(arr, sync=True)
        return arr

texture2d = Texture2D.texture2d


class Texture2DType:
    def __init__(self, dtype, channel):
        self.dtype = dtype
        self.channel = channel
        self.vectype = dtype if channel == 1 else getattr(lcapi, dtype.__name__ + str(channel))
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
        dtype4 = vector(element_of(dtype), 4)
        if length_of(dtype) == 4:
            @func
            def read(self, coord: uint2):
                return _builtin_call(dtype, "TEXTURE_READ", self, (coord))
            return read
        elif length_of(dtype) == 2:
            @func
            def read(self, coord: uint2):
                return _builtin_call(dtype4, "TEXTURE_READ", self, (coord)).xy
            return read
        elif length_of(dtype) == 1:
            @func
            def read(self, coord: uint2):
                return _builtin_call(dtype4, "TEXTURE_READ", self, (coord)).x
            return read
        else:
            assert False
    @staticmethod
    @cache
    def get_write_method(dtype):
        if length_of(dtype) == 4:
            @func
            def write(self, coord: uint2, value: dtype):
                _builtin_call("TEXTURE_WRITE", self, (coord), value)
            return write
        else:
            # convert to vector4
            dtype4 = vector(element_of(dtype), 4)
            opstr = "MAKE_" + dtype4.__name__.upper()
            zero = element_of(dtype)(0)
            if length_of(dtype) == 2:
                @func
                def write(self, coord: uint2, value: dtype):
                    tmp = _builtin_call(dtype4, opstr, value, zero, zero)
                    _builtin_call("TEXTURE_WRITE", self, (coord), tmp)
                return write
            if length_of(dtype) == 1:
                @func
                def write(self, coord: uint2, value: dtype):
                    tmp = _builtin_call(dtype4, opstr, value, zero, zero, zero)
                    _builtin_call("TEXTURE_WRITE", self, (coord), tmp)
                return write
            assert False

