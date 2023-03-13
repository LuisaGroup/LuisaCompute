from turtle import width
import lcapi
from . import globalvars
from .globalvars import get_global_device
from .types import dtype_of, length_of, element_of, vector
from functools import cache
from .func import func
from .builtin import _builtin_call
from .mathtypes import *
from .types import uint, uint3

def _check_storage(storage_name, dtype):
    compatible = { float: {'byte','short','half','float'}, int: {'byte','short','int','uint'}, uint: {'byte','short','int','uint'}}
    if storage_name.lower() not in compatible[dtype]:
        raise TypeError(f"{dtype} texture is only compatible with storage: {compatible[dtype]}")

class Texture3D:
    def __init__(self, width, height, volume, channel, dtype, mip=1, storage = None):
        if not dtype in {int, uint, float}:
            raise Exception('Texture3D only supports int / uint / float')
        if not channel in (1,3,4):
            raise Exception('Texture3D can only have 1/2/4 channels')
        self.width = width
        self.height = height
        self.channel = channel
        self.dtype = dtype
        self.mip = mip
        self.volume = volume
        self.vectype = dtype if channel == 1 else getattr(lcapi, dtype.__name__ + str(channel))
        # default storage type: max precision
        self.storage_name = storage.upper() if storage is not None else dtype.__name__.upper()
        _check_storage(self.storage_name, dtype)
        self.storage = getattr(lcapi.PixelStorage, self.storage_name + str(channel))
        self.format = getattr(lcapi, "pixel_storage_to_format_" + dtype.__name__)(self.storage)

        self.bytesize = lcapi.pixel_storage_size(self.storage, width, height, volume)
        self.texture3DType = Texture3DType(dtype, channel)
        self.read = self.texture3DType.read
        self.write = self.texture3DType.write
        # instantiate texture on device
        self.handle = get_global_device().impl().create_texture(self.format, 3, width, height, volume, mip)
    def __del__(self):
        if self.handle != None:
            device = get_global_device()
            if device != None:
                device.impl().destroy_texture(self.handle)    
    @staticmethod
    def texture3d(arr):
        if type(arr).__name__ == "ndarray":
            return Texture3D.from_array(arr)
        else:
            raise TypeError(f"Texture3D from unrecognized type: {type(arr)}")

    @staticmethod
    def empty(width, height, channel, dtype, storage = None):
        return Texture3D(width, height, channel, dtype, 1, storage)
    
    def copy_to_tex(self, tex, sync = False, stream = None):
        if stream is None:
            stream = globalvars.stream
        assert self.storage == tex.storage and self.width == tex.width and self.volume == tex.volume and self.height == tex.height
        cpcmd = lcapi.TextureCopyCommand.create(self.storage, self.handle, tex.handle, 0, 0, lcapi.uint3(self.width,self.height,self.volume))
        stream.add(cpcmd)
        if sync:
            stream.synchronize()

    def copy_from_tex(self, tex, sync = False, stream = None):
        if stream is None:
            stream = globalvars.stream
        assert self.storage == tex.storage and self.width == tex.width and self.volume == tex.volume and self.height == tex.height
        cpcmd = lcapi.TextureCopyCommand.create(self.storage, tex.handle, self.handle, 0, 0, lcapi.uint3(self.width,self.height,self.volume))
        stream.add(cpcmd)
        if sync:
            stream.synchronize()

    def copy_from(self, arr, sync = False, stream = None):
        if type(arr).__name__ == "ndarray":
            self.copy_from_array(arr, sync, stream)
        else:
            self.copy_from_tex(arr, sync, stream)

    def copy_from_array(self, arr, sync = False, stream = None): # arr: numpy array
        if stream is None:
            stream = globalvars.stream
        assert arr.size * arr.itemsize == self.bytesize
        ulcmd = lcapi.TextureUploadCommand.create(self.handle, self.storage, 0, lcapi.uint3(self.width,self.height,self.volume), arr)
        stream.add(ulcmd)
        if sync:
            stream.synchronize()

    def copy_to(self, arr, sync = True, stream = None): # arr: numpy array
        if stream is None:
            stream = globalvars.stream
        assert arr.size * arr.itemsize == self.bytesize
        dlcmd = lcapi.TextureDownloadCommand.create(self.handle, self.storage, 0, lcapi.uint3(self.width,self.height,self.volume), arr)
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

texture3d = Texture3D.texture3d


class Texture3DType:
    def __init__(self, dtype, channel):
        self.dtype = dtype
        self.channel = channel
        self.vectype = dtype if channel == 1 else getattr(lcapi, dtype.__name__ + str(channel))
        self.luisa_type = lcapi.Type.from_("texture<3," + dtype.__name__ + ">")
        self.read = self.get_read_method(self.vectype)
        self.write = self.get_write_method(self.vectype)

    def __eq__(self, other):
        return type(other) is Texture3DType and self.dtype == other.dtype and self.channel == other.channel

    def __hash__(self):
        return hash(self.dtype) ^ hash(self.channel) ^ 127858794396757894

    @staticmethod
    @cache
    def get_read_method(dtype):
        dtype4 = vector(element_of(dtype), 4)
        if length_of(dtype) == 4:
            @func
            def read(self, coord: uint3):
                return _builtin_call(dtype, "TEXTURE_READ", self, (coord))
            return read
        elif length_of(dtype) == 2:
            @func
            def read(self, coord: uint3):
                return _builtin_call(dtype4, "TEXTURE_READ", self, (coord)).xy
            return read
        elif length_of(dtype) == 1:
            @func
            def read(self, coord: uint3):
                return _builtin_call(dtype4, "TEXTURE_READ", self, (coord)).x
            return read
        else:
            assert False
    
    @staticmethod
    @cache
    def get_write_method(dtype):
        if length_of(dtype) == 4:
            @func
            def write(self, coord: uint3, value: dtype):
                _builtin_call("TEXTURE_WRITE", self, (coord), value)
            return write
        else:
            # convert to vector4
            dtype4 = vector(element_of(dtype), 4)
            opstr = "MAKE_" + dtype4.__name__.upper()
            if element_of(dtype) == uint:
                zero = 0
            else:
                zero = element_of(dtype)(0)
            if length_of(dtype) == 2:
                @func
                def write(self, coord: uint3, value: dtype):
                    tmp = _builtin_call(dtype4, opstr, value, zero, zero)
                    _builtin_call("TEXTURE_WRITE", self, (coord), tmp)
                return write
            if length_of(dtype) == 1:
                @func
                def write(self, coord: uint3, value: dtype):
                    tmp = _builtin_call(dtype4, opstr, value, zero, zero, zero)
                    _builtin_call("TEXTURE_WRITE", self, (coord), tmp)
                return write
            assert False

