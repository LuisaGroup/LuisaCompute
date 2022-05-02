import lcapi
from . import globalvars
from .globalvars import get_global_device


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
        # default storage type: max precision
        if storage is None:
            storage = getattr(lcapi.PixelStorage, dtype.__name__.upper() + str(channel))
        self.storage = storage
        if lcapi.pixel_storage_channel_count(storage) != channel:
            raise TypeError("pixel storage inconsistent with channel count")
        self.format = getattr(lcapi, "pixel_storage_to_format_" + dtype.__name__)(storage)

        self.bytesize = lcapi.pixel_storage_size(storage) * width * height;
        self.texture2DType = Texture2DType(dtype)
        # instantiate texture on device
        self.handle = get_global_device().impl().create_texture(self.format, 2, width, height, 1, 1)

    def copy_from(self, arr, sync = False, stream = None): # arr: numpy array
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


    @callable
    def read(self: Texture2DType(dtype), coord: int2):
        return _builtin_call(dtype, "TEXTURE_READ", [self, coord])

    @callable
    def write(self: Texture2DType(dtype), coord: int2, value: dtype):
        _builtin_call("TEXTURE_WRITE", [self, coord, value])


class Texture2DType:
    def __init__(self, dtype):
        self.dtype = dtype
        self.luisa_type = lcapi.Type.from_("texture<2," + dtype.__name__ + ">")

    def __eq__(self, other):
        return type(other) is Texture2DType and self.dtype == other.dtype

    read = BuiltinFuncEntry("texture2d_read")
    write = BuiltinFuncEntry("texture2d_write")
