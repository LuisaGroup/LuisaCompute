import lcapi
from . import globalvars
from .globalvars import get_global_device as device
from . import Buffer, Texture2D, int2
from .types import BuiltinFuncBuilder
from .builtin import check_exact_signature
from . import callable
from .builtin import _builtin_call

class BindlessArray:
    def __init__(self, n_slots = 65536):
        self.handle = device().impl().create_bindless_array(n_slots)

    def emplace(self, idx, res):
        if type(res) is Buffer:
            device().impl().emplace_buffer_in_bindless_array(self.handle, idx, res.handle, 0)
        elif type(res) is Texture2D:
            if res.dtype != float:
                raise TypeError("Type of emplaced Texture2D must be float")
            sampler = lcapi.Sampler(lcapi.Sampler.Filter.POINT, lcapi.Sampler.Address.EDGE)
            device().impl().emplace_tex2d_in_bindless_array(self.handle, idx, res.handle, sampler)
        else:
            raise TypeError(f"can't emplace {type(res)} in bindless array")

    def remove_buffer(self, idx):
        device().impl().remove_buffer_in_bindless_array(self.handle, idx)
        
    def remove_texture2d(self, idx):
        device().impl().remove_tex2d_in_bindless_array(self.handle, idx)

    def __contains__(self, res):
        return device().impl().is_resource_in_bindless_array(self.handle, res.handle)

    def update(self, stream = None):
        if stream is None:
            stream = globalvars.stream
        cmd = lcapi.BindlessArrayUpdateCommand.create(self.handle)
        stream.add(cmd)

    @callable
    def buffer_read(self: BindlessArray, dtype: type, buffer_index: int, element_index: int):
        return _builtin_call(dtype, "BINDLESS_BUFFER_READ", self, buffer_index, element_index)
    # might not be possible, because "type" is not a valid data type in LC

    # @BuiltinFuncBuilder
    # def buffer_read(argnodes): # (dtype, buffer_index, element_index)
    #     check_exact_signature([type, int, int], argnodes[1:], "buffer_read")
    #     dtype = argnodes[1]
    #     expr = lcapi.builder().call(to_lctype(dtype), lcapi.CallOp.BINDLESS_BUFFER_READ, [argnodes[0]] + argnodes[2:])
    #     return dtype, expr

    @callable
    def texture2d_read(self: BindlessArray, texture2d_index: int, coord: int2):
        return _builtin_call(float4, "BINDLESS_TEXTURE2D_READ", self, texture2d_index, uint2(coord))

    @callable
    def texture2d_sample(self: BindlessArray, texture2d_index: int, uv: float2):
        return _builtin_call(float4, "BINDLESS_TEXTURE2D_SAMPLE", self, texture2d_index, uv)

    @callable
    def texture2d_size(self: BindlessArray, texture2d_index: int):
        return int2(_builtin_call(uint2, "BINDLESS_TEXTURE2D_SIZE", self, texture2d_index))


