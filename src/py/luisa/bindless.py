from . import lcapi
from .lcapi import uint2
from . import globalvars
from .globalvars import get_global_device as device
from .mathtypes import *
from . import Buffer, Texture2D
from .types import BuiltinFuncBuilder, to_lctype
from .builtin import check_exact_signature
from .func import func
from .builtin import _builtin_call

class BindlessArray:
    def __init__(self, n_slots = 65536):
        self.handle = device().impl().create_bindless_array(n_slots)

    @staticmethod
    def bindless_array(dic):
        arr = BindlessArray.empty()
        for i in dic:
            arr.emplace(i, dic[i])
        arr.update()
        return arr

    @staticmethod
    def empty(n_slots = 65536):
        return BindlessArray(n_slots)

    def emplace(self, idx, res):
        if type(res) is Buffer:
            device().impl().emplace_buffer_in_bindless_array(self.handle, idx, res.handle, 0)
        elif type(res) is Texture2D:
            if res.dtype != float:
                raise TypeError("Type of emplaced Texture2D must be float")
            sampler = lcapi.Sampler(lcapi.Sampler.Filter.LINEAR_POINT, lcapi.Sampler.Address.REPEAT)
            device().impl().emplace_tex2d_in_bindless_array(self.handle, idx, res.handle, sampler)
        else:
            raise TypeError(f"can't emplace {type(res)} in bindless array")

    def remove_buffer(self, idx):
        device().impl().remove_buffer_in_bindless_array(self.handle, idx)
        
    def remove_texture2d(self, idx):
        device().impl().remove_tex2d_in_bindless_array(self.handle, idx)

    def __contains__(self, res):
        return device().impl().is_resource_in_bindless_array(self.handle, res.handle)

    def update(self, sync = False, stream = None):
        if stream is None:
            stream = globalvars.stream
        cmd = lcapi.BindlessArrayUpdateCommand.create(self.handle)
        stream.add(cmd)
        if sync:
            stream.synchronize()

    # @func
    # def buffer_read(self: BindlessArray, dtype: type, buffer_index: int, element_index: int):
    #     return _builtin_call(dtype, "BINDLESS_BUFFER_READ", self, buffer_index, element_index)
    # might not be possible, because "type" is not a valid data type in LC

    @BuiltinFuncBuilder
    def buffer_read(*argnodes): # (dtype, buffer_index, element_index)
        check_exact_signature([type, int, int], argnodes[1:], "buffer_read")
        dtype = argnodes[1].expr
        expr = lcapi.builder().call(to_lctype(dtype), lcapi.CallOp.BINDLESS_BUFFER_READ, [x.expr for x in [argnodes[0]] + list(argnodes[2:])])
        return dtype, expr

    @func
    def texture2d_read(self, texture2d_index: int, coord: int2):
        return _builtin_call(float4, "BINDLESS_TEXTURE2D_READ", self, texture2d_index, uint2(coord))

    @func
    def texture2d_sample(self, texture2d_index: int, uv: float2):
        return _builtin_call(float4, "BINDLESS_TEXTURE2D_SAMPLE", self, texture2d_index, uv)

    @func
    def texture2d_sample_grad(self, texture2d_index: int, uv: float2, ddx: float2, ddy: float2):
        return _builtin_call(float4, "BINDLESS_TEXTURE2D_SAMPLE_GRAD", self, texture2d_index, uv, ddx, ddy)

    @func
    def texture2d_size(self, texture2d_index: int):
        return int2(_builtin_call(uint2, "BINDLESS_TEXTURE2D_SIZE", self, texture2d_index))

bindless_array = BindlessArray.bindless_array
