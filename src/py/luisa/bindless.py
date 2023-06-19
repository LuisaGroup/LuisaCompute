from .dylibs import lcapi
from .dylibs.lcapi import uint2, uint3
from . import globalvars
from .globalvars import get_global_device as device
from .mathtypes import *
from . import Buffer, Image2D, Image3D
from .types import BuiltinFuncBuilder, to_lctype, uint
from .builtin import check_exact_signature


class BindlessArray:
    def __init__(self, n_slots=65536):
        self.array = device().impl().create_bindless_array(n_slots)
        self.handle = lcapi.get_bindless_handle(self.array)

    def __del__(self):
        if (self.array is not None):
            local_device = device()
            if local_device is not None:
                local_device.impl().destroy_bindless_array(self.array)

    @staticmethod
    def bindless_array(dic):
        arr = BindlessArray.empty()
        for i in dic:
            arr.emplace(i, dic[i])
        arr.update()
        return arr

    @staticmethod
    def empty(n_slots=65536):
        return BindlessArray(n_slots)

    def emplace(self, idx, res, filter=None, address=None, byte_offset=0):
        if type(res) is Buffer:
            device().impl().emplace_buffer_in_bindless_array(self.array, idx, res.handle, byte_offset)
        elif type(res) is Image2D:
            if res.dtype != float:
                raise TypeError("Type of emplaced Image2D must be float")
            if filter is None:
                filter = lcapi.Filter.LINEAR_POINT
            if address is None:
                address = lcapi.Address.REPEAT
            sampler = lcapi.Sampler(filter, address)
            device().impl().emplace_tex2d_in_bindless_array(self.array, idx, res.handle, sampler)
        elif type(res) is Image3D:
            if res.dtype != float:
                raise TypeError("Type of emplaced Image3D must be float")
            if filter is None:
                filter = lcapi.Filter.LINEAR_POINT
            if address is None:
                address = lcapi.Address.REPEAT
            sampler = lcapi.Sampler(filter, address)
            device().impl().emplace_tex3d_in_bindless_array(self.array, idx, res.handle, sampler)
        else:
            raise TypeError(f"can't emplace {type(res)} in bindless array")

    def remove_buffer(self, idx):
        device().impl().remove_buffer_in_bindless_array(self.array, idx)

    def remove_texture2d(self, idx):
        device().impl().remove_tex2d_in_bindless_array(self.array, idx)

    def remove_texture3d(self, idx):
        device().impl().remove_tex3d_in_bindless_array(self.array, idx)

    def update(self, sync=False, stream=None):
        if stream is None:
            stream = globalvars.vars.stream
        stream.update_bindless(self.array)
        if sync:
            stream.synchronize()

    @BuiltinFuncBuilder
    def buffer_read(*argnodes):  # (dtype, buffer_index, element_index)
        check_exact_signature([type, int, uint], argnodes[1:], "buffer_read")
        dtype = argnodes[1].expr
        expr = lcapi.builder().call(to_lctype(dtype), lcapi.CallOp.BINDLESS_BUFFER_READ,
                                    [x.expr for x in [argnodes[0]] + list(argnodes[2:])])
        return dtype, expr
    
    @BuiltinFuncBuilder
    def byte_address_buffer_read(*argnodes):  # (dtype, buffer_index, element_index)
        check_exact_signature([type, int, uint], argnodes[1:], "byte_address_buffer_read")
        dtype = argnodes[1].expr
        expr = lcapi.builder().call(to_lctype(dtype), lcapi.CallOp.BINDLESS_BYTE_ADDRESS_BUFFER_READ,
                                    [x.expr for x in [argnodes[0]] + list(argnodes[2:])])
        return dtype, expr

    @BuiltinFuncBuilder
    def texture2d_read(self, texture2d_index, coord):
        check_exact_signature([uint, uint2], [texture2d_index, coord], "texture2d_read")
        return float4, lcapi.builder().call(to_lctype(float4), lcapi.CallOp.BINDLESS_TEXTURE2D_READ, [self.expr, texture2d_index.expr, coord.expr])

    @BuiltinFuncBuilder
    def texture2d_sample(self, texture2d_index, uv):
        check_exact_signature([uint, float2], [texture2d_index, uv], "texture2d_sample")
        return float4, lcapi.builder().call(to_lctype(float4), lcapi.CallOp.BINDLESS_TEXTURE2D_SAMPLE, [self.expr, texture2d_index.expr, uv.expr])

    @BuiltinFuncBuilder
    def texture2d_sample_mip(self, texture2d_index, uv, mip):
        check_exact_signature([uint, float2, uint], [texture2d_index, uv, mip], "texture2d_sample_mip")
        return float4, lcapi.builder().call(to_lctype(float4), lcapi.CallOp.BINDLESS_TEXTURE2D_SAMPLE_LEVEL, [self.expr, texture2d_index.expr, uv.expr, mip.expr])

    @BuiltinFuncBuilder
    def texture2d_sample_grad(self, texture2d_index, uv, ddx, ddy):
        check_exact_signature([uint, float2, float2, float2], [texture2d_index, uv, ddx, ddy], "texture2d_sample_grad")
        return float4, lcapi.builder().call(to_lctype(float4), lcapi.CallOp.BINDLESS_TEXTURE2D_SAMPLE_GRAD, [self.expr, texture2d_index.expr, uv.expr, ddx.expr, ddy.expr])
    
    @BuiltinFuncBuilder
    def texture2d_sample_grad_level(self, texture2d_index, uv, ddx, ddy, min_mip):
        check_exact_signature([uint, float2, float2, float2, float], [texture2d_index, uv, ddx, ddy, min_mip], "texture2d_sample_grad_level")
        return float4, lcapi.builder().call(to_lctype(float4), lcapi.CallOp.BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL, [self.expr, texture2d_index.expr, uv.expr, ddx.expr, ddy.expr, min_mip.expr])

    @BuiltinFuncBuilder
    def buffer_size(self, buffer_index):
        check_exact_signature([uint], [buffer_index], "texture2d_size")
        return uint, lcapi.builder().call(to_lctype(uint), lcapi.CallOp.BINDLESS_BUFFER_SIZE, [self.expr, buffer_index.expr])

    @BuiltinFuncBuilder
    def texture2d_size(self, texture2d_index):
        check_exact_signature([uint], [texture2d_index], "texture2d_size")
        return uint2, lcapi.builder().call(to_lctype(uint2), lcapi.CallOp.BINDLESS_TEXTURE2D_SIZE, [self.expr, texture2d_index.expr])

    @BuiltinFuncBuilder
    def texture3d_read(self, texture3d_index, coord):
        check_exact_signature([uint, uint3], [texture3d_index, coord], "texture3d_read")
        return float4, lcapi.builder().call(to_lctype(float4), lcapi.CallOp.BINDLESS_TEXTURE3D_READ, [self.expr, texture3d_index.expr, coord.expr])

    @BuiltinFuncBuilder
    def texture3d_sample(self, texture3d_index, uv):
        check_exact_signature([uint, float3], [texture3d_index, uv], "texture3d_sample")
        return float4, lcapi.builder().call(to_lctype(float4), lcapi.CallOp.BINDLESS_TEXTURE3D_SAMPLE, [self.expr, texture3d_index.expr, uv.expr])

    @BuiltinFuncBuilder
    def texture3d_sample_mip(self, texture3d_index, uv, mip):
        check_exact_signature([uint, float3, uint], [texture3d_index, uv, mip], "texture3d_sample_mip")
        return float4, lcapi.builder().call(to_lctype(float4), lcapi.CallOp.BINDLESS_TEXTURE3D_SAMPLE_LEVEL, [self.expr, texture3d_index.expr, uv.expr, mip.expr])

    @BuiltinFuncBuilder
    def texture3d_sample_grad(self, texture3d_index, uv, ddx, ddy):
        check_exact_signature([uint, float3, float3, float3], [texture3d_index, uv, ddx, ddy], "texture3d_sample_grad")
        return float4, lcapi.builder().call(to_lctype(float4), lcapi.CallOp.BINDLESS_TEXTURE3D_SAMPLE_GRAD, [self.expr, texture3d_index.expr, uv.expr, ddx.expr, ddy.expr])
    
    @BuiltinFuncBuilder
    def texture3d_sample_grad_level(self, texture3d_index, uv, ddx, ddy, min_mip):
        check_exact_signature([uint, float3, float3, float3, float], [texture3d_index, uv, ddx, ddy, min_mip], "texture3d_sample_grad_level")
        return float4, lcapi.builder().call(to_lctype(float4), lcapi.CallOp.BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL, [self.expr, texture3d_index.expr, uv.expr, ddx.expr, ddy.expr, min_mip.expr])

    @BuiltinFuncBuilder
    def texture3d_size(self, texture3d_index):
        check_exact_signature([uint], [texture3d_index], "texture3d_size")
        return uint3, lcapi.builder().call(to_lctype(uint3), lcapi.CallOp.BINDLESS_TEXTURE3D_SIZE, [self.expr, texture3d_index.expr])


bindless_array = BindlessArray.bindless_array
