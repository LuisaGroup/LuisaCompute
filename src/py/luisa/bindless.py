import lcapi
from . import globalvars
from .globalvars import get_global_device as device
from . import Buffer, Texture2D, int2
from .types import BuiltinFuncBuilder
from .builtin import check_exact_signature

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

    @BuiltinFuncBuilder
    def buffer_read(argnodes): # (dtype, buffer_index, element_index)
        check_exact_signature([type, int, int], argnodes[1:], "buffer_read")
        dtype = argnodes[1]
        expr = lcapi.builder().call(to_lctype(dtype), lcapi.CallOp.BINDLESS_BUFFER_READ, [argnodes[0]] + argnodes[2:])
        return dtype, expr

    @BuiltinFuncBuilder
    def texture2d_read(argnodes): # (texture2d_index, coord: int2)
        check_exact_signature([int, int2], argnodes[1:], "texture2d_read")
        args[2].dtype, args[2].expr = builtin_type_cast(lcapi.uint2, [args[2]]) # convert int2 to uint2
        expr = lcapi.builder().call(to_lctype(float4), lcapi.CallOp.BINDLESS_TEXTURE2D_READ, argnodes)
        return float4, expr

    @BuiltinFuncBuilder
    def texture2d_sample(argnodes): # (texture2d_index, uv: float4)
        check_exact_signature([int, float2], argnodes[1:], "texture2d_sample")
        expr = lcapi.builder().call(to_lctype(float4), lcapi.CallOp.BINDLESS_TEXTURE2D_SAMPLE, argnodes)
        return float4, expr

    @BuiltinFuncBuilder
    def texture2d_size(argnodes): # (texture2d_index)
        check_exact_signature([int], argnodes[1:], "texture2d_size")
        expr = lcapi.builder().call(to_lctype(uint2), BINDLESS_TEXTURE2D_SIZE, argnodes)
        # convert expr to int2
        return int2, lcapi.builder().call(to_lctype(int2), lcapi.CallOp.MAKE_INT2, [expr])


