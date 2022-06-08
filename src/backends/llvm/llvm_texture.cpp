//
// Created by Mike Smith on 2022/6/8.
//

#include <backends/llvm/llvm_texture.h>

namespace luisa::compute::llvm {

LLVMTexture::LLVMTexture(PixelStorage storage, uint3 size, uint levels) noexcept
    : _storage{storage}, _pixel_stride{static_cast<uint>(pixel_storage_size(storage))} {
    for (auto i = 0u; i < 3u; i++) { _size[i] = size[i]; }
    _mip_offsets[0] = 0u;
    for (auto i = 1u; i < levels; i++) {
        _mip_offsets[i] = _mip_offsets[i - 1u] +
                          _size[0] * _size[1] * _size[2] * _pixel_stride;
    }
    auto size_bytes = _mip_offsets[levels - 1u] +
                      _size[0] * _size[1] * _size[2] * _pixel_stride;
    _data = luisa::allocate<std::byte>(size_bytes);
}

void texture_write_2d_float(LLVMTextureView tex, uint2 xy, float4 v) noexcept { tex.write2d<float>(xy, v); }
void texture_write_3d_float(LLVMTextureView tex, uint3 xyz, float4 v) noexcept { tex.write3d<float>(xyz, v); }
void texture_write_2d_int(LLVMTextureView tex, uint2 xy, int4 v) noexcept { tex.write2d<int>(xy, v); }
void texture_write_3d_int(LLVMTextureView tex, uint3 xyz, int4 v) noexcept { tex.write3d<int>(xyz, v); }
void texture_write_2d_uint(LLVMTextureView tex, uint2 xy, uint4 v) noexcept { tex.write2d<uint>(xy, v); }
void texture_write_3d_uint(LLVMTextureView tex, uint3 xyz, uint4 v) noexcept { tex.write3d<uint>(xyz, v); }
float4 texture_read_2d_float(LLVMTextureView tex, uint2 xy) noexcept { return tex.read2d<float>(xy); }
float4 texture_read_3d_float(LLVMTextureView tex, uint3 xyz) noexcept { return tex.read3d<float>(xyz); }
int4 texture_read_2d_int(LLVMTextureView tex, uint2 xy) noexcept { return tex.read2d<int>(xy); }
int4 texture_read_3d_int(LLVMTextureView tex, uint3 xyz) noexcept { return tex.read3d<int>(xyz); }
uint4 texture_read_2d_uint(LLVMTextureView tex, uint2 xy) noexcept { return tex.read2d<uint>(xy); }
uint4 texture_read_3d_uint(LLVMTextureView tex, uint3 xyz) noexcept { return tex.read3d<uint>(xyz); }

}// namespace luisa::compute::llvm
