//
// Created by Mike Smith on 2022/6/8.
//

#include <backends/llvm/llvm_texture.h>

namespace luisa::compute::llvm {

namespace detail {

// from tinyexr: https://github.com/syoyo/tinyexr/blob/master/tinyexr.h
uint float_to_half(float f) noexcept {
    auto bits = luisa::bit_cast<uint>(f);
    auto fp32_sign = bits >> 31u;
    auto fp32_exponent = (bits >> 23u) & 0xffu;
    auto fp32_mantissa = bits & ((1u << 23u) - 1u);
    auto make_fp16 = [](uint sign, uint exponent, uint mantissa) noexcept {
        return (sign << 15u) | (exponent << 10u) | mantissa;
    };
    // Signed zero/denormal (which will underflow)
    if (fp32_exponent == 0u) { return make_fp16(fp32_sign, 0u, 0u); }
    // Inf or NaN (all exponent bits set)
    if (fp32_exponent == 255u) {
        return make_fp16(
            fp32_sign, 31u,
            // NaN->qNaN and Inf->Inf
            fp32_mantissa ? 0x200u : 0u);
    }
    // Exponent unbias the single, then bias the halfp
    auto newexp = static_cast<int>(fp32_exponent - 127u + 15u);
    // Overflow, return signed infinity
    if (newexp >= 31) { return make_fp16(fp32_sign, 31u, 0u); }
    // Underflow
    if (newexp <= 0) {
        if ((14 - newexp) > 24) { return 0u; }
        // Mantissa might be non-zero
        unsigned int mant = fp32_mantissa | 0x800000u;// Hidden 1 bit
        auto fp16 = make_fp16(fp32_sign, 0u, mant >> (14u - newexp));
        if ((mant >> (13u - newexp)) & 1u) { fp16++; }// Check for rounding
        return fp16;
    }
    auto fp16 = make_fp16(fp32_sign, newexp, fp32_mantissa >> 13u);
    if (fp32_mantissa & 0x1000u) { fp16++; }// Check for rounding
    return fp16;
}

}// namespace detail

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

LLVMTexture::~LLVMTexture() noexcept { luisa::deallocate(_data); }

LLVMTextureView LLVMTexture::view(uint level) const noexcept {
    auto size = luisa::max(make_uint2(_size[0], _size[1]) >> level, 1u);
    return LLVMTextureView{_data + _mip_offsets[level], size.x, size.y,
                           _storage, _pixel_stride};
}

int4 texture_read_2d_int(LLVMTextureView tex, uint2 xy) noexcept { return tex.read2d<int>(xy); }
int4 texture_read_3d_int(LLVMTextureView tex, uint3 xyz) noexcept { return tex.read3d<int>(xyz); }
uint4 texture_read_2d_uint(LLVMTextureView tex, uint2 xy) noexcept { return tex.read2d<uint>(xy); }
uint4 texture_read_3d_uint(LLVMTextureView tex, uint3 xyz) noexcept { return tex.read3d<uint>(xyz); }
float4 texture_read_2d_float(LLVMTextureView tex, uint2 xy) noexcept { return tex.read2d<float>(xy); }
float4 texture_read_3d_float(LLVMTextureView tex, uint3 xyz) noexcept { return tex.read3d<float>(xyz); }

void texture_write_2d_int(LLVMTextureView tex, uint2 xy, int4 v) noexcept { tex.write2d<int>(xy, v); }
void texture_write_3d_int(LLVMTextureView tex, uint3 xyz, int4 v) noexcept { tex.write3d<int>(xyz, v); }
void texture_write_2d_uint(LLVMTextureView tex, uint2 xy, uint4 v) noexcept { tex.write2d<uint>(xy, v); }
void texture_write_3d_uint(LLVMTextureView tex, uint3 xyz, uint4 v) noexcept { tex.write3d<uint>(xyz, v); }
void texture_write_2d_float(LLVMTextureView tex, uint2 xy, float4 v) noexcept { tex.write2d<float>(xy, v); }
void texture_write_3d_float(LLVMTextureView tex, uint3 xyz, float4 v) noexcept { tex.write3d<float>(xyz, v); }

}// namespace luisa::compute::llvm
