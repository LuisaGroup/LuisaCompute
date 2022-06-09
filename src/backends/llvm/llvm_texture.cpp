//
// Created by Mike Smith on 2022/6/8.
//

#include <backends/llvm/llvm_texture.h>

namespace luisa::compute::llvm {

namespace detail {

[[nodiscard]] inline auto decode_texture_view(uint64_t t0, uint64_t t1) noexcept {
    return luisa::bit_cast<LLVMTextureView>(ulong2{t0, t1});
}

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
    auto size = luisa::max(make_uint3(_size[0], _size[1], _size[2]) >> level, 1u);
    return LLVMTextureView{_data + _mip_offsets[level],
                           size.x, size.y, size.z,
                           _storage, _pixel_stride};
}

void texture_write_2d_int(uint64_t t0, uint64_t t1, uint64_t c0, uint64_t c1, uint64_t v0, uint64_t v1) noexcept {
    detail::decode_texture_view(t0, t1).write2d<int>(
        detail::decode_uint2(c0), detail::decode_int4(v0, v1));
}

void texture_write_3d_int(uint64_t t0, uint64_t t1, uint64_t c0, uint64_t c1, uint64_t v0, uint64_t v1) noexcept {
    detail::decode_texture_view(t0, t1).write3d<int>(
        detail::decode_uint3(c0, c1), detail::decode_int4(v0, v1));
}

void texture_write_2d_uint(uint64_t t0, uint64_t t1, uint64_t c0, uint64_t c1, uint64_t v0, uint64_t v1) noexcept {
    detail::decode_texture_view(t0, t1).write2d<uint>(
        detail::decode_uint2(c0),
        detail::decode_uint4(v0, v1));
}

void texture_write_3d_uint(uint64_t t0, uint64_t t1, uint64_t c0, uint64_t c1, uint64_t v0, uint64_t v1) noexcept {
    detail::decode_texture_view(t0, t1).write3d<uint>(
        detail::decode_uint3(c0, c1), detail::decode_uint4(v0, v1));
}

void texture_write_2d_float(uint64_t t0, uint64_t t1, uint64_t c0, uint64_t c1, uint64_t v0, uint64_t v1) noexcept {
    detail::decode_texture_view(t0, t1).write2d<float>(
        detail::decode_uint2(c0), detail::decode_float4(v0, v1));
}

void texture_write_3d_float(uint64_t t0, uint64_t t1, uint64_t c0, uint64_t c1, uint64_t v0, uint64_t v1) noexcept {
    detail::decode_texture_view(t0, t1).write3d<float>(
        detail::decode_uint3(c0, c1), detail::decode_float4(v0, v1));
}

detail::ulong2 texture_read_2d_int(uint64_t t0, uint64_t t1, uint64_t c0, uint64_t c1) noexcept {
    return detail::encode_int4(
        detail::decode_texture_view(t0, t1).read2d<int>(
            detail::decode_uint2(c0)));
}

detail::ulong2 texture_read_3d_int(uint64_t t0, uint64_t t1, uint64_t c0, uint64_t c1) noexcept {
    return detail::encode_int4(
        detail::decode_texture_view(t0, t1).read3d<int>(
            detail::decode_uint3(c0, c1)));
}

detail::ulong2 texture_read_2d_uint(uint64_t t0, uint64_t t1, uint64_t c0, uint64_t c1) noexcept {
    return detail::encode_uint4(
        detail::decode_texture_view(t0, t1).read2d<uint>(
            detail::decode_uint2(c0)));
}

detail::ulong2 texture_read_3d_uint(uint64_t t0, uint64_t t1, uint64_t c0, uint64_t c1) noexcept {
    return detail::encode_uint4(
        detail::decode_texture_view(t0, t1).read3d<uint>(
            detail::decode_uint3(c0, c1)));
}

detail::ulong2 texture_read_2d_float(uint64_t t0, uint64_t t1, uint64_t c0, uint64_t c1) noexcept {
    return detail::encode_float4(
        detail::decode_texture_view(t0, t1).read2d<float>(
            detail::decode_uint2(c0)));
}

detail::ulong2 texture_read_3d_float(uint64_t t0, uint64_t t1, uint64_t c0, uint64_t c1) noexcept {
    return detail::encode_float4(
        detail::decode_texture_view(t0, t1).read3d<float>(
            detail::decode_uint3(c0, c1)));
}

detail::ulong2 bindless_texture_2d_read(const LLVMTexture *tex, uint level, uint x, uint y) noexcept {
    auto v = tex->view(level).read2d<float>(make_uint2(x, y));
    return detail::encode_float4(v);
}

detail::ulong2 bindless_texture_3d_read(const LLVMTexture *tex, uint level, uint x, uint y, uint z) noexcept {
    auto v = tex->view(level).read3d<float>(make_uint3(x, y, z));
    return detail::encode_float4(v);
}

}// namespace luisa::compute::llvm
