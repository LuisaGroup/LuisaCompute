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
    : _storage{storage}, _mip_levels{levels},
      _pixel_stride{static_cast<uint>(pixel_storage_size(storage))} {
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

float4 LLVMTexture::read2d(uint level, uint2 uv) const noexcept {
    return view(level).read2d<float>(uv);
}

float4 LLVMTexture::read3d(uint level, uint3 uvw) const noexcept {
    return view(level).read3d<float>(uvw);
}

template<typename T>
[[nodiscard]] inline auto texture_coord_point(Sampler::Address address, T uv, T s) noexcept {
    switch (address) {
        case Sampler::Address::EDGE: return luisa::clamp(uv, 0.0f, one_minus_epsilon) * s;
        case Sampler::Address::REPEAT: return luisa::fract(uv) * s;
        case Sampler::Address::MIRROR: return luisa::fract(luisa::fmod(uv, T{2.f})) * s;
        case Sampler::Address::ZERO: return luisa::select(uv * s, T{65536.f}, uv < 0.f || uv >= s);
    }
    return T{65536.f};
}

[[nodiscard]] inline auto texture_coord_linear(Sampler::Address address, float2 uv, float2 size) noexcept {
    auto s = make_float2(size);
    auto inv_s = 1.f / s;
    auto c_min = texture_coord_point(address, uv - .5f * inv_s, s);
    auto c_max = texture_coord_point(address, uv + .5f * inv_s, s);
    return std::make_pair(luisa::min(c_min, c_max), luisa::max(c_min, c_max));
}

[[nodiscard]] inline auto texture_coord_linear(Sampler::Address address, float3 uv, float3 size) noexcept {
    auto s = make_float3(size);
    auto inv_s = 1.f / s;
    auto c_min = texture_coord_point(address, uv - .5f * inv_s, s);
    auto c_max = texture_coord_point(address, uv + .5f * inv_s, s);
    return std::make_pair(luisa::min(c_min, c_max), luisa::max(c_min, c_max));
}

[[nodiscard]] inline auto texture_sample_linear(LLVMTextureView view, Sampler::Address address, float2 uv, float2 size) noexcept {
    auto [st_min, st_max] = texture_coord_linear(address, uv, size);
    auto t = 1.f - luisa::fract(st_max);
    auto c0 = make_uint2(st_min);
    auto c1 = make_uint2(st_max);
    auto v00 = view.read2d<float>(c0);
    auto v01 = view.read2d<float>(make_uint2(c1.x, c0.y));
    auto v10 = view.read2d<float>(make_uint2(c0.x, c1.y));
    auto v11 = view.read2d<float>(c1);
    return luisa::lerp(luisa::lerp(v00, v01, t.x),
                       luisa::lerp(v10, v11, t.x), t.y);
}

[[nodiscard]] inline auto texture_sample_linear(LLVMTextureView view, Sampler::Address address, float3 uvw, float3 size) noexcept {
    auto [st_min, st_max] = texture_coord_linear(address, uvw, size);
    auto t = 1.f - luisa::fract(st_max);
    auto c0 = make_uint3(st_min);
    auto c1 = make_uint3(st_max);
    auto v000 = view.read3d<float>(make_uint3(c0.x, c0.y, c0.z));
    auto v001 = view.read3d<float>(make_uint3(c1.x, c0.y, c0.z));
    auto v010 = view.read3d<float>(make_uint3(c0.x, c1.y, c0.z));
    auto v011 = view.read3d<float>(make_uint3(c1.x, c1.y, c0.z));
    auto v100 = view.read3d<float>(make_uint3(c0.x, c0.y, c1.z));
    auto v101 = view.read3d<float>(make_uint3(c1.x, c0.y, c1.z));
    auto v110 = view.read3d<float>(make_uint3(c0.x, c1.y, c1.z));
    auto v111 = view.read3d<float>(make_uint3(c1.x, c1.y, c1.z));
    return luisa::lerp(
        luisa::lerp(luisa::lerp(v000, v001, t.x),
                    luisa::lerp(v010, v011, t.x), t.y),
        luisa::lerp(luisa::lerp(v100, v101, t.x),
                    luisa::lerp(v110, v111, t.x), t.y),
        t.z);
}

[[nodiscard]] inline auto texture_sample_point(LLVMTextureView view, Sampler::Address address, float2 uv, float2 size) noexcept {
    auto c = make_uint2(texture_coord_point(address, uv, size));
    return view.read2d<float>(c);
}

[[nodiscard]] inline auto texture_sample_point(LLVMTextureView view, Sampler::Address address, float3 uvw, float3 size) noexcept {
    auto c = make_uint3(texture_coord_point(address, uvw, size));
    return view.read3d<float>(c);
}

float4 LLVMTexture::sample2d(Sampler sampler, float2 uv) const noexcept {
    auto size = make_float2(_size[0], _size[1]);
    return sampler.filter() == Sampler::Filter::POINT ?
               texture_sample_point(view(0), sampler.address(), uv, size) :
               texture_sample_linear(view(0), sampler.address(), uv, size);
}

float4 LLVMTexture::sample3d(Sampler sampler, float3 uvw) const noexcept {
    auto size = make_float3(_size[0], _size[1], _size[2]);
    return sampler.filter() == Sampler::Filter::POINT ?
               texture_sample_point(view(0), sampler.address(), uvw, size) :
               texture_sample_linear(view(0), sampler.address(), uvw, size);
}

float4 LLVMTexture::sample2d(Sampler sampler, float2 uv, float lod) const noexcept {
    auto filter = sampler.filter();
    if (lod <= 0.f || _mip_levels == 0u ||
        filter == Sampler::Filter::POINT) {
        return sample2d(sampler, uv);
    }
    auto level0 = std::min(static_cast<uint32_t>(lod),
                           _mip_levels - 1u);
    auto view0 = view(level0);
    auto v0 = texture_sample_linear(
        view0, sampler.address(), uv,
        make_float2(view0.size2d()));
    if (level0 == _mip_levels - 1u ||
        filter == Sampler::Filter::LINEAR_POINT) {
        return v0;
    }
    auto view1 = view(level0 + 1u);
    auto v1 = texture_sample_linear(
        view1, sampler.address(), uv,
        make_float2(view1.size2d()));
    return luisa::lerp(v0, v1, luisa::fract(lod));
}

float4 LLVMTexture::sample3d(Sampler sampler, float3 uvw, float lod) const noexcept {
    auto filter = sampler.filter();
    if (lod <= 0.f || _mip_levels == 0u ||
        filter == Sampler::Filter::POINT) {
        return sample3d(sampler, uvw);
    }
    auto level0 = std::min(static_cast<uint32_t>(lod),
                           _mip_levels - 1u);
    auto view0 = view(level0);
    auto v0 = texture_sample_linear(
        view0, sampler.address(), uvw,
        make_float3(view0.size3d()));
    if (level0 == _mip_levels - 1u ||
        filter == Sampler::Filter::LINEAR_POINT) {
        return v0;
    }
    auto view1 = view(level0 + 1u);
    auto v1 = texture_sample_linear(
        view1, sampler.address(), uvw,
        make_float3(view1.size3d()));
    return luisa::lerp(v0, v1, luisa::fract(lod));
}

float4 LLVMTexture::sample2d(Sampler sampler, float2 uv, float2 dpdx, float2 dpdy) const noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
    return {};// TODO
}

float4 LLVMTexture::sample3d(Sampler sampler, float3 uvw, float3 dpdx, float3 dpdy) const noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
    return {};// TODO
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
    return detail::encode_float4(tex->read2d(level, make_uint2(x, y)));
}

detail::ulong2 bindless_texture_3d_read(const LLVMTexture *tex, uint level, uint x, uint y, uint z) noexcept {
    return detail::encode_float4(tex->read3d(level, make_uint3(x, y, z)));
}

detail::ulong2 bindless_texture_2d_sample(const LLVMTexture *tex, uint sampler, float u, float v) noexcept {
    return detail::encode_float4(tex->sample2d(Sampler::decode(sampler), make_float2(u, v)));
}

detail::ulong2 bindless_texture_3d_sample(const LLVMTexture *tex, uint sampler, float u, float v, float w) noexcept {
    return detail::encode_float4(tex->sample3d(Sampler::decode(sampler), make_float3(u, v, w)));
}

detail::ulong2 bindless_texture_2d_sample_level(const LLVMTexture *tex, uint sampler, float u, float v, float lod) noexcept {
    return detail::encode_float4(tex->sample2d(Sampler::decode(sampler), make_float2(u, v), lod));
}

detail::ulong2 bindless_texture_3d_sample_level(const LLVMTexture *tex, uint sampler, float u, float v, float w, float lod) noexcept {
    return detail::encode_float4(tex->sample3d(Sampler::decode(sampler), make_float3(u, v, w), lod));
}

detail::ulong2 bindless_texture_2d_sample_grad(const LLVMTexture *tex, uint sampler, float u, float v, uint64_t dpdx, uint64_t dpdy) noexcept {
    return detail::encode_float4(tex->sample2d(
        Sampler::decode(sampler), make_float2(u, v),
        detail::decode_float2(dpdx), detail::decode_float2(dpdy)));
}

detail::ulong2 bindless_texture_3d_sample_grad(const LLVMTexture *tex, uint64_t sampler_w, uint64_t uv, uint64_t dudxy, uint64_t dvdxy, uint64_t dwdxy) noexcept {
    struct alignas(8) sampler_and_float {
        uint sampler;
        float w;
    };
    auto du_dxy = detail::decode_float2(dudxy);
    auto dv_dxy = detail::decode_float2(dvdxy);
    auto dw_dxy = detail::decode_float2(dwdxy);
    auto dpdx = make_float3(du_dxy.x, dv_dxy.x, dw_dxy.x);
    auto dpdy = make_float3(du_dxy.y, dv_dxy.y, dw_dxy.y);
    auto [sampler, w] = luisa::bit_cast<sampler_and_float>(sampler_w);
    auto uvw = make_float3(detail::decode_float2(uv), w);
    return detail::encode_float4(tex->sample3d(Sampler::decode(sampler), uvw, dpdx, dpdy));
}

}// namespace luisa::compute::llvm
