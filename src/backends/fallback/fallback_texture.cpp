//
// Created by Mike Smith on 2022/6/8.
//

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include "fallback_texture.h"

namespace luisa::compute::fallback {

namespace detail {
template<size_t stride>
struct alignas(stride) Pixel : std::array<std::byte, stride> {};
}// namespace detail

void FallbackTextureView::copy_from(const void *data) const noexcept {
    auto LC_TEXTURE_COPY = [data, this]<uint dim, uint stride>() mutable noexcept {
        auto p = static_cast<const detail::Pixel<stride> *>(data);
        for (auto z = 0u; z < (dim == 2u ? 1u : _depth); z++) {
            for (auto y = 0u; y < _height; y++) {
                for (auto x = 0u; x < _width; x++) {
                    auto pp = p + ((z * _height + y) * _width + x);
                    if constexpr (dim == 2) {
                        auto pixel = reinterpret_cast<detail::Pixel<stride> *>(
                            _pixel2d(make_uint2(x, y)));
                        *pixel = *pp;
                    } else {
                        auto pixel = reinterpret_cast<detail::Pixel<stride> *>(
                            _pixel3d(make_uint3(x, y, z)));
                        *pixel = *pp;
                    }
                }
            }
        }
    };
    if (_dimension == 2u) {
        switch (_pixel_stride_shift) {
            case 0u: LC_TEXTURE_COPY.operator()<2u, 1u>(); break;
            case 1u: LC_TEXTURE_COPY.operator()<2u, 2u>(); break;
            case 2u: LC_TEXTURE_COPY.operator()<2u, 4u>(); break;
            case 3u: LC_TEXTURE_COPY.operator()<2u, 8u>(); break;
            case 4u: LC_TEXTURE_COPY.operator()<2u, 16u>(); break;
            default: break;
        }
    } else {
        switch (_pixel_stride_shift) {
            case 0u: LC_TEXTURE_COPY.operator()<3u, 1u>(); break;
            case 1u: LC_TEXTURE_COPY.operator()<3u, 2u>(); break;
            case 2u: LC_TEXTURE_COPY.operator()<3u, 4u>(); break;
            case 3u: LC_TEXTURE_COPY.operator()<3u, 8u>(); break;
            case 4u: LC_TEXTURE_COPY.operator()<3u, 16u>(); break;
            default: break;
        }
    }
}

void FallbackTextureView::copy_to(void *data) const noexcept {
    memcpy(data, this->_data, this->size_bytes());
}

void FallbackTextureView::copy_from(FallbackTextureView dst) const noexcept {
    LUISA_ASSERT(size_bytes() == dst.size_bytes(), "Texture sizes must match.");
    std::memcpy(dst._data, _data, size_bytes());
}

namespace detail {

// MIP-Map EWA filtering LUT from PBRT-v4
static constexpr const std::array ewa_filter_weight_lut{
    0.8646647330f, 0.8490400310f, 0.8336595300f, 0.8185192940f, 0.8036156300f, 0.78894478100f, 0.7745032310f, 0.7602872850f,
    0.7462934850f, 0.7325183150f, 0.7189583780f, 0.7056102750f, 0.6924707890f, 0.67953658100f, 0.6668044920f, 0.6542713050f,
    0.6419339780f, 0.6297893520f, 0.6178345080f, 0.6060665250f, 0.5944823620f, 0.58307915900f, 0.5718541740f, 0.5608045460f,
    0.5499275920f, 0.5392205720f, 0.5286808610f, 0.5183058380f, 0.5080928800f, 0.49803954400f, 0.4881432650f, 0.4784016010f,
    0.4688121680f, 0.4593725800f, 0.4500804540f, 0.4409335260f, 0.4319294690f, 0.42306613900f, 0.4143413310f, 0.4057527780f,
    0.3972984550f, 0.3889762160f, 0.3807840350f, 0.3727198840f, 0.3647816180f, 0.35696744900f, 0.3492754100f, 0.3417034750f,
    0.3342499140f, 0.3269128200f, 0.3196903470f, 0.3125807050f, 0.3055821660f, 0.29869294200f, 0.2919114230f, 0.2852358220f,
    0.2786645290f, 0.2721959350f, 0.2658283710f, 0.2595603470f, 0.2533901930f, 0.24731649500f, 0.2413376720f, 0.2354522790f,
    0.2296588570f, 0.2239559440f, 0.2183421400f, 0.2128160450f, 0.2073762860f, 0.20202152400f, 0.1967504470f, 0.1915617140f,
    0.1864540130f, 0.1814261530f, 0.1764768510f, 0.1716048870f, 0.1668090670f, 0.16208814100f, 0.1574410050f, 0.1528664680f,
    0.1483634260f, 0.1439307180f, 0.1395672710f, 0.1352720110f, 0.1310438660f, 0.12688179300f, 0.1227847190f, 0.1187516900f,
    0.1147816330f, 0.1108736400f, 0.1070266960f, 0.1032398790f, 0.0995122194f, 0.09584279360f, 0.0922307223f, 0.0886750817f,
    0.0851749927f, 0.0817295909f, 0.0783380121f, 0.0749994367f, 0.0717130303f, 0.06847797330f, 0.0652934611f, 0.0621587038f,
    0.0590728968f, 0.0560353249f, 0.0530452281f, 0.0501018465f, 0.0472044498f, 0.04435232280f, 0.0415447652f, 0.0387810767f,
    0.0360605568f, 0.0333825648f, 0.0307464004f, 0.0281514227f, 0.0255970061f, 0.02308247980f, 0.0206072628f, 0.0181707144f,
    0.0157722086f, 0.0134112090f, 0.0110870898f, 0.0087992847f, 0.0065472275f, 0.00433036685f, 0.0021481365f, 0.0000000000f};

}// namespace detail

FallbackTexture::FallbackTexture(PixelStorage storage, uint dim, uint3 size, uint levels) noexcept
    : _storage{static_cast<uint16_t>(storage)}, _mip_levels{levels}, _dimension{dim}, _is_external{false} {
    if (_dimension == 2u) {
        _pixel_stride_shift = std::bit_width(static_cast<uint>(pixel_storage_size(storage, make_uint3(1u)))) - 1u;
        if (storage == PixelStorage::BC6 || storage == PixelStorage::BC7) {
            _pixel_stride_shift = 0u;
        }
        _size[0] = size.x;
        _size[1] = size.y;
        _size[2] = 1u;
        _mip_offsets[0] = 0u;
        auto sz = make_uint2(size);
        for (auto i = 1u; i < levels; i++) {
            auto s = (sz + block_size - 1u) / block_size * block_size;
            _mip_offsets[i] = _mip_offsets[i - 1u] + s.x * s.y;
            sz = luisa::max(sz >> 1u, 1u);
        }
        auto s = (sz + block_size - 1u) / block_size * block_size;
        auto size_pixels = _mip_offsets[levels - 1u] + s.x * s.y;
        _data = luisa::allocate_with_allocator<std::byte>(static_cast<size_t>(size_pixels) << _pixel_stride_shift);
    } else {
        _pixel_stride_shift = std::bit_width(static_cast<uint>(pixel_storage_size(storage, make_uint3(1u)))) - 1u;
        _size[0] = size.x;
        _size[1] = size.y;
        _size[2] = size.z;
        _mip_offsets[0] = 0u;
        for (auto i = 1u; i < levels; i++) {
            auto s = (size + block_size - 1u) / block_size * block_size;
            _mip_offsets[i] = _mip_offsets[i - 1u] + s.x * s.y * s.z;
            size = luisa::max(size >> 1u, 1u);
        }
        auto s = (size + block_size - 1u) / block_size * block_size;
        auto size_pixels = _mip_offsets[levels - 1u] + s.x * s.y * s.z;
        _data = luisa::allocate_with_allocator<std::byte>(static_cast<size_t>(size_pixels) << _pixel_stride_shift);
    }
}

FallbackTexture::FallbackTexture(PixelStorage storage, uint dim, uint3 size, uint levels, std::byte *external_buffer) noexcept
    : _storage{static_cast<uint16_t>(storage)}, _mip_levels{levels}, _dimension{dim}, _is_external{true} {
    if (_dimension == 2u) {
        _pixel_stride_shift = std::bit_width(static_cast<uint>(pixel_storage_size(storage, make_uint3(1u)))) - 1u;
        if (storage == PixelStorage::BC6 || storage == PixelStorage::BC7) {
            _pixel_stride_shift = 0u;
        }
        _size[0] = size.x;
        _size[1] = size.y;
        _size[2] = 1u;
        _mip_offsets[0] = 0u;
        auto sz = make_uint2(size);
        for (auto i = 1u; i < levels; i++) {
            auto s = (sz + block_size - 1u) / block_size * block_size;
            _mip_offsets[i] = _mip_offsets[i - 1u] + s.x * s.y;
            sz = luisa::max(sz >> 1u, 1u);
        }
        auto s = (sz + block_size - 1u) / block_size * block_size;
        auto size_pixels = _mip_offsets[levels - 1u] + s.x * s.y;
        _data = external_buffer;
    } else {
        _pixel_stride_shift = std::bit_width(static_cast<uint>(pixel_storage_size(storage, make_uint3(1u)))) - 1u;
        _size[0] = size.x;
        _size[1] = size.y;
        _size[2] = size.z;
        _mip_offsets[0] = 0u;
        for (auto i = 1u; i < levels; i++) {
            auto s = (size + block_size - 1u) / block_size * block_size;
            _mip_offsets[i] = _mip_offsets[i - 1u] + s.x * s.y * s.z;
            size = luisa::max(size >> 1u, 1u);
        }
        auto s = (size + block_size - 1u) / block_size * block_size;
        auto size_pixels = _mip_offsets[levels - 1u] + s.x * s.y * s.z;
        _data = external_buffer;
    }
}

FallbackTexture::~FallbackTexture() noexcept {
    if (!_is_external) {
        luisa::deallocate_with_allocator(_data);
    }
}

FallbackTextureView FallbackTexture::view(uint level) const noexcept {
    auto size = luisa::max(make_uint3(_size[0], _size[1], _size[2]) >> level, 1u);
    return FallbackTextureView{_data + (static_cast<size_t>(_mip_offsets[level]) << _pixel_stride_shift),
                               _dimension, size.x, size.y, size.z, storage(), _pixel_stride_shift};
}

// template<typename T>
// [[nodiscard]] inline auto texture_coord_point(Sampler::Address address, T uv, T s) noexcept {
//     switch (address) {
//         case Sampler::Address::EDGE: return luisa::clamp(uv, 0.0f, one_minus_epsilon) * s;
//         case Sampler::Address::REPEAT: return luisa::fract(uv) * s;
//         case Sampler::Address::MIRROR: {
//             uv = luisa::fmod(luisa::abs(uv), T{2.0f});
//             uv = select(2.f - uv, uv, uv < T{1.f});
//             return luisa::min(uv, one_minus_epsilon) * s;
//         }
//         case Sampler::Address::ZERO: return luisa::select(uv * s, T{65536.f}, uv < 0.f || uv >= 1.f);
//     }
//     return T{65536.f};
// }
//
// [[nodiscard]] inline auto texture_coord_linear(Sampler::Address address, float3 uv, float3 size) noexcept {
//     auto s = make_float3(size);
//     auto inv_s = 1.f / s;
//     auto c_min = texture_coord_point(address, uv - .5f * inv_s, s);
//     auto c_max = texture_coord_point(address, uv + .5f * inv_s, s);
//     return std::make_pair(luisa::min(c_min, c_max), luisa::max(c_min, c_max));
// }
//
// [[nodiscard]] inline auto texture_sample_linear(FallbackTextureView view, Sampler::Address address, float3 uvw) noexcept {
//     auto size = make_float3(view.size3d());
//     auto [st_min, st_max] = texture_coord_linear(address, uvw, size);
//     auto t = luisa::fract(st_max);
//     auto c0 = make_uint3(st_min);
//     auto c1 = make_uint3(st_max);
//     auto v000 = view.read3d<float>(make_uint3(c0.x, c0.y, c0.z));
//     auto v001 = view.read3d<float>(make_uint3(c1.x, c0.y, c0.z));
//     auto v010 = view.read3d<float>(make_uint3(c0.x, c1.y, c0.z));
//     auto v011 = view.read3d<float>(make_uint3(c1.x, c1.y, c0.z));
//     auto v100 = view.read3d<float>(make_uint3(c0.x, c0.y, c1.z));
//     auto v101 = view.read3d<float>(make_uint3(c1.x, c0.y, c1.z));
//     auto v110 = view.read3d<float>(make_uint3(c0.x, c1.y, c1.z));
//     auto v111 = view.read3d<float>(make_uint3(c1.x, c1.y, c1.z));
//     return luisa::lerp(
//         luisa::lerp(luisa::lerp(v000, v001, t.x),
//                     luisa::lerp(v010, v011, t.x), t.y),
//         luisa::lerp(luisa::lerp(v100, v101, t.x),
//                     luisa::lerp(v110, v111, t.x), t.y),
//         t.z);
// }
//
// [[nodiscard]] inline auto texture_sample_point(FallbackTextureView view, Sampler::Address address, float2 uv) noexcept {
//     auto size = make_float2(view.size2d());
//     auto c = make_uint2(texture_coord_point(address, uv, size));
//     return view.read2d<float>(c);
// }
//
// [[nodiscard]] inline auto texture_sample_point(FallbackTextureView view, Sampler::Address address, float3 uvw) noexcept {
//     auto size = make_float3(view.size3d());
//     auto c = make_uint3(texture_coord_point(address, uvw, size));
//     return view.read3d<float>(c);
// }
//
// // from PBRT-v4
// [[nodiscard]] inline auto texture_sample_ewa(FallbackTextureView view, Sampler::Address address,
//                                              float2 uv, float2 dst0, float2 dst1) noexcept {
//     auto size = make_float2(view.size2d());
//     auto st = uv * size - .5f;
//     dst0 = dst0 * size;
//     dst1 = dst1 * size;
//
//     constexpr auto sqr = [](float x) noexcept { return x * x; };
//     constexpr auto safe_sqrt = [](float x) noexcept { return luisa::select(std::sqrt(x), 0.f, x <= 0.f); };
//
//     // Find ellipse coefficients that bound EWA filter region
//     auto A = sqr(dst0.y) + sqr(dst1.y) + 1.f;
//     auto B = -2.f * (dst0.x * dst0.y + dst1.x * dst1.y);
//     auto C = sqr(dst0.x) + sqr(dst1.x) + 1.f;
//     auto inv_f = 1.f / (A * C - sqr(B) * 0.25f);
//     A *= inv_f;
//     B *= inv_f;
//     C *= inv_f;
//
//     // Compute the ellipse's $(s,t)$ bounding box in texture space
//     auto det = -sqr(B) + 4.f * A * C;
//     auto inv_det = 1.f / det;
//     auto sqrt_u = safe_sqrt(det * C);
//     auto sqrt_v = safe_sqrt(A * det);
//     auto s_min = static_cast<int>(std::ceil(st.x - 2.f * inv_det * sqrt_u));
//     auto s_max = static_cast<int>(std::floor(st.x + 2.f * inv_det * sqrt_u));
//     auto t_min = static_cast<int>(std::ceil(st.y - 2.f * inv_det * sqrt_v));
//     auto t_max = static_cast<int>(std::floor(st.y + 2.f * inv_det * sqrt_v));
//
//     // Scan over ellipse bound and evaluate quadratic equation to filter image
//     auto sum = make_float4();
//     auto sum_w = 0.f;
//     auto inv_size = 1.f / size;
//     for (auto t = t_min; t <= t_max; t++) {
//         for (auto s = s_min; s <= s_max; s++) {
//             auto ss = static_cast<float>(s) - st.x;
//             auto tt = static_cast<float>(t) - st.y;
//             // Compute squared radius and filter texel if it is inside the ellipse
//             if (auto rr = A * sqr(ss) + B * ss * tt + C * sqr(tt); rr < 1.f) {
//                 constexpr auto lut_size = static_cast<float>(detail::ewa_filter_weight_lut.size());
//                 auto index = std::clamp(rr * lut_size, 0.f, lut_size - 1.f);
//                 auto weight = detail::ewa_filter_weight_lut[static_cast<int>(index)];
//                 auto p = texture_coord_point(address, uv + make_float2(ss, tt) * inv_size, size);
//                 sum += weight * view.read2d<float>(make_uint2(p));
//                 sum_w += weight;
//             }
//         }
//     }
//     return select(sum / sum_w, make_float4(0.f), sum_w <= 0.f);
// }
//
// [[nodiscard]] inline auto texture_sample_ewa(FallbackTextureView view, Sampler::Address address,
//                                              float3 uvw, float3 ddx, float3 ddy) noexcept {
//     // FIXME: anisotropic filtering
//     return texture_sample_linear(view, address, uvw);
// }

//float4 FallbackTexture::sample2d(Sampler sampler, float2 uv) const noexcept {
//    return sampler.filter() == Sampler::Filter::POINT ?
//               texture_sample_point(view(0), sampler.address(), uv) :
//               texture_sample_linear(view(0), sampler.address(), uv);
//}

//float4 FallbackTexture::sample3d(Sampler sampler, float3 uvw) const noexcept {
//    return sampler.filter() == Sampler::Filter::POINT ?
//               texture_sample_point(view(0), sampler.address(), uvw) :
//               texture_sample_linear(view(0), sampler.address(), uvw);
//}

//float4 FallbackTexture::sample2d(Sampler sampler, float2 uv, float lod) const noexcept {
//    auto filter = sampler.filter();
//    if (lod <= 0.f || _mip_levels == 0u ||
//        filter == Sampler::Filter::POINT) {
//        return sample2d(sampler, uv);
//    }
//    auto level0 = std::min(static_cast<uint32_t>(lod),
//                           _mip_levels - 1u);
//    auto v0 = texture_sample_linear(
//        view(level0), sampler.address(), uv);
//    if (level0 == _mip_levels - 1u ||
//        filter == Sampler::Filter::LINEAR_POINT) {
//        return v0;
//    }
//    auto v1 = texture_sample_linear(
//        view(level0 + 1u), sampler.address(), uv);
//    return luisa::lerp(v0, v1, luisa::fract(lod));
//}

//float4 FallbackTexture::sample3d(Sampler sampler, float3 uvw, float lod) const noexcept {
//    auto filter = sampler.filter();
//    if (lod <= 0.f || _mip_levels == 0u ||
//        filter == Sampler::Filter::POINT) {
//        return sample3d(sampler, uvw);
//    }
//    auto level0 = std::min(static_cast<uint32_t>(lod),
//                           _mip_levels - 1u);
//    auto v0 = texture_sample_linear(
//        view(level0), sampler.address(), uvw);
//    if (level0 == _mip_levels - 1u ||
//        filter == Sampler::Filter::LINEAR_POINT) {
//        return v0;
//    }
//    auto v1 = texture_sample_linear(
//        view(level0 + 1u), sampler.address(), uvw);
//    return luisa::lerp(v0, v1, luisa::fract(lod));
//}

//float4 FallbackTexture::sample2d(Sampler sampler, float2 uv, float2 dpdx, float2 dpdy) const noexcept {
//    constexpr auto ll = [](float2 v) noexcept { return dot(v, v); };
//    if (all(dpdx == 0.f) || all(dpdy == 0.f)) { return sample2d(sampler, uv); }
//    if (sampler.filter() != Sampler::Filter::ANISOTROPIC) {
//        auto s = make_float2(_size[0], _size[1]);
//        auto level = .5f * std::log2(std::max(ll(dpdx * s), ll(dpdy * s)));
//        return sample2d(sampler, uv, level);
//    }
//    auto longer = length(dpdx);
//    auto shorter = length(dpdy);
//    if (longer < shorter) {
//        std::swap(longer, shorter);
//        std::swap(dpdx, dpdy);
//    }
//    constexpr auto max_anisotropy = 16.f;
//    if (auto s = shorter * max_anisotropy; s < longer) {
//        auto scale = longer / s;
//        dpdy *= scale;
//        shorter *= scale;
//    }
//    auto last_level = static_cast<float>(_mip_levels - 1u);
//    auto level = std::clamp(last_level + std::log2(shorter), 0.f, last_level);
//    auto level_uint = static_cast<uint>(level);
//    auto v0 = texture_sample_ewa(view(level_uint), sampler.address(), uv, dpdx, dpdy);
//    if (level == 0.f || level == last_level) { return v0; }
//    auto v1 = texture_sample_ewa(view(level_uint + 1u), sampler.address(), uv, dpdx, dpdy);
//    return luisa::lerp(v0, v1, luisa::fract(level));
//}
//
//float4 FallbackTexture::sample3d(Sampler sampler, float3 uvw, float3 dpdx, float3 dpdy) const noexcept {
//    constexpr auto ll = [](float3 v) noexcept { return dot(v, v); };
//    if (all(dpdx == 0.f) || all(dpdy == 0.f)) { return sample3d(sampler, uvw); }
//    if (sampler.filter() != Sampler::Filter::ANISOTROPIC) {
//        auto s = make_float3(_size[0], _size[1], _size[2]);
//        auto level = .5f * std::log2(std::max(ll(dpdx * s), ll(dpdy * s)));
//        return sample3d(sampler, uvw, level);
//    }
//    auto longer = length(dpdx);
//    auto shorter = length(dpdy);
//    if (longer < shorter) {
//        std::swap(longer, shorter);
//        std::swap(dpdx, dpdy);
//    }
//    constexpr auto max_anisotropy = 16.f;
//    if (auto s = shorter * max_anisotropy; s < longer) {
//        auto scale = longer / s;
//        dpdy *= scale;
//        shorter *= scale;
//    }
//    auto last_level = static_cast<float>(_mip_levels - 1u);
//    auto level = std::clamp(last_level + std::log2(shorter), 0.f, last_level);
//    auto level_uint = static_cast<uint>(level);
//    auto v0 = texture_sample_ewa(view(level_uint), sampler.address(), uvw, dpdx, dpdy);
//    if (level == 0.f || level == last_level) { return v0; }
//    auto v1 = texture_sample_ewa(view(level_uint + 1u), sampler.address(), uvw, dpdx, dpdy);
//    return luisa::lerp(v0, v1, luisa::fract(level));
//}

}// namespace luisa::compute::fallback
