#include "simd_texture.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <type_traits>

#include <luisa/core/logging.h>

#include "../../common/env_flag.h"

namespace {

template<size_t Width>
[[nodiscard]] constexpr uint64_t full_lane_mask() noexcept {
    static_assert(Width < 64u);
    return (uint64_t{1u} << Width) - 1u;
}

[[nodiscard]] constexpr uint64_t lane_mask(
    uint32_t lane_count) noexcept {
    return lane_count >= 64u ? ~uint64_t{0u} :
                               (uint64_t{1u} << lane_count) - 1u;
}

using Sampler = luisa::compute::Sampler;
using Address = Sampler::Address;
using Filter = Sampler::Filter;

template<Address Mode>
[[nodiscard]] LUISA_FORCE_INLINE float texture_coordinate_point_unchecked(
    float uv, float size) noexcept {
    constexpr auto invalid_coordinate = 65536.0f;
    if constexpr (Mode == Address::EDGE) {
        uv = std::clamp(
            uv, 0.0f, luisa::one_minus_epsilon);
    } else if constexpr (Mode == Address::REPEAT) {
        uv -= std::floor(uv);
    } else if constexpr (Mode == Address::MIRROR) {
        uv = std::abs(uv);
        // fmod(x, 2) compiled to a serialized x87 fprem sequence on the
        // measured x86 host. The quotient/remainder identity is equivalent
        // for finite non-negative float inputs and vectorizes to ordinary
        // floor/multiply/subtract operations.
        uv -= std::floor(uv * 0.5f) * 2.0f;
        uv = uv < 1.0f ? uv : 2.0f - uv;
        uv = std::min(uv, luisa::one_minus_epsilon);
    } else {
        static_assert(Mode == Address::ZERO);
        if (uv < 0.0f || uv >= 1.0f) {
            return invalid_coordinate;
        }
    }
    return uv * size;
}

template<Address Mode>
[[nodiscard]] LUISA_FORCE_INLINE float texture_coordinate_point(
    float uv, float size) noexcept {
    return std::isfinite(uv) && size > 0.0f ?
               texture_coordinate_point_unchecked<Mode>(uv, size) :
               65536.0f;
}

template<uint32_t Dimension, Address Mode>
[[nodiscard]] LUISA_FORCE_INLINE luisa::float4 sample_point(
    luisa::compute::fallback::FallbackTextureView view,
    float u, float v, float w) noexcept {
    static_assert(Dimension == 2u || Dimension == 3u);
    auto size = view.size3d();
    auto x = static_cast<uint32_t>(
        texture_coordinate_point<Mode>(
            u, static_cast<float>(size.x)));
    auto y = static_cast<uint32_t>(
        texture_coordinate_point<Mode>(
            v, static_cast<float>(size.y)));
    if constexpr (Dimension == 2u) {
        return view.read2d<float>(luisa::make_uint2(x, y));
    } else {
        auto z = static_cast<uint32_t>(
            texture_coordinate_point<Mode>(
                w, static_cast<float>(size.z)));
        return view.read3d<float>(luisa::make_uint3(x, y, z));
    }
}

struct LinearCoordinate {
    uint32_t lo;
    uint32_t hi;
    float t;
};

template<Address Mode>
[[nodiscard]] LUISA_FORCE_INLINE LinearCoordinate
texture_coordinate_linear_precomputed(
    float uv, float size, float half_inverse_size) noexcept {
    constexpr auto invalid_coordinate = 65536u;
    if (!(size > 0.0f) || !std::isfinite(size) ||
        !std::isfinite(uv) || !std::isfinite(half_inverse_size)) {
        return {65536u, 65536u, 0.0f};
    }
    if constexpr (Mode == Address::ZERO) {
        auto position = uv * size - 0.5f;
        if (!std::isfinite(position)) {
            return {invalid_coordinate, invalid_coordinate, 0.0f};
        }
        auto lower = std::floor(position);
        auto upper = lower + 1.0f;
        auto encode = [size](float index) noexcept {
            return index >= 0.0f && index < size ?
                       static_cast<uint32_t>(index) :
                       invalid_coordinate;
        };
        return {encode(lower), encode(upper), position - lower};
    }
    auto a = texture_coordinate_point_unchecked<Mode>(
        uv - half_inverse_size, size);
    auto b = texture_coordinate_point_unchecked<Mode>(
        uv + half_inverse_size, size);
    if constexpr (Mode == Address::MIRROR) {
        auto lo = std::min(a, b);
        auto hi = std::max(a, b);
        return {
            static_cast<uint32_t>(lo),
            static_cast<uint32_t>(hi),
            hi - std::floor(hi),
        };
    }
    return {
        static_cast<uint32_t>(a),
        static_cast<uint32_t>(b),
        b - std::floor(b),
    };
}

template<Address Mode>
[[nodiscard]] LUISA_FORCE_INLINE LinearCoordinate
texture_coordinate_linear(float uv, float size) noexcept {
    return texture_coordinate_linear_precomputed<Mode>(
        uv, size, 0.5f / size);
}

template<uint32_t Dimension, Address Mode>
[[nodiscard]] LUISA_FORCE_INLINE luisa::float4 sample_linear(
    luisa::compute::fallback::FallbackTextureView view,
    float u, float v, float w) noexcept {
    static_assert(Dimension == 2u || Dimension == 3u);
    auto size = view.size3d();
    auto x = texture_coordinate_linear<Mode>(
        u, static_cast<float>(size.x));
    auto y = texture_coordinate_linear<Mode>(
        v, static_cast<float>(size.y));
    auto lerp = [](luisa::float4 a, luisa::float4 b, float t) noexcept {
        return a + (b - a) * t;
    };
    if constexpr (Dimension == 2u) {
        auto v00 = view.read2d<float>(luisa::make_uint2(x.lo, y.lo));
        auto v01 = view.read2d<float>(luisa::make_uint2(x.hi, y.lo));
        auto v10 = view.read2d<float>(luisa::make_uint2(x.lo, y.hi));
        auto v11 = view.read2d<float>(luisa::make_uint2(x.hi, y.hi));
        return lerp(
            lerp(v00, v01, x.t),
            lerp(v10, v11, x.t), y.t);
    } else {
        auto z = texture_coordinate_linear<Mode>(
            w, static_cast<float>(size.z));
        auto v000 = view.read3d<float>(
            luisa::make_uint3(x.lo, y.lo, z.lo));
        auto v001 = view.read3d<float>(
            luisa::make_uint3(x.hi, y.lo, z.lo));
        auto v010 = view.read3d<float>(
            luisa::make_uint3(x.lo, y.hi, z.lo));
        auto v011 = view.read3d<float>(
            luisa::make_uint3(x.hi, y.hi, z.lo));
        auto v100 = view.read3d<float>(
            luisa::make_uint3(x.lo, y.lo, z.hi));
        auto v101 = view.read3d<float>(
            luisa::make_uint3(x.hi, y.lo, z.hi));
        auto v110 = view.read3d<float>(
            luisa::make_uint3(x.lo, y.hi, z.hi));
        auto v111 = view.read3d<float>(
            luisa::make_uint3(x.hi, y.hi, z.hi));
        return lerp(
            lerp(lerp(v000, v001, x.t),
                 lerp(v010, v011, x.t), y.t),
            lerp(lerp(v100, v101, x.t),
                 lerp(v110, v111, x.t), y.t),
            z.t);
    }
}

template<Filter FilterMode, Address AddressMode>
[[nodiscard]] LUISA_FORCE_INLINE float sample_byte1_2d(
    const uint8_t *data, luisa::uint2 size,
    float u, float v,
    float half_inverse_width,
    float half_inverse_height) noexcept {
    static_assert(
        FilterMode == Filter::POINT ||
        FilterMode == Filter::LINEAR_POINT ||
        FilterMode == Filter::LINEAR_LINEAR ||
        FilterMode == Filter::ANISOTROPIC);
    auto read = [=](uint32_t x, uint32_t y) noexcept {
        if (x >= size.x || y >= size.y) [[unlikely]] { return 0.0f; }
        auto index = static_cast<size_t>(y) * size.x + x;
        return static_cast<float>(data[index]) * (1.0f / 255.0f);
    };
    if constexpr (FilterMode == Filter::POINT) {
        auto x = static_cast<uint32_t>(
            texture_coordinate_point<AddressMode>(
                u, static_cast<float>(size.x)));
        auto y = static_cast<uint32_t>(
            texture_coordinate_point<AddressMode>(
                v, static_cast<float>(size.y)));
        return read(x, y);
    } else {
        auto x = texture_coordinate_linear_precomputed<AddressMode>(
            u, static_cast<float>(size.x), half_inverse_width);
        auto y = texture_coordinate_linear_precomputed<AddressMode>(
            v, static_cast<float>(size.y), half_inverse_height);
        auto v00 = read(x.lo, y.lo);
        auto v01 = read(x.hi, y.lo);
        auto v10 = read(x.lo, y.hi);
        auto v11 = read(x.hi, y.hi);
        auto row0 = v00 + (v01 - v00) * x.t;
        auto row1 = v10 + (v11 - v10) * x.t;
        return row0 + (row1 - row0) * y.t;
    }
}

template<size_t Width, Filter FilterMode, Address AddressMode>
LUISA_FORCE_INLINE void sample_byte1_2d_lane(
    const uint8_t *data, luisa::uint2 size,
    const float *u, const float *v, float *values,
    float half_inverse_width, float half_inverse_height,
    uint32_t lane) noexcept {
    auto value = sample_byte1_2d<FilterMode, AddressMode>(
        data, size, u[lane], v[lane],
        half_inverse_width, half_inverse_height);
    values[lane] = value;
    values[Width + lane] = 0.0f;
    values[2u * Width + lane] = 0.0f;
    values[3u * Width + lane] = 0.0f;
}

template<size_t Width, Filter FilterMode, Address AddressMode>
LUISA_FORCE_INLINE void sample_byte1_2d_packet(
    luisa::compute::fallback::FallbackTextureView view,
    uint64_t active_mask_bits,
    const float *u, const float *v, float *values) noexcept {
    auto size = view.size2d();
    auto *data = reinterpret_cast<const uint8_t *>(view.data());
    auto half_inverse_width = 0.5f / static_cast<float>(size.x);
    auto half_inverse_height = 0.5f / static_cast<float>(size.y);
    if (active_mask_bits == full_lane_mask<Width>()) {
#if defined(__GNUC__)
#pragma GCC ivdep
#endif
        for (auto lane = 0u; lane < Width; lane++) {
            sample_byte1_2d_lane<Width, FilterMode, AddressMode>(
                data, size, u, v, values,
                half_inverse_width, half_inverse_height, lane);
        }
    } else {
        for (auto remaining = active_mask_bits; remaining != 0u;
             remaining &= remaining - 1u) {
            auto lane = static_cast<uint32_t>(
                std::countr_zero(remaining));
            sample_byte1_2d_lane<Width, FilterMode, AddressMode>(
                data, size, u, v, values,
                half_inverse_width, half_inverse_height, lane);
        }
    }
}

template<uint32_t Dimension, Filter FilterMode, Address AddressMode>
[[nodiscard]] LUISA_FORCE_INLINE luisa::float4 sample_spatial(
    luisa::compute::fallback::FallbackTextureView view,
    float u, float v, float w) noexcept {
    if constexpr (FilterMode == Filter::POINT) {
        return sample_point<Dimension, AddressMode>(view, u, v, w);
    } else {
        return sample_linear<Dimension, AddressMode>(view, u, v, w);
    }
}

template<uint32_t Dimension, Filter FilterMode, Address AddressMode>
[[nodiscard]] LUISA_FORCE_INLINE luisa::float4 sample_texture(
    luisa::compute::simd::SIMDTexture *texture,
    uint32_t base_level, float u, float v, float w,
    float level) noexcept {
    auto total_mip_levels = texture->mip_levels();
    if (base_level >= total_mip_levels) { return {}; }
    auto mip_levels = total_mip_levels - base_level;
    if (std::isnan(level) || level < 0.0f) { level = 0.0f; }
    if (!std::isfinite(level)) {
        level = static_cast<float>(mip_levels - 1u);
    }
    auto spatial = [&](uint32_t mip) noexcept {
        auto view = texture->view(base_level + mip);
        return sample_spatial<Dimension, FilterMode, AddressMode>(
            view, u, v, w);
    };
    if (level <= 0.0f || mip_levels == 1u) {
        return spatial(0u);
    }
    auto last_level = mip_levels - 1u;
    if (level >= static_cast<float>(last_level)) {
        return spatial(last_level);
    }
    if constexpr (FilterMode == Filter::POINT ||
                  FilterMode == Filter::LINEAR_POINT) {
        auto nearest_level = static_cast<uint32_t>(
            std::floor(level + 0.5f));
        return spatial(std::min(nearest_level, last_level));
    }
    auto level0 = static_cast<uint32_t>(level);
    auto value0 = spatial(level0);
    auto value1 = spatial(level0 + 1u);
    auto t = level - std::floor(level);
    return value0 + (value1 - value0) * t;
}

template<size_t Width, uint32_t Dimension,
         Filter FilterMode, Address AddressMode>
LUISA_FORCE_INLINE void sample_packet_fixed_specialized(
    luisa::compute::simd::SIMDTexture *texture,
    uint32_t base_level, uint64_t active_mask_bits, const float *u,
    const float *v, const float *w, const float *levels,
    float *values) noexcept {
    active_mask_bits &= full_lane_mask<Width>();
    if (active_mask_bits == 0u) { return; }
    auto store_pixel = [&](uint32_t lane, luisa::float4 pixel) noexcept {
        for (auto component = 0u; component < 4u; component++) {
            values[component * Width + lane] = pixel[component];
        }
    };
    auto visit_lanes = [&](auto &&sample_lane) noexcept {
        if (active_mask_bits == full_lane_mask<Width>()) {
            for (auto lane = 0u; lane < Width; lane++) {
                sample_lane(lane);
            }
        } else {
            for (auto remaining = active_mask_bits; remaining != 0u;
                 remaining &= remaining - 1u) {
                auto lane = static_cast<uint32_t>(
                    std::countr_zero(remaining));
                sample_lane(lane);
            }
        }
    };

    // Stored-sampler calls without an explicit level are the common graphics
    // path. The texture, mip and pixel format are packet-invariant after the
    // bindless cohort has been grouped, so resolve them once rather than once
    // per lane. BYTE1 additionally bypasses the generic pixel-format switch
    // for each of the four bilinear taps.
    if (levels == nullptr && base_level < texture->mip_levels()) {
        auto view = texture->view(base_level);
        if constexpr (Dimension == 2u) {
            if (view.storage() == luisa::compute::PixelStorage::BYTE1) {
                sample_byte1_2d_packet<
                    Width, FilterMode, AddressMode>(
                    view, active_mask_bits, u, v, values);
                return;
            }
        }
        visit_lanes([&](uint32_t lane) noexcept {
            store_pixel(
                lane,
                sample_spatial<Dimension, FilterMode, AddressMode>(
                    view, u[lane], v[lane], w[lane]));
        });
        return;
    }

    visit_lanes([&](uint32_t lane) noexcept {
        store_pixel(
            lane,
            sample_texture<Dimension, FilterMode, AddressMode>(
                texture, base_level, u[lane], v[lane], w[lane],
                levels == nullptr ? 0.0f : levels[lane]));
    });
}

template<size_t Width, uint32_t Dimension, Filter FilterMode>
void sample_packet_fixed_address(
    luisa::compute::simd::SIMDTexture *texture,
    uint32_t base_level, Address address,
    uint64_t active_mask_bits, const float *u,
    const float *v, const float *w, const float *levels,
    float *values) noexcept {
    switch (address) {
        case Address::EDGE:
            sample_packet_fixed_specialized<
                Width, Dimension, FilterMode, Address::EDGE>(
                texture, base_level, active_mask_bits,
                u, v, w, levels, values);
            break;
        case Address::REPEAT:
            sample_packet_fixed_specialized<
                Width, Dimension, FilterMode, Address::REPEAT>(
                texture, base_level, active_mask_bits,
                u, v, w, levels, values);
            break;
        case Address::MIRROR:
            sample_packet_fixed_specialized<
                Width, Dimension, FilterMode, Address::MIRROR>(
                texture, base_level, active_mask_bits,
                u, v, w, levels, values);
            break;
        case Address::ZERO:
            sample_packet_fixed_specialized<
                Width, Dimension, FilterMode, Address::ZERO>(
                texture, base_level, active_mask_bits,
                u, v, w, levels, values);
            break;
    }
}

template<size_t Width, uint32_t Dimension>
void sample_packet_fixed_dimension(
    luisa::compute::simd::SIMDTexture *texture,
    Sampler sampler, uint32_t base_level, uint64_t active_mask_bits,
    const float *u, const float *v, const float *w,
    const float *levels, float *values) noexcept {
    switch (sampler.filter()) {
        case Filter::POINT:
            sample_packet_fixed_address<Width, Dimension, Filter::POINT>(
                texture, base_level, sampler.address(), active_mask_bits,
                u, v, w, levels, values);
            break;
        case Filter::LINEAR_POINT:
            sample_packet_fixed_address<
                Width, Dimension, Filter::LINEAR_POINT>(
                texture, base_level, sampler.address(), active_mask_bits,
                u, v, w, levels, values);
            break;
        case Filter::LINEAR_LINEAR:
            sample_packet_fixed_address<
                Width, Dimension, Filter::LINEAR_LINEAR>(
                texture, base_level, sampler.address(), active_mask_bits,
                u, v, w, levels, values);
            break;
        case Filter::ANISOTROPIC:
            sample_packet_fixed_address<
                Width, Dimension, Filter::ANISOTROPIC>(
                texture, base_level, sampler.address(), active_mask_bits,
                u, v, w, levels, values);
            break;
    }
}

template<size_t Width>
void sample_packet_fixed(
    luisa::compute::simd::SIMDTexture *texture,
    Sampler sampler, uint32_t base_level, uint64_t active_mask_bits,
    const float *u, const float *v, const float *w,
    const float *levels, float *values) noexcept {
    if (texture->dimension() == 2u) {
        sample_packet_fixed_dimension<Width, 2u>(
            texture, sampler, base_level, active_mask_bits,
            u, v, w, levels, values);
    } else {
        sample_packet_fixed_dimension<Width, 3u>(
            texture, sampler, base_level, active_mask_bits,
            u, v, w, levels, values);
    }
}

void sample_packet_dynamic(
    luisa::compute::simd::SIMDTexture *texture,
    Sampler sampler, uint32_t base_level, uint32_t lane_count,
    uint64_t active_mask_bits, const float *u,
    const float *v, const float *w, const float *levels,
    float *values) noexcept {
    active_mask_bits &= lane_mask(lane_count);
    for (auto remaining = active_mask_bits; remaining != 0u;
         remaining &= remaining - 1u) {
        auto lane = static_cast<uint32_t>(std::countr_zero(remaining));
        std::array<float, 4u> pixel{};
        sample_packet_fixed<1u>(
            texture, sampler, base_level, 1u,
            u + lane, v + lane, w + lane,
            levels == nullptr ? nullptr : levels + lane,
            pixel.data());
        for (auto component = 0u; component < 4u; component++) {
            values[component * lane_count + lane] = pixel[component];
        }
    }
}

template<typename T>
[[nodiscard]] auto read_pixel(
    luisa::compute::simd::SIMDTexture *texture,
    luisa::compute::fallback::FallbackTextureView view,
    uint32_t x, uint32_t y, uint32_t z) noexcept {
    return texture->dimension() == 2u ?
               view.read2d<T>(luisa::make_uint2(x, y)) :
               view.read3d<T>(luisa::make_uint3(x, y, z));
}

template<typename T>
[[nodiscard]] constexpr auto native_four_channel_storage() noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return luisa::compute::PixelStorage::FLOAT4;
    } else {
        static_assert(std::is_same_v<T, uint32_t>);
        return luisa::compute::PixelStorage::INT4;
    }
}

template<typename T, size_t Width>
[[nodiscard]] bool read_contiguous_native_four_channel(
    luisa::compute::simd::SIMDTexture *texture,
    luisa::compute::fallback::FallbackTextureView view,
    uint32_t x, uint32_t y, T *values) noexcept {
    static_assert(sizeof(luisa::Vector<T, 4u>) == sizeof(T) * 4u);
    if (!texture->contiguous_packets_enabled() ||
        view.storage() != native_four_channel_storage<T>()) {
        return false;
    }
    auto size = view.size2d();
    if (x >= size.x || y >= size.y || Width > size.x - x) {
        return false;
    }
    auto pixel_index = static_cast<size_t>(y) * size.x + x;
    std::array<luisa::Vector<T, 4u>, Width> pixels{};
    std::memcpy(
        pixels.data(),
        view.data() + pixel_index * sizeof(luisa::Vector<T, 4u>),
        sizeof(pixels));
    for (auto component = 0u; component < 4u; component++) {
        for (auto lane = 0u; lane < Width; lane++) {
            values[component * Width + lane] = pixels[lane][component];
        }
    }
    return true;
}

template<typename T, size_t Width>
[[nodiscard]] bool write_contiguous_native_four_channel(
    luisa::compute::simd::SIMDTexture *texture,
    luisa::compute::fallback::FallbackTextureView view,
    uint32_t x, uint32_t y, const T *values) noexcept {
    static_assert(sizeof(luisa::Vector<T, 4u>) == sizeof(T) * 4u);
    if (!texture->contiguous_packets_enabled() ||
        view.storage() != native_four_channel_storage<T>()) {
        return false;
    }
    auto size = view.size2d();
    if (x >= size.x || y >= size.y || Width > size.x - x) {
        return false;
    }
    auto pixel_index = static_cast<size_t>(y) * size.x + x;
    std::array<luisa::Vector<T, 4u>, Width> pixels{};
    for (auto lane = 0u; lane < Width; lane++) {
        for (auto component = 0u; component < 4u; component++) {
            pixels[lane][component] = values[component * Width + lane];
        }
    }
    std::memcpy(
        view.data() + pixel_index * sizeof(luisa::Vector<T, 4u>),
        pixels.data(), sizeof(pixels));
    return true;
}

template<typename T, size_t Width>
void read_packet_fixed(
    luisa::compute::simd::SIMDTexture *texture, uint32_t level,
    uint64_t active_mask_bits, const uint32_t *x,
    const uint32_t *y, const uint32_t *z, T *values) noexcept {
    active_mask_bits &= full_lane_mask<Width>();
    if (active_mask_bits == 0u) { return; }
    auto view = texture->view(level);
    auto contiguous = active_mask_bits == full_lane_mask<Width>() &&
                      texture->dimension() == 2u;
    for (auto lane = 1u; lane < Width && contiguous; lane++) {
        contiguous = x[lane] == x[0u] + lane &&
                     y[lane] == y[0u] && z[lane] == z[0u];
    }
    if (contiguous && read_contiguous_native_four_channel<T, Width>(
                          texture, view, x[0u], y[0u], values)) {
        return;
    }
    auto first_lane = static_cast<uint32_t>(
        std::countr_zero(active_mask_bits));
    auto broadcast = true;
    for (auto remaining = active_mask_bits &
                          (active_mask_bits - 1u);
         remaining != 0u; remaining &= remaining - 1u) {
        auto lane = static_cast<uint32_t>(std::countr_zero(remaining));
        broadcast &= x[lane] == x[first_lane] &&
                     y[lane] == y[first_lane] &&
                     z[lane] == z[first_lane];
    }
    if (broadcast) {
        auto pixel = read_pixel<T>(
            texture, view, x[first_lane], y[first_lane], z[first_lane]);
        for (auto remaining = active_mask_bits; remaining != 0u;
             remaining &= remaining - 1u) {
            auto lane = static_cast<uint32_t>(std::countr_zero(remaining));
            for (auto component = 0u; component < 4u; component++) {
                values[component * Width + lane] = pixel[component];
            }
        }
        return;
    }
    if (contiguous) {
        for (auto lane = 0u; lane < Width; lane++) {
            auto pixel = view.read2d<T>(luisa::make_uint2(x[lane], y[lane]));
            for (auto component = 0u; component < 4u; component++) {
                values[component * Width + lane] = pixel[component];
            }
        }
        return;
    }
    for (auto remaining = active_mask_bits; remaining != 0u;
         remaining &= remaining - 1u) {
        auto lane = static_cast<uint32_t>(std::countr_zero(remaining));
        auto pixel = read_pixel<T>(
            texture, view, x[lane], y[lane], z[lane]);
        for (auto component = 0u; component < 4u; component++) {
            values[component * Width + lane] = pixel[component];
        }
    }
}

template<typename T>
void read_packet_dynamic(
    luisa::compute::simd::SIMDTexture *texture, uint32_t level,
    uint32_t lane_count, uint64_t active_mask_bits,
    const uint32_t *x, const uint32_t *y, const uint32_t *z,
    T *values) noexcept {
    active_mask_bits &= lane_mask(lane_count);
    if (active_mask_bits == 0u) { return; }
    auto view = texture->view(level);
    for (auto remaining = active_mask_bits; remaining != 0u;
         remaining &= remaining - 1u) {
        auto lane = static_cast<uint32_t>(std::countr_zero(remaining));
        auto pixel = read_pixel<T>(
            texture, view, x[lane], y[lane], z[lane]);
        for (auto component = 0u; component < 4u; component++) {
            values[component * lane_count + lane] = pixel[component];
        }
    }
}

template<typename T>
void read_packet(
    luisa::compute::simd::SIMDTexture *texture, uint32_t level,
    uint32_t lane_count, uint64_t active_mask_bits,
    const uint32_t *x, const uint32_t *y, const uint32_t *z,
    T *values) noexcept {
    switch (lane_count) {
        case 1u:
            read_packet_fixed<T, 1u>(
                texture, level, active_mask_bits, x, y, z, values);
            break;
        case 2u:
            read_packet_fixed<T, 2u>(
                texture, level, active_mask_bits, x, y, z, values);
            break;
        case 4u:
            read_packet_fixed<T, 4u>(
                texture, level, active_mask_bits, x, y, z, values);
            break;
        case 8u:
            read_packet_fixed<T, 8u>(
                texture, level, active_mask_bits, x, y, z, values);
            break;
        case 16u:
            read_packet_fixed<T, 16u>(
                texture, level, active_mask_bits, x, y, z, values);
            break;
        default:
            read_packet_dynamic(
                texture, level, lane_count, active_mask_bits,
                x, y, z, values);
            break;
    }
}

template<typename T, size_t Width>
void write_packet_fixed(
    luisa::compute::simd::SIMDTexture *texture, uint32_t level,
    uint64_t active_mask_bits, const uint32_t *x,
    const uint32_t *y, const uint32_t *z, const T *values) noexcept {
    active_mask_bits &= full_lane_mask<Width>();
    if (active_mask_bits == 0u) { return; }
    auto view = texture->view(level);
    auto write_lane = [&](uint32_t lane) noexcept {
        auto pixel = luisa::Vector<T, 4u>{
            values[lane], values[Width + lane],
            values[2u * Width + lane], values[3u * Width + lane]};
        if (texture->dimension() == 2u) {
            view.write2d(luisa::make_uint2(x[lane], y[lane]), pixel);
        } else {
            view.write3d(
                luisa::make_uint3(x[lane], y[lane], z[lane]), pixel);
        }
    };
    auto contiguous = active_mask_bits == full_lane_mask<Width>() &&
                      texture->dimension() == 2u;
    for (auto lane = 1u; lane < Width && contiguous; lane++) {
        contiguous = x[lane] == x[0u] + lane &&
                     y[lane] == y[0u] && z[lane] == z[0u];
    }
    if (contiguous) {
        if (write_contiguous_native_four_channel<T, Width>(
                texture, view, x[0u], y[0u], values)) {
            return;
        }
        for (auto lane = 0u; lane < Width; lane++) { write_lane(lane); }
        return;
    }
    for (auto remaining = active_mask_bits; remaining != 0u;
         remaining &= remaining - 1u) {
        write_lane(static_cast<uint32_t>(std::countr_zero(remaining)));
    }
}

template<typename T>
void write_packet_dynamic(
    luisa::compute::simd::SIMDTexture *texture, uint32_t level,
    uint32_t lane_count, uint64_t active_mask_bits,
    const uint32_t *x, const uint32_t *y, const uint32_t *z,
    const T *values) noexcept {
    active_mask_bits &= lane_mask(lane_count);
    if (active_mask_bits == 0u) { return; }
    auto view = texture->view(level);
    for (auto remaining = active_mask_bits; remaining != 0u;
         remaining &= remaining - 1u) {
        auto lane = static_cast<uint32_t>(std::countr_zero(remaining));
        auto pixel = luisa::Vector<T, 4u>{
            values[lane], values[lane_count + lane],
            values[2u * lane_count + lane],
            values[3u * lane_count + lane]};
        if (texture->dimension() == 2u) {
            view.write2d(luisa::make_uint2(x[lane], y[lane]), pixel);
        } else {
            view.write3d(
                luisa::make_uint3(x[lane], y[lane], z[lane]), pixel);
        }
    }
}

template<typename T>
void write_packet(
    luisa::compute::simd::SIMDTexture *texture, uint32_t level,
    uint32_t lane_count, uint64_t active_mask_bits,
    const uint32_t *x, const uint32_t *y, const uint32_t *z,
    const T *values) noexcept {
    switch (lane_count) {
        case 1u:
            write_packet_fixed<T, 1u>(
                texture, level, active_mask_bits, x, y, z, values);
            break;
        case 2u:
            write_packet_fixed<T, 2u>(
                texture, level, active_mask_bits, x, y, z, values);
            break;
        case 4u:
            write_packet_fixed<T, 4u>(
                texture, level, active_mask_bits, x, y, z, values);
            break;
        case 8u:
            write_packet_fixed<T, 8u>(
                texture, level, active_mask_bits, x, y, z, values);
            break;
        case 16u:
            write_packet_fixed<T, 16u>(
                texture, level, active_mask_bits, x, y, z, values);
            break;
        default:
            write_packet_dynamic(
                texture, level, lane_count, active_mask_bits,
                x, y, z, values);
            break;
    }
}

}// namespace

namespace luisa::compute::simd {

SIMDTexture::SIMDTexture(
    PixelStorage storage, uint dimension, uint3 size,
    uint mip_levels) noexcept
    : _texture{storage, dimension, size, mip_levels},
      _dimension{dimension},
      _enable_contiguous_packets{
          !detail::env_flag(
              "LUISA_SIMD_DISABLE_CONTIGUOUS_TEXTURE_PACKETS")} {}

SIMDTexture::SIMDTexture(
    PixelStorage storage, uint dimension, uint3 size,
    uint mip_levels, std::byte *external_memory) noexcept
    : _texture{storage, dimension, size, mip_levels,
               external_memory},
      _dimension{dimension},
      _enable_contiguous_packets{
          !detail::env_flag(
              "LUISA_SIMD_DISABLE_CONTIGUOUS_TEXTURE_PACKETS")} {}

[[nodiscard]] uint3 SIMDTexture::size(uint32_t level) const noexcept {
    auto base_size = view(0u).size3d();
    if (level >= 32u) { return make_uint3(1u); }
    return luisa::max(base_size >> level, 1u);
}

void SIMDTexture::_read_float(
    void *texture, uint32_t level, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *x,
    const uint32_t *y, const uint32_t *z, void *values) noexcept {
    auto *self = static_cast<SIMDTexture *>(texture);
    if (lane_count == 1u) {
        if ((active_mask_bits & 1u) != 0u) {
            auto pixel = read_pixel<float>(
                self, self->view(level), x[0u], y[0u], z[0u]);
            auto *output = static_cast<float *>(values);
            for (auto component = 0u; component < 4u; component++) {
                output[component] = pixel[component];
            }
        }
        return;
    }
    read_packet(
        self, level, lane_count,
        active_mask_bits, x, y, z, static_cast<float *>(values));
}

void SIMDTexture::_read_uint(
    void *texture, uint32_t level, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *x,
    const uint32_t *y, const uint32_t *z, void *values) noexcept {
    auto *self = static_cast<SIMDTexture *>(texture);
    if (lane_count == 1u) {
        if ((active_mask_bits & 1u) != 0u) {
            auto pixel = read_pixel<uint>(
                self, self->view(level), x[0u], y[0u], z[0u]);
            auto *output = static_cast<uint *>(values);
            for (auto component = 0u; component < 4u; component++) {
                output[component] = pixel[component];
            }
        }
        return;
    }
    read_packet(
        self, level, lane_count,
        active_mask_bits, x, y, z, static_cast<uint *>(values));
}

void SIMDTexture::_write_float(
    void *texture, uint32_t level, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *x,
    const uint32_t *y, const uint32_t *z,
    const void *values) noexcept {
    auto *self = static_cast<SIMDTexture *>(texture);
    if (lane_count == 1u) {
        if ((active_mask_bits & 1u) != 0u) {
            auto *input = static_cast<const float *>(values);
            auto pixel = make_float4(
                input[0u], input[1u], input[2u], input[3u]);
            auto view = self->view(level);
            if (self->dimension() == 2u) {
                view.write2d(make_uint2(x[0u], y[0u]), pixel);
            } else {
                view.write3d(make_uint3(x[0u], y[0u], z[0u]), pixel);
            }
        }
        return;
    }
    write_packet(
        self, level, lane_count,
        active_mask_bits, x, y, z, static_cast<const float *>(values));
}

void SIMDTexture::_write_uint(
    void *texture, uint32_t level, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *x,
    const uint32_t *y, const uint32_t *z,
    const void *values) noexcept {
    auto *self = static_cast<SIMDTexture *>(texture);
    if (lane_count == 1u) {
        if ((active_mask_bits & 1u) != 0u) {
            auto *input = static_cast<const uint *>(values);
            auto pixel = make_uint4(
                input[0u], input[1u], input[2u], input[3u]);
            auto view = self->view(level);
            if (self->dimension() == 2u) {
                view.write2d(make_uint2(x[0u], y[0u]), pixel);
            } else {
                view.write3d(make_uint3(x[0u], y[0u], z[0u]), pixel);
            }
        }
        return;
    }
    write_packet(
        self, level, lane_count,
        active_mask_bits, x, y, z, static_cast<const uint *>(values));
}

uint32_t SIMDTexture::_size(
    void *texture, uint32_t level, uint32_t axis) noexcept {
    auto size = static_cast<SIMDTexture *>(texture)->view(level).size3d();
    return axis < 3u ? size[axis] : 0u;
}

void SIMDTexture::_sample_float(
    void *texture, uint32_t base_level, uint32_t dimension,
    uint32_t lane_count, uint64_t active_mask_bits,
    const uint32_t *sampler_codes, const float *u,
    const float *v, const float *w, const float *levels,
    float *values) noexcept {
    active_mask_bits &= lane_mask(lane_count);
    if (active_mask_bits == 0u) { return; }
    auto *self = static_cast<SIMDTexture *>(texture);
    LUISA_ASSERT(
        self != nullptr && sampler_codes != nullptr &&
            u != nullptr && v != nullptr && w != nullptr &&
            values != nullptr,
        "SIMD direct texture sample received a null packet field.");
    LUISA_ASSERT(
        dimension == self->dimension(),
        "SIMD direct texture sample dimension mismatch ({} vs {}).",
        dimension, self->dimension());
    LUISA_ASSERT(
        base_level < self->mip_levels(),
        "SIMD direct texture base mip {} is out of range (size {}).",
        base_level, self->mip_levels());
    for (auto remaining = active_mask_bits; remaining != 0u;) {
        auto seed = static_cast<uint32_t>(std::countr_zero(remaining));
        auto sampler_code = sampler_codes[seed];
        LUISA_ASSERT(
            sampler_code < 16u,
            "Invalid SIMD direct sampler code {}.", sampler_code);
        auto group = uint64_t{0u};
        for (auto candidates = remaining; candidates != 0u;
             candidates &= candidates - 1u) {
            auto lane = static_cast<uint32_t>(
                std::countr_zero(candidates));
            if (sampler_codes[lane] == sampler_code) {
                group |= uint64_t{1u} << lane;
            }
        }
        self->sample_float_packet(
            Sampler::decode(sampler_code), base_level,
            lane_count, group, u, v, w, levels, values);
        remaining &= ~group;
    }
}

void SIMDTexture::sample_float_packet(
    Sampler sampler, uint32_t base_level, uint32_t lane_count,
    uint64_t active_mask_bits, const float *u,
    const float *v, const float *w, const float *levels,
    float *values) noexcept {
    switch (lane_count) {
        case 1u:
            sample_packet_fixed<1u>(
                this, sampler, base_level, active_mask_bits,
                u, v, w, levels, values);
            break;
        case 2u:
            sample_packet_fixed<2u>(
                this, sampler, base_level, active_mask_bits,
                u, v, w, levels, values);
            break;
        case 4u:
            sample_packet_fixed<4u>(
                this, sampler, base_level, active_mask_bits,
                u, v, w, levels, values);
            break;
        case 8u:
            sample_packet_fixed<8u>(
                this, sampler, base_level, active_mask_bits,
                u, v, w, levels, values);
            break;
        case 16u:
            sample_packet_fixed<16u>(
                this, sampler, base_level, active_mask_bits,
                u, v, w, levels, values);
            break;
        default:
            sample_packet_dynamic(
                this, sampler, base_level, lane_count, active_mask_bits,
                u, v, w, levels, values);
            break;
    }
}

SIMDHostTextureView SIMDTexture::host_view(uint level) noexcept {
    return {
        .texture = this,
        .read_float = _read_float,
        .read_uint = _read_uint,
        .write_float = _write_float,
        .write_uint = _write_uint,
        .size = _size,
        .level = level,
        .dimension = _dimension,
        .sample_float = _sample_float,
    };
}

}// namespace luisa::compute::simd
