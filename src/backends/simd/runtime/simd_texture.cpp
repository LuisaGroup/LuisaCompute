#include "simd_texture.h"

#include <bit>

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

template<typename T>
[[nodiscard]] auto read_pixel(
    luisa::compute::simd::SIMDTexture *texture,
    luisa::compute::fallback::FallbackTextureView view,
    uint32_t x, uint32_t y, uint32_t z) noexcept {
    return texture->dimension() == 2u ?
        view.read2d<T>(luisa::make_uint2(x, y)) :
        view.read3d<T>(luisa::make_uint3(x, y, z));
}

template<typename T, size_t Width>
void read_packet_fixed(
    luisa::compute::simd::SIMDTexture *texture, uint32_t level,
    uint64_t active_mask_bits, const uint32_t *x,
    const uint32_t *y, const uint32_t *z, T *values) noexcept {
    active_mask_bits &= full_lane_mask<Width>();
    if (active_mask_bits == 0u) { return; }
    auto view = texture->view(level);
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
    auto contiguous = active_mask_bits == full_lane_mask<Width>() &&
                      texture->dimension() == 2u;
    for (auto lane = 1u; lane < Width && contiguous; lane++) {
        contiguous = x[lane] == x[0u] + lane &&
                     y[lane] == y[0u] && z[lane] == z[0u];
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
      _dimension{dimension} {}

SIMDTexture::SIMDTexture(
    PixelStorage storage, uint dimension, uint3 size,
    uint mip_levels, std::byte *external_memory) noexcept
    : _texture{storage, dimension, size, mip_levels,
               external_memory},
      _dimension{dimension} {}

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
    };
}

}// namespace luisa::compute::simd
