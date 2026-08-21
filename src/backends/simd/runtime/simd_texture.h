#pragma once

#include "../../fallback/fallback_texture.h"
#include "../llvm/llvm_schedule_codegen.h"

namespace luisa::compute::simd {

class SIMDTexture {

private:
    fallback::FallbackTexture _texture;
    uint _dimension{2u};
    bool _enable_contiguous_packets{true};
    bool _enable_direct_native_packets{true};
    bool _enable_direct_byte4_packets{true};

private:
    static void _read_float(
        void *texture, uint32_t level, uint32_t lane_count,
        uint64_t active_mask_bits, const uint32_t *x,
        const uint32_t *y, const uint32_t *z, void *values) noexcept;
    static void _read_uint(
        void *texture, uint32_t level, uint32_t lane_count,
        uint64_t active_mask_bits, const uint32_t *x,
        const uint32_t *y, const uint32_t *z, void *values) noexcept;
    static void _write_float(
        void *texture, uint32_t level, uint32_t lane_count,
        uint64_t active_mask_bits, const uint32_t *x,
        const uint32_t *y, const uint32_t *z,
        const void *values) noexcept;
    static void _write_uint(
        void *texture, uint32_t level, uint32_t lane_count,
        uint64_t active_mask_bits, const uint32_t *x,
        const uint32_t *y, const uint32_t *z,
        const void *values) noexcept;
    [[nodiscard]] static uint32_t _size(
        void *texture, uint32_t level, uint32_t axis) noexcept;
    static void _sample_float(
        void *texture, uint32_t base_level, uint32_t dimension,
        uint32_t lane_count, uint64_t active_mask_bits,
        const uint32_t *sampler_codes, const float *u,
        const float *v, const float *w, const float *levels,
        float *values) noexcept;

public:
    SIMDTexture(
        PixelStorage storage, uint dimension, uint3 size,
        uint mip_levels) noexcept;
    SIMDTexture(
        PixelStorage storage, uint dimension, uint3 size,
        uint mip_levels, std::byte *external_memory) noexcept;
    ~SIMDTexture() noexcept = default;

    [[nodiscard]] auto view(uint level) const noexcept {
        return _texture.view(level);
    }
    [[nodiscard]] SIMDHostTextureView host_view(
        uint level) noexcept;
    [[nodiscard]] uint3 size(uint32_t level) const noexcept;
    void sample_float_packet(
        Sampler sampler, uint32_t base_level, uint32_t lane_count,
        uint64_t active_mask_bits, const float *u,
        const float *v, const float *w, const float *levels,
        float *values) noexcept;
    [[nodiscard]] auto native_handle() const noexcept {
        return _texture.native_handle();
    }
    [[nodiscard]] auto dimension() const noexcept { return _dimension; }
    [[nodiscard]] auto mip_levels() const noexcept {
        return _texture.mip_levels();
    }
    [[nodiscard]] auto contiguous_packets_enabled() const noexcept {
        return _enable_contiguous_packets;
    }
};

}// namespace luisa::compute::simd
