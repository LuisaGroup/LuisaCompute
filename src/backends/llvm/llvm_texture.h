//
// Created by Mike Smith on 2022/6/8.
//

#pragma once

#include <bit>

#include <core/basic_types.h>
#include <core/mathematics.h>
#include <core/stl.h>
#include <runtime/pixel.h>
#include <runtime/sampler.h>
#include <backends/llvm/llvm_abi.h>

namespace luisa::compute::llvm {

namespace detail {

[[nodiscard]] float16_t float_to_half(float f) noexcept;
[[nodiscard]] float half_to_float(float16_t h) noexcept;

template<typename T>
[[nodiscard]] inline float scalar_to_float(T x) noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return x;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return x / 255.f;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return x / 65535.f;
    } else if constexpr (std::is_same_v<T, float16_t>) {
        return half_to_float(x);
    } else {
        return 0.f;
    }
}

template<typename T>
[[nodiscard]] inline T float_to_scalar(float x) noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return x;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return static_cast<T>(std::clamp(std::round(x * 255.f), 0.f, 255.f));
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return static_cast<T>(std::clamp(std::round(x * 65535.f), 0.f, 65535.f));
    } else if constexpr (std::is_same_v<T, float16_t>) {
        return static_cast<T>(float_to_half(x));
    } else {
        return static_cast<T>(0);
    }
}

template<typename T>
[[nodiscard]] inline uint scalar_to_int(T x) noexcept {
    return static_cast<uint>(x);
}

template<typename T>
[[nodiscard]] inline T int_to_scalar(uint x) noexcept {
    return static_cast<T>(x);
}

template<typename T, uint dim>
[[nodiscard]] inline float4 pixel_to_float4(const std::byte *pixel) noexcept {
    auto value = reinterpret_cast<const T *>(pixel);
    if constexpr (dim == 1u) {
        return make_float4(
            scalar_to_float<T>(value[0]),
            0.f, 0.0f, 0.f);
    } else if constexpr (dim == 2u) {
        return make_float4(
            scalar_to_float<T>(value[0]),
            scalar_to_float<T>(value[1]),
            0.0f, 0.f);
    } else if constexpr (dim == 4u) {
        return make_float4(
            scalar_to_float<T>(value[0]),
            scalar_to_float<T>(value[1]),
            scalar_to_float<T>(value[2]),
            scalar_to_float<T>(value[3]));
    } else {
        return make_float4();
    }
}

template<typename T, uint dim>
inline void float4_to_pixel(std::byte *pixel, float4 v) noexcept {
    auto value = reinterpret_cast<T *>(pixel);
    if constexpr (dim == 1u) {
        value[0] = float_to_scalar<T>(v[0]);
    } else if constexpr (dim == 2u) {
        value[0] = float_to_scalar<T>(v[0]);
        value[1] = float_to_scalar<T>(v[1]);
    } else if constexpr (dim == 4u) {
        value[0] = float_to_scalar<T>(v[0]);
        value[1] = float_to_scalar<T>(v[1]);
        value[2] = float_to_scalar<T>(v[2]);
        value[3] = float_to_scalar<T>(v[3]);
    }
}

template<typename T, uint dim>
[[nodiscard]] inline uint4 pixel_to_int4(const std::byte *pixel) noexcept {
    auto value = reinterpret_cast<const T *>(pixel);
    if constexpr (dim == 1u) {
        return make_uint4(
            scalar_to_int<T>(value[0]),
            0u, 0u, 0u);
    } else if constexpr (dim == 2u) {
        return make_uint4(
            scalar_to_int<T>(value[0]),
            scalar_to_int<T>(value[1]),
            0u, 0u);
    } else if constexpr (dim == 4u) {
        return make_uint4(
            scalar_to_int<T>(value[0]),
            scalar_to_int<T>(value[1]),
            scalar_to_int<T>(value[2]),
            scalar_to_int<T>(value[3]));
    } else {
        return make_uint4();
    }
}

template<typename T, uint dim>
inline void int4_to_pixel(std::byte *pixel, uint4 v) noexcept {
    auto value = reinterpret_cast<T *>(pixel);
    if constexpr (dim == 1u) {
        value[0] = int_to_scalar<T>(v[0]);
    } else if constexpr (dim == 2u) {
        value[0] = int_to_scalar<T>(v[0]);
        value[1] = int_to_scalar<T>(v[1]);
    } else if constexpr (dim == 4u) {
        value[0] = int_to_scalar<T>(v[0]);
        value[1] = int_to_scalar<T>(v[1]);
        value[2] = int_to_scalar<T>(v[2]);
        value[3] = int_to_scalar<T>(v[3]);
    }
}

template<typename Dst, typename Src, uint dim>
[[nodiscard]] inline auto read_pixel(const std::byte *p) noexcept {
    if constexpr (std::is_same_v<Dst, float>) {
        return pixel_to_float4<Src, dim>(p);
    } else {
        static_assert(std::is_same_v<Dst, int> ||
                      std::is_same_v<Dst, uint>);
        return luisa::bit_cast<Vector<Dst, 4u>>(
            pixel_to_int4<Src, dim>(p));
    }
}

template<typename Dst, typename Src, uint dim>
[[nodiscard]] inline auto write_pixel(std::byte *p, Vector<Dst, 4u> value) noexcept {
    if constexpr (std::is_same_v<Dst, float>) {
        float4_to_pixel<Src, dim>(p, value);
    } else {
        static_assert(std::is_same_v<Dst, int> ||
                      std::is_same_v<Dst, uint>);
        int4_to_pixel<Src, dim>(
            p, luisa::bit_cast<uint4>(value));
    }
}

template<typename T>
[[nodiscard]] inline Vector<T, 4u> read_pixel(PixelStorage storage, const std::byte *p) noexcept {
    switch (storage) {
        case PixelStorage::BYTE1: return detail::read_pixel<T, uint8_t, 1u>(p);
        case PixelStorage::BYTE2: return detail::read_pixel<T, uint8_t, 2u>(p);
        case PixelStorage::BYTE4: return detail::read_pixel<T, uint8_t, 4u>(p);
        case PixelStorage::SHORT1: return detail::read_pixel<T, uint16_t, 1u>(p);
        case PixelStorage::SHORT2: return detail::read_pixel<T, uint16_t, 2u>(p);
        case PixelStorage::SHORT4: return detail::read_pixel<T, uint16_t, 4u>(p);
        case PixelStorage::INT1: return detail::read_pixel<T, uint32_t, 1u>(p);
        case PixelStorage::INT2: return detail::read_pixel<T, uint32_t, 2u>(p);
        case PixelStorage::INT4: return detail::read_pixel<T, uint32_t, 4u>(p);
        case PixelStorage::HALF1: return detail::read_pixel<T, float16_t, 1u>(p);
        case PixelStorage::HALF2: return detail::read_pixel<T, float16_t, 2u>(p);
        case PixelStorage::HALF4: return detail::read_pixel<T, float16_t, 4u>(p);
        case PixelStorage::FLOAT1: return detail::read_pixel<T, float, 1u>(p);
        case PixelStorage::FLOAT2: return detail::read_pixel<T, float, 2u>(p);
        case PixelStorage::FLOAT4: return detail::read_pixel<T, float, 4u>(p);
        default: break;
    }
    return {};
}

template<typename T>
inline void write_pixel(PixelStorage storage, std::byte *p, Vector<T, 4u> v) noexcept {
    switch (storage) {
        case PixelStorage::BYTE1: detail::write_pixel<T, uint8_t, 1u>(p, v); break;
        case PixelStorage::BYTE2: detail::write_pixel<T, uint8_t, 2u>(p, v); break;
        case PixelStorage::BYTE4: detail::write_pixel<T, uint8_t, 4u>(p, v); break;
        case PixelStorage::SHORT1: detail::write_pixel<T, uint16_t, 1u>(p, v); break;
        case PixelStorage::SHORT2: detail::write_pixel<T, uint16_t, 2u>(p, v); break;
        case PixelStorage::SHORT4: detail::write_pixel<T, uint16_t, 4u>(p, v); break;
        case PixelStorage::INT1: detail::write_pixel<T, uint32_t, 1u>(p, v); break;
        case PixelStorage::INT2: detail::write_pixel<T, uint32_t, 2u>(p, v); break;
        case PixelStorage::INT4: detail::write_pixel<T, uint32_t, 4u>(p, v); break;
        case PixelStorage::HALF1: detail::write_pixel<T, float16_t, 1u>(p, v); break;
        case PixelStorage::HALF2: detail::write_pixel<T, float16_t, 2u>(p, v); break;
        case PixelStorage::HALF4: detail::write_pixel<T, float16_t, 4u>(p, v); break;
        case PixelStorage::FLOAT1: detail::write_pixel<T, float, 1u>(p, v); break;
        case PixelStorage::FLOAT2: detail::write_pixel<T, float, 2u>(p, v); break;
        case PixelStorage::FLOAT4: detail::write_pixel<T, float, 4u>(p, v); break;
        default: break;
    }
}

}// namespace detail

class LLVMTexture;
class LLVMTextureView;

class alignas(16u) LLVMTexture {

public:
    static constexpr auto block_size = 4u;

private:
    std::byte *_data{nullptr};           // 8B
    std::array<uint16_t, 3u> _size{};    // 14B
    PixelStorage _storage : 16u;         // 16B
    uint _pixel_stride_shift : 8u;       // 18B
    uint _mip_levels : 8u;               // 19B
    uint _dimension : 8u;                // 20B
    std::array<uint, 15u> _mip_offsets{};// 80B

public:
    LLVMTexture(PixelStorage storage, uint dim, uint3 size, uint levels) noexcept;
    ~LLVMTexture() noexcept;
    LLVMTexture(LLVMTexture &&) noexcept = delete;
    LLVMTexture(const LLVMTexture &) noexcept = delete;
    LLVMTexture &operator=(LLVMTexture &&) noexcept = delete;
    LLVMTexture &operator=(const LLVMTexture &) noexcept = delete;
    [[nodiscard]] LLVMTextureView view(uint level) const noexcept;
    [[nodiscard]] auto storage() const noexcept { return _storage; }

    // reading
    [[nodiscard]] float4 read2d(uint level, uint2 uv) const noexcept;
    [[nodiscard]] float4 read3d(uint level, uint3 uvw) const noexcept;

    // sampling
    [[nodiscard]] float4 sample2d(Sampler sampler, float2 uv) const noexcept;
    [[nodiscard]] float4 sample3d(Sampler sampler, float3 uvw) const noexcept;
    [[nodiscard]] float4 sample2d(Sampler sampler, float2 uv, float lod) const noexcept;
    [[nodiscard]] float4 sample3d(Sampler sampler, float3 uvw, float lod) const noexcept;
    [[nodiscard]] float4 sample2d(Sampler sampler, float2 uv, float2 dpdx, float2 dpdy) const noexcept;
    [[nodiscard]] float4 sample3d(Sampler sampler, float3 uvw, float3 dpdx, float3 dpdy) const noexcept;
};

class alignas(16u) LLVMTextureView {

private:
    std::byte *_data;          // 8B
    uint _width : 16u;         // 10B
    uint _height : 16u;        // 12B
    uint _depth : 16u;         // 14B
    PixelStorage _storage : 8u;// 15B
    uint _dimension : 4u;
    uint _pixel_stride_shift : 4u;// 16B

public:
    static constexpr auto block_size = LLVMTexture::block_size;

private:
    [[nodiscard]] inline std::byte *_pixel2d(uint2 xy) const noexcept {
        auto block = (xy + block_size - 1u) / block_size;
        auto pixel = xy % block_size;
        auto grid_width = (_width + block_size - 1u) / block_size;
        auto block_index = grid_width * block.y + block.x;
        auto pixel_index = block_index * block_size * block_size +
                           pixel.y * block_size + pixel.x;
        return _data + (static_cast<size_t>(pixel_index) << _pixel_stride_shift);
    }
    [[nodiscard]] inline std::byte *_pixel3d(uint3 xyz) const noexcept {
        auto block = (xyz + block_size - 1u) / block_size;
        auto pixel = xyz % block_size;
        auto grid_width = (_width + block_size - 1u) / block_size;
        auto grid_height = (_height + block_size - 1u) / block_size;
        auto block_index = grid_width * grid_height * block.z + grid_width * block.y + block.x;
        auto pixel_index = block_index * block_size * block_size * block_size +
                           (pixel.z * block_size + pixel.y) * block_size + pixel.x;
        return _data + (static_cast<size_t>(pixel_index) << _pixel_stride_shift);
    }
    [[nodiscard]] inline auto _out_of_bounds(uint2 xy) const noexcept {
        return !(xy[0] < _width & xy[1] < _height);
    }
    [[nodiscard]] inline auto _out_of_bounds(uint3 xyz) const noexcept {
        return !(xyz[0] < _width & xyz[1] < _height & xyz[2] < _depth);
    }

private:
    friend class LLVMTexture;
    LLVMTextureView(std::byte *data, uint dim, uint w, uint h, uint d,
                    PixelStorage storage, uint pixel_stride_shift) noexcept
        : _data(data), _width{w}, _height{h}, _depth{d}, _storage(storage),
          _dimension{dim}, _pixel_stride_shift(pixel_stride_shift) {}

public:
    template<typename T>
    [[nodiscard]] inline Vector<T, 4u> read2d(uint2 xy) const noexcept {
        if (_out_of_bounds(xy)) [[unlikely]] { return {}; }
        return detail::read_pixel<T>(_storage, _pixel2d(xy));
    }
    template<typename T>
    [[nodiscard]] inline Vector<T, 4u> read3d(uint3 xyz) const noexcept {
        if (_out_of_bounds(xyz)) [[unlikely]] { return {}; }
        return detail::read_pixel<T>(_storage, _pixel3d(xyz));
    }
    template<typename T>
    inline void write2d(uint2 xy, Vector<T, 4u> value) const noexcept {
        if (_out_of_bounds(xy)) [[unlikely]] { return; }
        detail::write_pixel<T>(_storage, _pixel2d(xy), value);
    }
    template<typename T>
    inline void write3d(uint3 xyz, Vector<T, 4u> value) const noexcept {
        if (_out_of_bounds(xyz)) [[unlikely]] { return; }
        detail::write_pixel<T>(_storage, _pixel3d(xyz), value);
    }
    [[nodiscard]] auto size2d() const noexcept { return make_uint2(_width, _height); }
    [[nodiscard]] auto size3d() const noexcept { return make_uint3(_width, _height, _depth); }
    [[nodiscard]] auto size_bytes() const noexcept { return (_width * _height * _depth) << _pixel_stride_shift; }
    void copy_from(const void *data) const noexcept;
    void copy_to(void *data) const noexcept;
    void copy_from(LLVMTextureView dst) const noexcept;
};

static_assert(sizeof(LLVMTextureView) == 16u);

[[nodiscard]] float32x4_t texture_read_2d_int(int64_t t0, int64_t t1, int64_t c0, int64_t c1) noexcept;
[[nodiscard]] float32x4_t texture_read_3d_int(int64_t t0, int64_t t1, int64_t c0, int64_t c1) noexcept;
[[nodiscard]] float32x4_t texture_read_2d_uint(int64_t t0, int64_t t1, int64_t c0, int64_t c1) noexcept;
[[nodiscard]] float32x4_t texture_read_3d_uint(int64_t t0, int64_t t1, int64_t c0, int64_t c1) noexcept;
[[nodiscard]] float32x4_t texture_read_2d_float(int64_t t0, int64_t t1, int64_t c0, int64_t c1) noexcept;
[[nodiscard]] float32x4_t texture_read_3d_float(int64_t t0, int64_t t1, int64_t c0, int64_t c1) noexcept;
void texture_write_2d_int(int64_t t0, int64_t t1, int64_t c0, int64_t c1, int64_t v0, int64_t v1) noexcept;
void texture_write_3d_int(int64_t t0, int64_t t1, int64_t c0, int64_t c1, int64_t v0, int64_t v1) noexcept;
void texture_write_2d_uint(int64_t t0, int64_t t1, int64_t c0, int64_t c1, int64_t v0, int64_t v1) noexcept;
void texture_write_3d_uint(int64_t t0, int64_t t1, int64_t c0, int64_t c1, int64_t v0, int64_t v1) noexcept;
void texture_write_2d_float(int64_t t0, int64_t t1, int64_t c0, int64_t c1, int64_t v0, int64_t v1) noexcept;
void texture_write_3d_float(int64_t t0, int64_t t1, int64_t c0, int64_t c1, int64_t v0, int64_t v1) noexcept;

[[nodiscard]] float32x4_t bindless_texture_2d_read(const LLVMTexture *tex, uint level, uint x, uint y) noexcept;
[[nodiscard]] float32x4_t bindless_texture_3d_read(const LLVMTexture *tex, uint level, uint x, uint y, uint z) noexcept;
[[nodiscard]] float32x4_t bindless_texture_2d_sample(const LLVMTexture *tex, uint sampler, float u, float v) noexcept;
[[nodiscard]] float32x4_t bindless_texture_3d_sample(const LLVMTexture *tex, uint sampler, float u, float v, float w) noexcept;
[[nodiscard]] float32x4_t bindless_texture_2d_sample_level(const LLVMTexture *tex, uint sampler, float u, float v, float lod) noexcept;
[[nodiscard]] float32x4_t bindless_texture_3d_sample_level(const LLVMTexture *tex, uint sampler, float u, float v, float w, float lod) noexcept;
[[nodiscard]] float32x4_t bindless_texture_2d_sample_grad(const LLVMTexture *tex, uint sampler, float u, float v, int64_t dpdx, int64_t dpdy) noexcept;
[[nodiscard]] float32x4_t bindless_texture_3d_sample_grad(const LLVMTexture *tex, int64_t sampler_w, int64_t uv, int64_t dudxy, int64_t dvdxy, int64_t dwdxy) noexcept;

}// namespace luisa::compute::llvm
