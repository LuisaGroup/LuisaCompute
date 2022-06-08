//
// Created by Mike Smith on 2022/6/8.
//

#pragma once

#include <core/basic_types.h>
#include <core/mathematics.h>
#include <core/stl.h>
#include <runtime/pixel.h>

namespace luisa::compute::llvm {

namespace detail {

// from tinyexr: https://github.com/syoyo/tinyexr/blob/master/tinyexr.h
[[nodiscard]] inline uint float_to_half(float f) noexcept {
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

[[nodiscard]] inline float half_to_float(uint h) noexcept {
    auto x = ((h & 0x8000u) << 16u) |
             (((h & 0x7c00u) + 0x1c000u) << 13u) |
             ((h & 0x03ffu) << 13u);
    return luisa::bit_cast<float>(x);
}

template<typename T>
[[nodiscard]] inline float scalar_to_float(T x) noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return x;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return x / 255.f;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return x / 65535.f;
    } else if constexpr (std::is_same_v<T, int16_t>) {
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
    } else if constexpr (std::is_same_v<T, int16_t>) {
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
            scalar_to_float(value[0]),
            0.f, 0.0f, 0.f);
    } else if constexpr (dim == 2u) {
        return make_float4(
            scalar_to_float(value[0]),
            scalar_to_float(value[1]),
            0.0f, 0.f);
    } else if constexpr (dim == 4u) {
        return make_float4(
            scalar_to_float(value[0]),
            scalar_to_float(value[1]),
            scalar_to_float(value[2]),
            scalar_to_float(value[3]));
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
            scalar_to_int(value[0]),
            0u, 0u, 0u);
    } else if constexpr (dim == 2u) {
        return make_uint4(
            scalar_to_int(value[0]),
            scalar_to_int(value[1]),
            0u, 0u);
    } else if constexpr (dim == 4u) {
        return make_uint4(
            scalar_to_int(value[0]),
            scalar_to_int(value[1]),
            scalar_to_int(value[2]),
            scalar_to_int(value[3]));
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
        case PixelStorage::HALF1: return detail::read_pixel<T, int16_t, 1u>(p);
        case PixelStorage::HALF2: return detail::read_pixel<T, int16_t, 2u>(p);
        case PixelStorage::HALF4: return detail::read_pixel<T, int16_t, 4u>(p);
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
        case PixelStorage::HALF1: detail::write_pixel<T, int16_t, 1u>(p, v); break;
        case PixelStorage::HALF2: detail::write_pixel<T, int16_t, 2u>(p, v); break;
        case PixelStorage::HALF4: detail::write_pixel<T, int16_t, 4u>(p, v); break;
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

private:
    std::byte *_data{nullptr};           // 8B
    std::array<uint16_t, 3u> _size{};    // 14B
    PixelStorage _storage : 8u;          // 15B
    uint _pixel_stride : 8u;             // 16B
    std::array<uint, 16u> _mip_offsets{};// 80B

public:
    LLVMTexture(PixelStorage storage, uint3 size, uint levels) noexcept;
    ~LLVMTexture() noexcept { luisa::deallocate(_data); }
    LLVMTexture(LLVMTexture &&) noexcept = delete;
    LLVMTexture(const LLVMTexture &) noexcept = delete;
    LLVMTexture &operator=(LLVMTexture &&) noexcept = delete;
    LLVMTexture &operator=(const LLVMTexture &) noexcept = delete;
    [[nodiscard]] LLVMTextureView view(uint level) const noexcept;
    [[nodiscard]] auto storage() const noexcept { return _storage; }
};

class alignas(16u) LLVMTextureView {

private:
    std::byte *_data;           // 8B
    uint _width : 16u;          // 10B
    uint _height : 16u;         // 12B
    PixelStorage _storage : 16u;// 14B
    uint _pixel_stride : 16u;   // 16B

private:
    [[nodiscard]] inline std::byte *_pixel2d(uint2 xy) const noexcept {
        auto offset = _pixel_stride * (xy[0] + xy[1] * _width);
        return _data + offset;
    }
    [[nodiscard]] inline std::byte *_pixel3d(uint3 xyz) const noexcept {
        auto offset = _pixel_stride * (xyz[0] + xyz[1] * _width + xyz[2] * _width * _height);
        return _data + offset;
    }

private:
    friend class LLVMTexture;
    LLVMTextureView(std::byte *data, uint w, uint h,
                    PixelStorage storage, uint pixel_stride) noexcept
        : _data(data), _width{w}, _height{h},
          _storage(storage), _pixel_stride(pixel_stride) {}

public:
    template<typename T>
    [[nodiscard]] inline Vector<T, 4u> read2d(uint2 xy) const noexcept {
        return detail::read_pixel<T>(_storage, _pixel2d(xy));
    }
    template<typename T>
    [[nodiscard]] inline Vector<T, 4u> read3d(uint3 xyz) const noexcept {
        return detail::read_pixel<T>(_storage, _pixel3d(xyz));
    }
    template<typename T>
    inline void write2d(uint2 xy, Vector<T, 4u> value) const noexcept {
        detail::write_pixel<T>(_storage, _pixel2d(xy), value);
    }
    template<typename T>
    inline void write3d(uint3 xyz, Vector<T, 4u> value) const noexcept {
        detail::write_pixel<T>(_storage, _pixel3d(xyz), value);
    }
    [[nodiscard]] inline auto data() const noexcept { return _data; }
};

static_assert(sizeof(LLVMTextureView) == 16u);

inline LLVMTextureView LLVMTexture::view(uint level) const noexcept {
    auto size = luisa::max(make_uint2(_size[0], _size[1]) >> level, 1u);
    return LLVMTextureView{_data + _mip_offsets[level], size.x, size.y,
                           _storage, _pixel_stride};
}

void texture_write_2d_float(LLVMTextureView tex, uint2 xy, float4 v) noexcept;
void texture_write_3d_float(LLVMTextureView tex, uint3 xyz, float4 v) noexcept;
void texture_write_2d_int(LLVMTextureView tex, uint2 xy, int4 v) noexcept;
void texture_write_3d_int(LLVMTextureView tex, uint3 xyz, int4 v) noexcept;
void texture_write_2d_uint(LLVMTextureView tex, uint2 xy, uint4 v) noexcept;
void texture_write_3d_uint(LLVMTextureView tex, uint3 xyz, uint4 v) noexcept;
float4 texture_read_2d_float(LLVMTextureView tex, uint2 xy) noexcept;
float4 texture_read_3d_float(LLVMTextureView tex, uint3 xyz) noexcept;
int4 texture_read_2d_int(LLVMTextureView tex, uint2 xy) noexcept;
int4 texture_read_3d_int(LLVMTextureView tex, uint3 xyz) noexcept;
uint4 texture_read_2d_uint(LLVMTextureView tex, uint2 xy) noexcept;
uint4 texture_read_3d_uint(LLVMTextureView tex, uint3 xyz) noexcept;

}// namespace luisa::compute::llvm
