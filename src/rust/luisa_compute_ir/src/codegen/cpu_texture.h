constexpr float one_minus_epsilon = 0x1.fffffep-1f;
typedef enum LCPixelStorage {
    LC_PIXEL_STORAGE_BYTE1,
    LC_PIXEL_STORAGE_BYTE2,
    LC_PIXEL_STORAGE_BYTE4,
    LC_PIXEL_STORAGE_SHORT1,
    LC_PIXEL_STORAGE_SHORT2,
    LC_PIXEL_STORAGE_SHORT4,
    LC_PIXEL_STORAGE_INT1,
    LC_PIXEL_STORAGE_INT2,
    LC_PIXEL_STORAGE_INT4,
    LC_PIXEL_STORAGE_HALF1,
    LC_PIXEL_STORAGE_HALF2,
    LC_PIXEL_STORAGE_HALF4,
    LC_PIXEL_STORAGE_FLOAT1,
    LC_PIXEL_STORAGE_FLOAT2,
    LC_PIXEL_STORAGE_FLOAT4,
} LCPixelStorage;

typedef enum LCSamplerAddress {
    LC_SAMPLER_ADDRESS_EDGE,
    LC_SAMPLER_ADDRESS_REPEAT,
    LC_SAMPLER_ADDRESS_MIRROR,
    LC_SAMPLER_ADDRESS_ZERO,
} LCSamplerAddress;

typedef enum LCSamplerFilter {
    LC_SAMPLER_FILTER_POINT,
    LC_SAMPLER_FILTER_LINEAR_POINT,
    LC_SAMPLER_FILTER_LINEAR_LINEAR,
    LC_SAMPLER_FILTER_ANISOTROPIC,
} LCSamplerFilter;


#if defined(LUISA_ARCH_ARM64)
#include <arm_neon.h>

using float16_t = ::float16_t;

#else
#ifdef LUISA_ARCH_X86_64
#include <immintrin.h>
#include <xmmintrin.h>
#endif

using float16_t = int16_t;

#endif
namespace detail {
template<class A, class B>
struct lc_pair {
    A first;
    B second;
};
template<class A, class B>
lc_pair<A, B> lc_make_pair(A a, B b) noexcept {
    return {a, b};
}

float16_t float_to_half(float f) noexcept {
#if defined(LUISA_ARCH_ARM64)
    return static_cast<float16_t>(f);
#elif defined(LUISA_ARCH_X86_64)
    auto ss = _mm_set_ss(f);
    auto ph = _mm_cvtps_ph(ss, 0);
    return static_cast<float16_t>(_mm_cvtsi128_si32(ph));
#else
    auto bits = lc_bit_cast<uint>(f);
    auto fp32_sign = bits >> 31u;
    auto fp32_exponent = (bits >> 23u) & 0xffu;
    auto fp32_mantissa = bits & ((1u << 23u) - 1u);
    auto make_fp16 = [](uint sign, uint exponent, uint mantissa) noexcept {
        return static_cast<float16_t>((sign << 15u) | (exponent << 10u) | mantissa);
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
#endif
}

float half_to_float(float16_t half) noexcept {
#if defined(LUISA_ARCH_ARM64)
    return static_cast<float>(half);
#elif defined(LUISA_ARCH_X86_64)
    auto si = _mm_cvtsi32_si128(half);
    auto ps = _mm_cvtph_ps(si);
    return _mm_cvtss_f32(ps);
#else
    static_assert(std::endian::native == std::endian::little,
                  "Only little endian is supported");
    auto h = static_cast<uint>(half);
    union FP32 {
        unsigned int u;
        float f;
        struct {// FIXME: assuming little endian here
            unsigned int Mantissa : 23;
            unsigned int Exponent : 8;
            unsigned int Sign : 1;
        } s;
    };
    constexpr auto magic = FP32{113u << 23u};
    constexpr auto shifted_exp = 0x7c00u << 13u;// exponent mask after shift
    auto o = FP32{(h & 0x7fffu) << 13u};        // exponent/mantissa bits
    auto exp_ = shifted_exp & o.u;              // just the exponent
    o.u += (127u - 15u) << 23u;                 // exponent adjust

    // handle exponent special cases
    if (exp_ == shifted_exp) {     // Inf/NaN?
        o.u += (128u - 16u) << 23u;// extra exp adjust
    } else if (exp_ == 0u) {       // Zero/Denormal?
        o.u += 1u << 23u;          // extra exp adjust
        o.f -= magic.f;            // renormalize
    }
    o.u |= (h & 0x8000u) << 16u;// sign bit
    return o.f;
#endif
}


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
        return static_cast<T>(lc_clamp(std::round(x * 255.f), 0.f, 255.f));
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return static_cast<T>(lc_clamp(std::round(x * 65535.f), 0.f, 65535.f));
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
[[nodiscard]] inline lc_float4 pixel_to_float4(const uint8_t *pixel) noexcept {
    auto value = reinterpret_cast<const T *>(pixel);
    if constexpr (dim == 1u) {
        return lc_make_float4(
            scalar_to_float<T>(value[0]),
            0.f, 0.0f, 0.f);
    } else if constexpr (dim == 2u) {
        return lc_make_float4(
            scalar_to_float<T>(value[0]),
            scalar_to_float<T>(value[1]),
            0.0f, 0.f);
    } else if constexpr (dim == 4u) {
        return lc_make_float4(
            scalar_to_float<T>(value[0]),
            scalar_to_float<T>(value[1]),
            scalar_to_float<T>(value[2]),
            scalar_to_float<T>(value[3]));
    } else {
        return lc_make_float4();
    }
}

template<typename T, uint dim>
inline void float4_to_pixel(uint8_t *pixel, lc_float4 v) noexcept {
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
[[nodiscard]] inline lc_uint4 pixel_to_int4(const uint8_t *pixel) noexcept {
    auto value = reinterpret_cast<const T *>(pixel);
    if constexpr (dim == 1u) {
        return lc_make_uint4(
            scalar_to_int<T>(value[0]),
            0u, 0u, 0u);
    } else if constexpr (dim == 2u) {
        return lc_make_uint4(
            scalar_to_int<T>(value[0]),
            scalar_to_int<T>(value[1]),
            0u, 0u);
    } else if constexpr (dim == 4u) {
        return lc_make_uint4(
            scalar_to_int<T>(value[0]),
            scalar_to_int<T>(value[1]),
            scalar_to_int<T>(value[2]),
            scalar_to_int<T>(value[3]));
    } else {
        return lc_make_uint4();
    }
}

template<typename T, uint dim>
inline void int4_to_pixel(uint8_t *pixel, lc_uint4 v) noexcept {
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

template<class V, typename Dst, typename Src, uint dim>
[[nodiscard]] inline auto read_pixel(const uint8_t *p) noexcept {
    if constexpr (std::is_same_v<Dst, float>) {
        return pixel_to_float4<Src, dim>(p);
    } else {
        static_assert(std::is_same_v<Dst, int> ||
                      std::is_same_v<Dst, uint>);
        return lc_bit_cast<V>(
            pixel_to_int4<Src, dim>(p));
    }
}

template<typename V, typename Dst, typename Src, uint dim>
[[nodiscard]] inline auto write_pixel(uint8_t *p, V value) noexcept {
    if constexpr (std::is_same_v<Dst, float>) {
        float4_to_pixel<Src, dim>(p, value);
    } else {
        static_assert(std::is_same_v<Dst, int> ||
                      std::is_same_v<Dst, uint>);
        int4_to_pixel<Src, dim>(
            p, lc_bit_cast<lc_uint4>(value));
    }
}

template<class V, typename T>
[[nodiscard]] inline V read_pixel(LCPixelStorage storage, const uint8_t *p) noexcept {
    switch (storage) {
        case LC_PIXEL_STORAGE_BYTE1: return detail::read_pixel<V, T, uint8_t, 1u>(p);
        case LC_PIXEL_STORAGE_BYTE2: return detail::read_pixel<V, T, uint8_t, 2u>(p);
        case LC_PIXEL_STORAGE_BYTE4: return detail::read_pixel<V, T, uint8_t, 4u>(p);
        case LC_PIXEL_STORAGE_SHORT1: return detail::read_pixel<V, T, uint16_t, 1u>(p);
        case LC_PIXEL_STORAGE_SHORT2: return detail::read_pixel<V, T, uint16_t, 2u>(p);
        case LC_PIXEL_STORAGE_SHORT4: return detail::read_pixel<V, T, uint16_t, 4u>(p);
        case LC_PIXEL_STORAGE_INT1: return detail::read_pixel<V, T, uint32_t, 1u>(p);
        case LC_PIXEL_STORAGE_INT2: return detail::read_pixel<V, T, uint32_t, 2u>(p);
        case LC_PIXEL_STORAGE_INT4: return detail::read_pixel<V, T, uint32_t, 4u>(p);
        case LC_PIXEL_STORAGE_HALF1: return detail::read_pixel<V, T, float16_t, 1u>(p);
        case LC_PIXEL_STORAGE_HALF2: return detail::read_pixel<V, T, float16_t, 2u>(p);
        case LC_PIXEL_STORAGE_HALF4: return detail::read_pixel<V, T, float16_t, 4u>(p);
        case LC_PIXEL_STORAGE_FLOAT1: return detail::read_pixel<V, T, float, 1u>(p);
        case LC_PIXEL_STORAGE_FLOAT2: return detail::read_pixel<V, T, float, 2u>(p);
        case LC_PIXEL_STORAGE_FLOAT4: return detail::read_pixel<V, T, float, 4u>(p);
        default: break;
    }
    return {};
}

template<typename V, typename T>
inline void write_pixel(LCPixelStorage storage, uint8_t *p, V v) noexcept {
    switch (storage) {
        case LC_PIXEL_STORAGE_BYTE1: detail::write_pixel<V, T, uint8_t, 1u>(p, v); break;
        case LC_PIXEL_STORAGE_BYTE2: detail::write_pixel<V, T, uint8_t, 2u>(p, v); break;
        case LC_PIXEL_STORAGE_BYTE4: detail::write_pixel<V, T, uint8_t, 4u>(p, v); break;
        case LC_PIXEL_STORAGE_SHORT1: detail::write_pixel<V, T, uint16_t, 1u>(p, v); break;
        case LC_PIXEL_STORAGE_SHORT2: detail::write_pixel<V, T, uint16_t, 2u>(p, v); break;
        case LC_PIXEL_STORAGE_SHORT4: detail::write_pixel<V, T, uint16_t, 4u>(p, v); break;
        case LC_PIXEL_STORAGE_INT1: detail::write_pixel<V, T, uint32_t, 1u>(p, v); break;
        case LC_PIXEL_STORAGE_INT2: detail::write_pixel<V, T, uint32_t, 2u>(p, v); break;
        case LC_PIXEL_STORAGE_INT4: detail::write_pixel<V, T, uint32_t, 4u>(p, v); break;
        case LC_PIXEL_STORAGE_HALF1: detail::write_pixel<V, T, float16_t, 1u>(p, v); break;
        case LC_PIXEL_STORAGE_HALF2: detail::write_pixel<V, T, float16_t, 2u>(p, v); break;
        case LC_PIXEL_STORAGE_HALF4: detail::write_pixel<V, T, float16_t, 4u>(p, v); break;
        case LC_PIXEL_STORAGE_FLOAT1: detail::write_pixel<V, T, float, 1u>(p, v); break;
        case LC_PIXEL_STORAGE_FLOAT2: detail::write_pixel<V, T, float, 2u>(p, v); break;
        case LC_PIXEL_STORAGE_FLOAT4: detail::write_pixel<V, T, float, 4u>(p, v); break;
        default: break;
    }
}

// MIP-Map EWA filtering LUT from PBRT-v4
static constexpr const float ewa_filter_weight_lut[] = {
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
static constexpr const int ewa_filter_weight_lut_size = sizeof(ewa_filter_weight_lut) / sizeof(float);
}// namespace detail

struct TextureView {
    uint8_t *data;
    uint8_t dimension;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint8_t storage;
    uint8_t pixel_stride_shift;

    static constexpr auto block_size = 4;

    [[nodiscard]] inline uint8_t *_pixel2d(lc_uint2 xy) const noexcept {
        auto block = xy / block_size;
        auto pixel = xy % block_size;
        auto grid_width = (width + block_size - 1u) / block_size;
        auto block_index = grid_width * block.y + block.x;
        auto pixel_index = block_index * block_size * block_size +
                           pixel.y * block_size + pixel.x;
        return data + (static_cast<size_t>(pixel_index) << pixel_stride_shift);
    }
    [[nodiscard]] inline uint8_t *_pixel3d(lc_uint3 xyz) const noexcept {
        auto block = xyz / block_size;
        auto pixel = xyz % block_size;
        auto grid_width = (width + block_size - 1u) / block_size;
        auto grid_height = (height + block_size - 1u) / block_size;
        auto block_index = grid_width * grid_height * block.z + grid_width * block.y + block.x;
        auto pixel_index = block_index * block_size * block_size * block_size +
                           (pixel.z * block_size + pixel.y) * block_size + pixel.x;
        return data + (static_cast<size_t>(pixel_index) << pixel_stride_shift);
    }
    [[nodiscard]] inline auto _out_of_bounds(lc_uint2 xy) const noexcept {
        return !(xy[0] < width & xy[1] < height);
    }
    [[nodiscard]] inline auto _out_of_bounds(lc_uint3 xyz) const noexcept {
        return !(xyz[0] < width & xyz[1] < height & xyz[2] < depth);
    }
    template<typename V, typename T>
    [[nodiscard]] inline V read2d(lc_uint2 xy) const noexcept {
        if (_out_of_bounds(xy)) [[unlikely]] { return {}; }
        return detail::read_pixel<V, T>(LCPixelStorage(storage), _pixel2d(xy));
    }
    template<typename V, typename T>
    [[nodiscard]] inline V read3d(lc_uint3 xyz) const noexcept {
        if (_out_of_bounds(xyz)) [[unlikely]] { return {}; }
        return detail::read_pixel<V, T>(LCPixelStorage(storage), _pixel3d(xyz));
    }
    template<typename V, typename T>
    inline void write2d(lc_uint2 xy,V value) const noexcept {
        if (_out_of_bounds(xy)) [[unlikely]] { return; }
        detail::write_pixel<V, T>(LCPixelStorage(storage), _pixel2d(xy), value);
    }
    template<typename V, typename T>
    inline void write3d(lc_uint3 xyz, V value) const noexcept {
        if (_out_of_bounds(xyz)) [[unlikely]] { return; }
        detail::write_pixel<V, T>(LCPixelStorage(storage), _pixel3d(xyz), value);
    }
    [[nodiscard]] auto size2d() const noexcept { return lc_make_uint2(width, height); }
    [[nodiscard]] auto size3d() const noexcept { return lc_make_uint3(width, height, depth); }
    [[nodiscard]] auto size_bytes() const noexcept { return (width * height * depth) << pixel_stride_shift; }
};


#define LUISA_MAKE_TEXTURE_RW(dim, type)                               \
    [[nodiscard]] inline lc_##type##4 texture_read_##dim##d_##type(    \
        TextureView view, lc_##uint##dim c) noexcept {                 \
        return view.read##dim##d<type>(c);                             \
    }                                                                  \
    inline void texture_write_##dim##d_##type(                         \
        TextureView view, lc_##uint##dim c, lc_##type##4 v) noexcept { \
        view.write##dim##d<type>(c, v);                                \
    }
// LUISA_MAKE_TEXTURE_RW(2, int)
// LUISA_MAKE_TEXTURE_RW(2, uint)
// LUISA_MAKE_TEXTURE_RW(2, float)
// LUISA_MAKE_TEXTURE_RW(3, int)
// LUISA_MAKE_TEXTURE_RW(3, uint)
// LUISA_MAKE_TEXTURE_RW(3, float)
#undef LUISA_MAKE_TEXTURE_RW

template<typename T>
[[nodiscard]] inline auto texture_coord_point(LCSamplerAddress address, T uv, T s) noexcept {
    switch (address) {
        case LC_SAMPLER_ADDRESS_EDGE: return lc_clamp(uv, T(0.0f), T(one_minus_epsilon)) * s;
        case LC_SAMPLER_ADDRESS_REPEAT: return lc_fract(uv) * s;
        case LC_SAMPLER_ADDRESS_MIRROR: {
            uv = lc_fmod(lc_abs(uv), T{2.0f});
            uv = lc_select(2.f - uv, uv, uv < T{1.f});
            return lc_min(uv, T(one_minus_epsilon)) * s;
        }
        case LC_SAMPLER_ADDRESS_ZERO: return lc_select(uv * s, T{65536.f}, uv < 0.f || uv >= 1.f);
    }
    return T{65536.f};
}

[[nodiscard]] inline auto texture_coord_linear(LCSamplerAddress address, lc_float2 uv, lc_float2 size) noexcept {
    auto s = lc_make_float2(size);
    auto inv_s = 1.f / s;
    auto c_min = texture_coord_point(address, uv - .5f * inv_s, s);
    auto c_max = texture_coord_point(address, uv + .5f * inv_s, s);
    return detail::lc_make_pair(lc_min(c_min, c_max), lc_max(c_min, c_max));
}

[[nodiscard]] inline auto texture_coord_linear(LCSamplerAddress address, lc_float3 uv, lc_float3 size) noexcept {
    auto s = lc_make_float3(size);
    auto inv_s = 1.f / s;
    auto c_min = texture_coord_point(address, uv - .5f * inv_s, s);
    auto c_max = texture_coord_point(address, uv + .5f * inv_s, s);
    return detail::lc_make_pair(lc_min(c_min, c_max), lc_max(c_min, c_max));
}

[[nodiscard]] inline lc_float4 texture_sample_linear(TextureView view, LCSamplerAddress address, lc_float2 uv) noexcept {
    auto size = lc_make_float2(view.size2d());
    auto [st_min, st_max] = texture_coord_linear(address, uv, size);
    auto t = lc_fract(st_max);
    auto c0 = lc_make_uint2(st_min);
    auto c1 = lc_make_uint2(st_max);
    auto v00 = view.read2d<lc_float4, float>(c0);
    auto v01 = view.read2d<lc_float4, float>(lc_make_uint2(c1.x, c0.y));
    auto v10 = view.read2d<lc_float4, float>(lc_make_uint2(c0.x, c1.y));
    auto v11 = view.read2d<lc_float4, float>(c1);
    return lc_lerp(lc_lerp(v00, v01, lc_float4(t.x)),
                       lc_lerp(v10, v11, lc_float4(t.x)), lc_float4(t.y));
}

[[nodiscard]] inline lc_float4 texture_sample_linear(TextureView view, LCSamplerAddress address, lc_float3 uvw) noexcept {
    auto size = lc_make_float3(view.size3d());
    auto [st_min, st_max] = texture_coord_linear(address, uvw, size);
    auto t = lc_fract(st_max);
    auto c0 = lc_make_uint3(st_min);
    auto c1 = lc_make_uint3(st_max);
    auto v000 = view.read3d<lc_float4, float>(lc_make_uint3(c0.x, c0.y, c0.z));
    auto v001 = view.read3d<lc_float4, float>(lc_make_uint3(c1.x, c0.y, c0.z));
    auto v010 = view.read3d<lc_float4, float>(lc_make_uint3(c0.x, c1.y, c0.z));
    auto v011 = view.read3d<lc_float4, float>(lc_make_uint3(c1.x, c1.y, c0.z));
    auto v100 = view.read3d<lc_float4, float>(lc_make_uint3(c0.x, c0.y, c1.z));
    auto v101 = view.read3d<lc_float4, float>(lc_make_uint3(c1.x, c0.y, c1.z));
    auto v110 = view.read3d<lc_float4, float>(lc_make_uint3(c0.x, c1.y, c1.z));
    auto v111 = view.read3d<lc_float4, float>(lc_make_uint3(c1.x, c1.y, c1.z));
    return lc_lerp(
        lc_lerp(lc_lerp(v000, v001, lc_float4(t.x)),
                    lc_lerp(v010, v011, lc_float4(t.x)), lc_float4(t.y)),
        lc_lerp(lc_lerp(v100, v101, lc_float4(t.x)),
                    lc_lerp(v110, v111, lc_float4(t.x)), lc_float4(t.y)),
        lc_float4(t.z));
}

[[nodiscard]] inline lc_float4 texture_sample_point(TextureView view, LCSamplerAddress address, lc_float2 uv) noexcept {
    auto size = lc_make_float2(view.size2d());
    auto c = lc_make_uint2(texture_coord_point(address, uv, size));
    return view.read2d<lc_float4, float>(c);
}

[[nodiscard]] inline lc_float4 texture_sample_point(TextureView view, LCSamplerAddress address, lc_float3 uvw) noexcept {
    auto size = lc_make_float3(view.size3d());
    auto c = lc_make_uint3(texture_coord_point(address, uvw, size));
    return view.read3d<lc_float4, float>(c);
}

// from PBRT-v4
[[nodiscard]] inline auto texture_sample_ewa(TextureView view, LCSamplerAddress address,
                                             lc_float2 uv, lc_float2 dst0, lc_float2 dst1) noexcept {
    auto size = lc_make_float2(view.size2d());
    auto st = uv * size - .5f;
    dst0 = dst0 * size;
    dst1 = dst1 * size;

    constexpr auto sqr = [](float x) noexcept { return x * x; };
    constexpr auto safe_sqrt = [](float x) noexcept { return lc_select(sqrtf(x), 0.f, x <= 0.f); };

    // Find ellipse coefficients that bound EWA filter region
    auto A = sqr(dst0.y) + sqr(dst1.y) + 1.f;
    auto B = -2.f * (dst0.x * dst0.y + dst1.x * dst1.y);
    auto C = sqr(dst0.x) + sqr(dst1.x) + 1.f;
    auto inv_f = 1.f / (A * C - sqr(B) * 0.25f);
    A *= inv_f;
    B *= inv_f;
    C *= inv_f;

    // Compute the ellipse's $(s,t)$ bounding box in texture space
    auto det = -sqr(B) + 4.f * A * C;
    auto inv_det = 1.f / det;
    auto sqrt_u = safe_sqrt(det * C);
    auto sqrt_v = safe_sqrt(A * det);
    auto s_min = static_cast<int>(ceilf(st.x - 2.f * inv_det * sqrt_u));
    auto s_max = static_cast<int>(floorf(st.x + 2.f * inv_det * sqrt_u));
    auto t_min = static_cast<int>(ceilf(st.y - 2.f * inv_det * sqrt_v));
    auto t_max = static_cast<int>(floorf(st.y + 2.f * inv_det * sqrt_v));

    // Scan over ellipse bound and evaluate quadratic equation to filter image
    auto sum = lc_make_float4();
    auto sum_w = 0.f;
    auto inv_size = 1.f / size;
    for (auto t = t_min; t <= t_max; t++) {
        for (auto s = s_min; s <= s_max; s++) {
            auto ss = static_cast<float>(s) - st.x;
            auto tt = static_cast<float>(t) - st.y;
            // Compute squared radius and filter texel if it is inside the ellipse
            if (auto rr = A * sqr(ss) + B * ss * tt + C * sqr(tt); rr < 1.f) {
                constexpr auto lut_size = static_cast<float>(detail::ewa_filter_weight_lut_size);
                auto index = lc_clamp(rr * lut_size, 0.f, lut_size - 1.f);
                auto weight = detail::ewa_filter_weight_lut[static_cast<int>(index)];
                auto p = texture_coord_point(address, uv + lc_make_float2(ss, tt) * inv_size, size);
                sum += weight * view.read2d<lc_float4, float>(lc_make_uint2(p));
                sum_w += weight;
            }
        }
    }
    return lc_select(sum / sum_w, lc_make_float4(0.f), sum_w <= 0.f);
}

[[nodiscard]] inline auto texture_sample_ewa(TextureView view, LCSamplerAddress address,
                                             lc_float3 uvw, lc_float3 ddx, lc_float3 ddy) noexcept {
    // FIXME: anisotropic filtering
    return texture_sample_linear(view, address, uvw);
}
[[nodiscard]] inline TextureView lc_texture_view(const Texture* tex, lc_uint level) noexcept {
    auto size = lc_max(lc_make_uint3(tex->width, tex->height, tex->depth) >> level, lc_make_uint3(1u));
    return TextureView{tex->data + (static_cast<size_t>(tex->mip_offsets[level]) << tex->pixel_stride_shift),
                           tex->dimension, size.x, size.y, size.z, tex->storage, tex->pixel_stride_shift};
}
struct LCSampler {
    LCSamplerAddress address;
    LCSamplerFilter filter;
};
[[nodiscard]] inline lc_float4 lc_texture_2d_read(
    const Texture *tex, lc_uint level, lc_uint2 uv) noexcept {
    return lc_texture_view(tex, level).read2d<lc_float4, float>(uv);
}

[[nodiscard]] inline lc_float4 lc_texture_3d_read(
    const Texture *tex, lc_uint level, lc_uint3 uvw) noexcept {
    return lc_texture_view(tex, level).read3d<lc_float4, float>(uvw);
}

[[nodiscard]] inline lc_float4 lc_texture_2d_sample(
    const Texture *tex, LCSampler sampler, lc_float2 uv) noexcept {
    auto view = lc_texture_view(tex, 0u);
    if (sampler.filter == LC_SAMPLER_FILTER_POINT) {
        return texture_sample_point(view, sampler.address, uv);
    } else {
        return texture_sample_linear(view, sampler.address, uv);
    }
}

[[nodiscard]] inline lc_float4 lc_texture_3d_sample(
    const Texture *tex, LCSampler sampler, lc_float3 uvw) noexcept {
    auto view = lc_texture_view(tex, 0u);
    if (sampler.filter == LC_SAMPLER_FILTER_POINT) {
        return texture_sample_point(view, sampler.address, uvw);
    } else {
        return texture_sample_linear(view, sampler.address, uvw);
    }
}

[[nodiscard]] inline lc_float4 lc_texture_2d_sample_level(
    const Texture *tex, LCSampler sampler, lc_float2 uv, lc_float lod) noexcept {
    auto filter = sampler.filter;
    if (lod <= 0.0f || tex->mip_levels == 0 || filter == LC_SAMPLER_FILTER_POINT) {
        return lc_texture_2d_sample(tex, sampler, uv);
    }
    auto level0 = std::min<uint>(static_cast<uint>(lod), tex->mip_levels - 1u);
    auto v0 = texture_sample_linear(lc_texture_view(tex, level0), sampler.address, uv);
    if (level0 == tex->mip_levels - 1u || filter == LC_SAMPLER_FILTER_LINEAR_POINT) { return v0; }
    auto v1 = texture_sample_linear(lc_texture_view(tex, level0 + 1u), sampler.address, uv);
    return lc_lerp(v0, v1, lc_float4(lod - level0));
}

[[nodiscard]] inline lc_float4 lc_texture_3d_sample_level(
    const Texture *tex, LCSampler sampler, lc_float3 uvw, lc_float lod) noexcept {
    auto filter = sampler.filter;
    if (lod <= 0.0f || tex->mip_levels == 0 || filter == LC_SAMPLER_FILTER_POINT) {
        return lc_texture_3d_sample(tex, sampler, uvw);
    }
    auto level0 = std::min<uint>(static_cast<uint>(lod), tex->mip_levels - 1u);
    auto v0 = texture_sample_linear(lc_texture_view(tex, level0), sampler.address, uvw);
    if (level0 == tex->mip_levels - 1u || filter == LC_SAMPLER_FILTER_LINEAR_POINT) { return v0; }
    auto v1 = texture_sample_linear(lc_texture_view(tex, level0 + 1u), sampler.address, uvw);
    return lc_lerp(v0, v1, lc_float4(lod - level0));
}

[[nodiscard]] inline lc_float4 lc_texture_2d_sample(
    const Texture *tex, LCSampler sampler, lc_float2 uv,
    lc_float2 dpdx, lc_float2 dpdy) noexcept {
    constexpr auto ll = [](lc_float2 v) noexcept { return lc_dot(v, v); };
    if(lc_all(dpdx == 0.0) || lc_all(dpdy == 0.0)) {
        return lc_texture_2d_sample(tex, sampler, uv);
    }
    if(sampler.filter != LC_SAMPLER_FILTER_ANISOTROPIC) {
        auto s = lc_make_float2(tex->width, tex->height);
        auto level = 0.5f * log2f(lc_max(ll(dpdx * s), ll(dpdy * s)));
        return lc_texture_2d_sample_level(tex, sampler, uv, level);
    }
    auto len_dpdx = lc_length(dpdx);
    auto len_dpdy = lc_length(dpdy);
    auto longer = lc_max(len_dpdx, len_dpdy);
    auto shorter = lc_min(len_dpdx, len_dpdy);
     constexpr auto max_anisotropy = 16.f;
    auto s = shorter * max_anisotropy;
    if (s != 0.0 && s < longer) {
        auto scale = longer / s;
        dpdy *= scale;
        shorter *= scale;
    }
    auto last_level = static_cast<float>(tex->mip_levels - 1u);
    auto level = lc_clamp(last_level + log2f(shorter), 0.f, last_level);
    auto level_uint = static_cast<uint>(level);
    auto v0 = texture_sample_ewa(lc_texture_view(tex, level_uint), sampler.address, uv, dpdx, dpdy);
    if (level == 0.0 || level == last_level) { return v0; }
    auto v1 = texture_sample_ewa(lc_texture_view(tex, level_uint + 1u), sampler.address, uv, dpdx, dpdy);
    return lc_lerp(v0, v1, lc_make_float4(level - level_uint));

}

[[nodiscard]] inline lc_float4 lc_texture_3d_sample_grad(
    const Texture *tex, LCSampler sampler, lc_float3 uvw, lc_float3 dpdx, lc_float3 dpdy) noexcept {
    constexpr auto ll = [](lc_float3 v) noexcept { return lc_dot(v, v); };
    if(lc_all(dpdx == 0.0) || lc_all(dpdy == 0.0)) {
        return lc_texture_3d_sample(tex, sampler, uvw);
    }
    if(sampler.filter != LC_SAMPLER_FILTER_ANISOTROPIC) {
        auto s = lc_make_float3(tex->width, tex->height, tex->depth);
        auto level = 0.5f * log2f(lc_max(ll(dpdx * s), ll(dpdy * s)));
        return lc_texture_3d_sample_level(tex, sampler, uvw, level);
    }
    auto len_dpdx = lc_length(dpdx);
    auto len_dpdy = lc_length(dpdy);
    auto longer = lc_max(len_dpdx, len_dpdy);
    auto shorter = lc_min(len_dpdx, len_dpdy);
     constexpr auto max_anisotropy = 16.f;
    auto s = shorter * max_anisotropy;
    if (s != 0.0 && s < longer) {
        auto scale = longer / s;
        dpdy *= scale;
        shorter *= scale;
    }
    auto last_level = static_cast<float>(tex->mip_levels - 1u);
    auto level = lc_clamp(last_level + log2f(shorter), 0.f, last_level);
    auto level_uint = static_cast<uint>(level);
    auto v0 = texture_sample_ewa(lc_texture_view(tex, level_uint), sampler.address, uvw, dpdx, dpdy);
    if (level == 0.0 || level == last_level) { return v0; }
    auto v1 = texture_sample_ewa(lc_texture_view(tex, level_uint + 1u), sampler.address, uvw, dpdx, dpdy);
    return lc_lerp(v0, v1, lc_make_float4(level - level_uint));
}
