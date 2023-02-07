#ifndef DEBUG_DISABLE_TEXTURE

#if defined(__x86_64__) || defined(_M_X64)
#define LUISA_ARCH_X86_64
#elif defined(__aarch64__)
#define LUISA_ARCH_ARM64
#endif

#if defined(LUISA_ARCH_ARM64)
#include <arm_neon.h>
namespace luisa {
using float16_t = ::float16_t;
}// namespace luisa
#else
#ifdef LLUISA_ARCH_X86_64
#include <immintrin.h>
#endif
namespace luisa {
using float16_t = int16_t;
}// namespace luisa
#endif
namespace detail {
template<class A, class B>
struct lc_pair {
    A first;
    B second;
};

float16_t float_to_half(float f) noexcept {
#if defined(LUISA_ARCH_ARM64)
    return static_cast<float16_t>(f);
#elif defined(LUISA_ARCH_X86_64)
    auto ss = _mm_set_ss(f);
    auto ph = _mm_cvtps_ph(ss, 0);
    return static_cast<float16_t>(_mm_cvtsi128_si32(ph));
#else
    auto bits = luisa::bit_cast<uint>(f);
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

}// namespace detail

#define LUISA_MAKE_TEXTURE_RW(dim, type)                               \
    [[nodiscard]] inline lc_##type##4 texture_read_##dim##d_##type(    \
        TextureView view, lc_##uint##dim c) noexcept {                 \
        return view.read##dim##d<type>(c);                             \
    }                                                                  \
    inline void texture_write_##dim##d_##type(                         \
        TextureView view, lc_##uint##dim c, lc_##type##4 v) noexcept { \
        view.write##dim##d<type>(c, v);                                \
    }
LUISA_MAKE_TEXTURE_RW(2, int)
LUISA_MAKE_TEXTURE_RW(2, uint)
LUISA_MAKE_TEXTURE_RW(2, float)
LUISA_MAKE_TEXTURE_RW(3, int)
LUISA_MAKE_TEXTURE_RW(3, uint)
LUISA_MAKE_TEXTURE_RW(3, float)
#undef LUISA_MAKE_TEXTURE_RW


[[nodiscard]] inline lc_float4 bindless_texture_2d_read(
    const LLVMTexture *tex, lc_uint level, lc_uint2 uv) noexcept {
    return tex->read2d(level, uv);
}

[[nodiscard]] inline lc_float4 bindless_texture_3d_read(
    const LLVMTexture *tex, lc_uint level, lc_uint3 uvw) noexcept {
    return tex->read3d(level, uvw);
}

[[nodiscard]] inline lc_float4 bindless_texture_2d_sample(
    const LLVMTexture *tex, lc_uint sampler, lc_float2 uv) noexcept {
    return tex->sample2d(sampler, uv);
}

[[nodiscard]] inline lc_float4 bindless_texture_3d_sample(
    const LLVMTexture *tex, lc_uint sampler, lc_float3 uvw) noexcept {
    return tex->sample3d(sampler, uvw);
}

[[nodiscard]] inline lc_float4 bindless_texture_2d_sample_level(
    const LLVMTexture *tex, lc_uint sampler, lc_float2 uv, lc_float lod) noexcept {
    return tex->sample2d(sampler, uv, lod);
}

[[nodiscard]] inline lc_float4 bindless_texture_3d_sample_level(
    const LLVMTexture *tex, lc_uint sampler, lc_float3 uvw, lc_float lod) noexcept {
    return tex->sample3d(sampler, uvw, lod);
}

[[nodiscard]] inline lc_float4 bindless_texture_2d_sample_grad(
    const LLVMTexture *tex, lc_uint sampler, lc_float2 uv,
    lc_float2 dpdx, lc_float2 dpdy) noexcept {
    return tex->sample2d(sampler, uv, dpdx, dpdy);
}

[[nodiscard]] inline lc_float4 bindless_texture_3d_sample_grad(
    const LLVMTexture *tex, lc_uint sampler, lc_float3 uvw, lc_float3 dpdx, lc_float3 dpdy) noexcept {
    return tex->sample3d(sampler, uvw, dpdx, dpdy);
}

#endif
