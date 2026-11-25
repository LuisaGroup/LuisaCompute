#include "fallback_texture.h"
#include "fallback_texture_bc.h"
#include "fallback_accel.h"
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string_view>
#if defined(_M_X64) || defined(__x86_64__)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#elif (defined(_M_ARM64) || defined(__aarch64__)) && !defined(__clang__)
#if __has_include(<arm_neon.h>)
#include <arm_neon.h>
#define LUISA_FALLBACK_HAS_ARM_NEON_HEADER 1
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif
#include <luisa/core/stl/memory.h>

namespace luisa::compute::fallback::api {

namespace detail_api_profile {

[[nodiscard]] inline uint64_t read_tsc() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
    return __rdtsc();
#elif defined(_M_ARM64) || defined(__aarch64__)
#if defined(_MSC_VER) && !defined(__clang__)
    return static_cast<uint64_t>(_ReadStatusReg(ARM64_CNTVCT));
#else
    uint64_t cnt = 0ull;
    asm volatile("mrs %0, cntvct_el0" : "=r"(cnt));
    return cnt;
#endif
#else
    return 0ull;
#endif
}

inline const bool g_enabled = []() noexcept {
    const char *raw = std::getenv("LUISA_FALLBACK_API_PROFILE");
    return raw != nullptr && std::string_view{raw} == "1";
}();

[[nodiscard]] inline bool enabled() noexcept {
    return g_enabled;
}

struct Counters {
    uint64_t trace_closest_calls = 0;
    uint64_t trace_closest_tsc = 0;
    uint64_t rq_all_calls = 0;
    uint64_t rq_all_total_tsc = 0;
    uint64_t rq_all_callback_calls = 0;
    uint64_t rq_all_callback_tsc = 0;
    uint64_t trace_any_calls = 0;
    uint64_t trace_any_tsc = 0;
    uint64_t rq_any_calls = 0;
    uint64_t rq_any_total_tsc = 0;
    uint64_t rq_any_callback_calls = 0;
    uint64_t rq_any_callback_tsc = 0;
    uint64_t tex2d_sample_calls = 0;
    uint64_t tex2d_sample_tsc = 0;
    uint64_t tex2d_in_shadow_cb_tsc = 0;
    uint64_t tex3d_sample_calls = 0;
    uint64_t tex3d_sample_tsc = 0;

    void add(const Counters &o) noexcept {
        trace_closest_calls += o.trace_closest_calls;
        trace_closest_tsc += o.trace_closest_tsc;
        rq_all_calls += o.rq_all_calls;
        rq_all_total_tsc += o.rq_all_total_tsc;
        rq_all_callback_calls += o.rq_all_callback_calls;
        rq_all_callback_tsc += o.rq_all_callback_tsc;
        trace_any_calls += o.trace_any_calls;
        trace_any_tsc += o.trace_any_tsc;
        rq_any_calls += o.rq_any_calls;
        rq_any_total_tsc += o.rq_any_total_tsc;
        rq_any_callback_calls += o.rq_any_callback_calls;
        rq_any_callback_tsc += o.rq_any_callback_tsc;
        tex2d_sample_calls += o.tex2d_sample_calls;
        tex2d_sample_tsc += o.tex2d_sample_tsc;
        tex2d_in_shadow_cb_tsc += o.tex2d_in_shadow_cb_tsc;
        tex3d_sample_calls += o.tex3d_sample_calls;
        tex3d_sample_tsc += o.tex3d_sample_tsc;
    }
};

struct GlobalSink {
    std::mutex mtx;
    Counters total{};
    void flush(Counters &local) noexcept {
        std::scoped_lock lk{mtx};
        total.add(local);
        local = Counters{};
    }
};

inline GlobalSink &sink() noexcept {
    static GlobalSink s;
    return s;
}

struct ThreadSlot {
    Counters c{};
    uint32_t in_shadow_cb = 0u;
    ~ThreadSlot() noexcept {
        if (enabled()) { sink().flush(c); }
    }
};

inline thread_local ThreadSlot tls_slot{};

} // namespace detail_api_profile

void luisa_fallback_dump_api_profile_if_enabled() noexcept {
    if (!detail_api_profile::enabled()) { return; }
    detail_api_profile::sink().flush(detail_api_profile::tls_slot.c);
    detail_api_profile::Counters snap{};
    {
        std::scoped_lock lk{detail_api_profile::sink().mtx};
        snap = detail_api_profile::sink().total;
        detail_api_profile::sink().total = detail_api_profile::Counters{};
    }
    std::printf(
        "LUISA_FALLBACK_API_PROFILE_JSON: "
        "{\"trace_closest_calls\":%llu,\"trace_closest_tsc\":%llu,"
        "\"rq_all_calls\":%llu,\"rq_all_total_tsc\":%llu,"
        "\"rq_all_callback_calls\":%llu,\"rq_all_callback_tsc\":%llu,"
        "\"trace_any_calls\":%llu,\"trace_any_tsc\":%llu,"
        "\"rq_any_calls\":%llu,\"rq_any_total_tsc\":%llu,"
        "\"rq_any_callback_calls\":%llu,\"rq_any_callback_tsc\":%llu,"
        "\"tex2d_sample_calls\":%llu,\"tex2d_sample_tsc\":%llu,\"tex2d_in_shadow_cb_tsc\":%llu,"
        "\"tex3d_sample_calls\":%llu,\"tex3d_sample_tsc\":%llu}\n",
        static_cast<unsigned long long>(snap.trace_closest_calls),
        static_cast<unsigned long long>(snap.trace_closest_tsc),
        static_cast<unsigned long long>(snap.rq_all_calls),
        static_cast<unsigned long long>(snap.rq_all_total_tsc),
        static_cast<unsigned long long>(snap.rq_all_callback_calls),
        static_cast<unsigned long long>(snap.rq_all_callback_tsc),
        static_cast<unsigned long long>(snap.trace_any_calls),
        static_cast<unsigned long long>(snap.trace_any_tsc),
        static_cast<unsigned long long>(snap.rq_any_calls),
        static_cast<unsigned long long>(snap.rq_any_total_tsc),
        static_cast<unsigned long long>(snap.rq_any_callback_calls),
        static_cast<unsigned long long>(snap.rq_any_callback_tsc),
        static_cast<unsigned long long>(snap.tex2d_sample_calls),
        static_cast<unsigned long long>(snap.tex2d_sample_tsc),
        static_cast<unsigned long long>(snap.tex2d_in_shadow_cb_tsc),
        static_cast<unsigned long long>(snap.tex3d_sample_calls),
        static_cast<unsigned long long>(snap.tex3d_sample_tsc));
    std::fflush(stdout);
}

void luisa_bc7_read(const FallbackTextureView *tex, uint x, uint y, float4 &out) noexcept {
    auto size = tex->size2d();
    if (size.x == 0u || size.y == 0u) {
        out = {};
        return;
    }
    auto clamp_coord = [](auto v, uint dim) noexcept -> uint {
        if constexpr (std::is_signed_v<decltype(v)>) {
            return v <= 0 ? 0u : luisa::min(static_cast<uint>(v), dim - 1u);
        } else {
            return luisa::min(static_cast<uint>(v), dim - 1u);
        }
    };
    auto ux = clamp_coord(x, size.x);
    auto uy = clamp_coord(y, size.y);
    auto block_pos = make_uint2(ux / 4u, uy / 4u);
    auto block_per_row = (size.x + 3u) / 4u;
    const bc::BC7Block *bc_block = reinterpret_cast<const bc::BC7Block *>(tex->data()) +
                                   (block_pos.x + block_pos.y * block_per_row);
    bc_block->Decode(ux % 4u, uy % 4u, reinterpret_cast<float *>(&out));
}

void luisa_bc6h_read(const FallbackTextureView *tex, int x, int y, float4 &out) noexcept {
    auto size = tex->size2d();
    if (size.x == 0u || size.y == 0u) {
        out = {};
        return;
    }
    auto clamp_coord = [](auto v, uint dim) noexcept -> uint {
        if constexpr (std::is_signed_v<decltype(v)>) {
            return v <= 0 ? 0u : luisa::min(static_cast<uint>(v), dim - 1u);
        } else {
            return luisa::min(static_cast<uint>(v), dim - 1u);
        }
    };
    auto ux = clamp_coord(x, size.x);
    auto uy = clamp_coord(y, size.y);
    auto block_pos = make_uint2(ux / 4u, uy / 4u);
    auto block_per_row = (size.x + 3u) / 4u;
    const bc::BC6HBlock *bc_block = reinterpret_cast<const bc::BC6HBlock *>(tex->data()) + (block_pos.x + block_pos.y * block_per_row);
    bc_block->Decode(false, ux % 4u, uy % 4u, reinterpret_cast<float *>(&out));
}

[[nodiscard]] int4 luisa_fallback_texture2d_read_int(void *texture_data, uint64_t texture_data_extra, uint x, uint y) noexcept {
    PackedTextureView view{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 2D int texture");
    auto v = tex->read2d<int>(make_uint2(x, y));
    return {v.x, v.y, v.z, v.w};
}

[[nodiscard]] uint4 luisa_fallback_texture2d_read_uint(void *texture_data, uint64_t texture_data_extra, uint x, uint y) noexcept {
    PackedTextureView view{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 2D uint texture");
    auto v = tex->read2d<uint>(make_uint2(x, y));
    return {v.x, v.y, v.z, v.w};
}

[[nodiscard]] float4 luisa_fallback_texture2d_read_float(void *texture_data, uint64_t texture_data_extra, uint x, uint y) noexcept {
    PackedTextureView view{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    switch (tex->storage()) {
        case PixelStorage::BC7: {
            float4 out{};
            luisa_bc7_read(tex, x, y, out);
            return out;
        }
        case PixelStorage::BC6: {
            float4 out{};
            luisa_bc6h_read(tex, x, y, out);
            return out;
        }
        default: {
            auto v = tex->read2d<float>(make_uint2(x, y));
            return {v.x, v.y, v.z, v.w};
        }
    }
}

void luisa_fallback_texture2d_write_float(void *texture_data, uint64_t texture_data_extra, uint x, uint y, float4 value) noexcept {
    PackedTextureView view{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 2D texture write");
    auto v = luisa::make_float4(value.x, value.y, value.z, value.w);
    tex->write2d<float>(make_uint2(x, y), v);
}

void luisa_fallback_texture2d_write_uint(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint4 value) noexcept {
    PackedTextureView view{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 2D texture write");
    auto v = luisa::make_uint4(value.x, value.y, value.z, value.w);
    tex->write2d<uint>(make_uint2(x, y), v);
}

void luisa_fallback_texture2d_write_int(void *texture_data, uint64_t texture_data_extra, uint x, uint y, int4 value) noexcept {
    PackedTextureView view{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 2D texture write");
    auto v = luisa::make_int4(value.x, value.y, value.z, value.w);
    tex->write2d<int>(make_uint2(x, y), v);
}

[[nodiscard]] int4 luisa_fallback_texture3d_read_int(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z) noexcept {
    auto view = PackedTextureView{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 3D texture");
    auto v = tex->read3d<int>(make_uint3(x, y, z));
    return {v.x, v.y, v.z, v.w};
}

[[nodiscard]] uint4 luisa_fallback_texture3d_read_uint(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z) noexcept {
    auto view = PackedTextureView{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 3D texture");
    auto v = tex->read3d<uint>(make_uint3(x, y, z));
    return {v.x, v.y, v.z, v.w};
}

[[nodiscard]] float4 luisa_fallback_texture3d_read_float(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z) noexcept {
    auto view = PackedTextureView{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 3D texture");
    auto v = tex->read3d<float>(make_uint3(x, y, z));
    return {v.x, v.y, v.z, v.w};
}

void luisa_fallback_texture3d_write_float(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z, float4 value) noexcept {
    auto view = PackedTextureView{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 3D texture");
    auto v = luisa::make_float4(value.x, value.y, value.z, value.w);
    tex->write3d<float>(make_uint3(x, y, z), v);
}

void luisa_fallback_texture3d_write_uint(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z, uint4 value) noexcept {
    auto view = PackedTextureView{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 3D texture");
    auto v = luisa::make_uint4(value.x, value.y, value.z, value.w);
    tex->write3d<uint>(make_uint3(x, y, z), v);
}

void luisa_fallback_texture3d_write_int(void *texture_data, uint64_t texture_data_extra, uint x, uint y, uint z, int4 value) noexcept {
    auto view = PackedTextureView{texture_data, texture_data_extra};
    auto tex = reinterpret_cast<const FallbackTextureView *>(&view);
    LUISA_DEBUG_ASSERT(!is_block_compressed(tex->storage()), "Block compression doesn't work for 3D texture");
    auto v = luisa::make_int4(value.x, value.y, value.z, value.w);
    tex->write3d<int>(make_uint3(x, y, z), v);
}

template<typename T>
[[nodiscard]] inline auto texture_coord_point(Sampler::Address address, T uv, T s) noexcept {
    switch (address) {
        case Sampler::Address::EDGE: return luisa::clamp(uv, 0.0f, one_minus_epsilon) * s;
        case Sampler::Address::REPEAT: return luisa::fract(uv) * s;
        case Sampler::Address::MIRROR: {
            uv = luisa::fmod(luisa::abs(uv), T{2.f});
            uv = select(2.f - uv, uv, uv < T{1.f});
            return luisa::min(uv, one_minus_epsilon) * s;
        }
        case Sampler::Address::ZERO: return luisa::select(uv * s, T{65536.f}, uv < 0.f || uv >= 1.f);
    }
    return T{65536.f};
}

[[nodiscard]] inline auto texture_sample_point(FallbackTextureView view, Sampler::Address address, luisa::float2 uv) noexcept {
    auto size = make_float2(view.size2d());
    auto p = texture_coord_point(address, uv, size);
    auto c = make_uint2(p);
    auto handle = reinterpret_cast<const PackedTextureView *>(&view);
    return bit_cast<luisa::float4>(luisa_fallback_texture2d_read_float(handle->data, handle->extra, c.x, c.y));
}

inline void texture_coord_linear_1d(
    Sampler::Address address, float u, uint size,
    uint &c0, uint &c1, float &t) noexcept {
    if (size == 0u || !std::isfinite(u)) {
        c0 = size;
        c1 = size;
        t = 0.f;
        return;
    }
    const float fs = static_cast<float>(size);
    const int isize = static_cast<int>(size);
    switch (address) {
        case Sampler::Address::EDGE: {
            const float px = u * fs - 0.5f;
            if (px <= 0.f) {
                c0 = 0u;
                c1 = 0u;
                t = 0.f;
            } else if (px >= static_cast<float>(size - 1u)) {
                c0 = size - 1u;
                c1 = size - 1u;
                t = 0.f;
            } else {
                const float fl = std::floor(px);
                const uint i0 = static_cast<uint>(fl);
                c0 = i0 < size ? i0 : (size - 1u);
                c1 = (i0 + 1u) < size ? (i0 + 1u) : (size - 1u);
                t = px - fl;
            }
            return;
        }
        case Sampler::Address::REPEAT: {
            float f = u - std::floor(u);
            if (!(f >= 0.f && f < 1.f)) { f = 0.f; }
            const float px = f * fs - 0.5f;
            const float fl = std::floor(px);
            t = px - fl;
            const int i0 = static_cast<int>(fl);
            const int i1 = i0 + 1;
            const int w0 = i0 < 0 ? (isize - 1) : (i0 >= isize ? (isize - 1) : i0);
            const int w1 = i1 < 0 ? 0 : (i1 >= isize ? 0 : i1);
            c0 = static_cast<uint>(w0);
            c1 = static_cast<uint>(w1);
            return;
        }
        case Sampler::Address::MIRROR: {
            const float raw_a = std::abs(u);
            float a = raw_a - 2.f * std::floor(raw_a * 0.5f);
            if (!(a >= 0.f && a < 2.f)) { a = 0.f; }
            const float folded = a < 1.f ? a : (2.f - a);
            const float px = folded * fs - 0.5f;
            const float fl = std::floor(px);
            t = px - fl;
            const int i0 = static_cast<int>(fl);
            const int i1 = i0 + 1;
            const int w0 = i0 < 0 ? 0 : (i0 >= isize ? (isize - 1) : i0);
            const int w1 = i1 < 0 ? 0 : (i1 >= isize ? (isize - 1) : i1);
            c0 = static_cast<uint>(w0);
            c1 = static_cast<uint>(w1);
            return;
        }
        case Sampler::Address::ZERO: {
            const float px = u * fs - 0.5f;
            if (px <= -1.f || px >= fs) {
                c0 = size;
                c1 = size;
                t = 0.f;
            } else {
                const float fl = std::floor(px);
                t = px - fl;
                const int i0 = static_cast<int>(fl);
                c0 = (i0 < 0 || i0 >= isize) ? size : static_cast<uint>(i0);
                c1 = (i0 + 1 < 0 || i0 + 1 >= isize) ? size : static_cast<uint>(i0 + 1);
            }
            return;
        }
    }
    c0 = size;
    c1 = size;
    t = 0.f;
}

[[nodiscard]] inline luisa::float4 bilinear_rgba8_pair(
    const uint8_t *p00, const uint8_t *p01,
    const uint8_t *p10, const uint8_t *p11,
    float tx, float ty) noexcept {
#if defined(_M_X64) || defined(__x86_64__)
    const __m128i zero = _mm_setzero_si128();
    const __m128i u0 = _mm_unpacklo_epi8(_mm_set_epi32(0, 0, *reinterpret_cast<const int *>(p01), *reinterpret_cast<const int *>(p00)), zero);
    const __m128i u1 = _mm_unpacklo_epi8(_mm_set_epi32(0, 0, *reinterpret_cast<const int *>(p11), *reinterpret_cast<const int *>(p10)), zero);
    const __m128 f00 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(u0, zero));
    const __m128 f01 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(u0, zero));
    const __m128 f10 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(u1, zero));
    const __m128 f11 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(u1, zero));
    const __m128 vtx = _mm_set1_ps(tx);
    const __m128 vty = _mm_set1_ps(ty);
    const __m128 r0 = _mm_add_ps(f00, _mm_mul_ps(_mm_sub_ps(f01, f00), vtx));
    const __m128 r1 = _mm_add_ps(f10, _mm_mul_ps(_mm_sub_ps(f11, f10), vtx));
    const __m128 res = _mm_mul_ps(_mm_add_ps(r0, _mm_mul_ps(_mm_sub_ps(r1, r0), vty)), _mm_set1_ps(1.f / 255.f));
    alignas(16) float out[4];
    _mm_store_ps(out, res);
    return luisa::make_float4(out[0], out[1], out[2], out[3]);
#elif defined(LUISA_FALLBACK_HAS_ARM_NEON_HEADER)
    uint32_t w00 = 0u, w01 = 0u, w10 = 0u, w11 = 0u;
    std::memcpy(&w00, p00, 4u);
    std::memcpy(&w01, p01, 4u);
    std::memcpy(&w10, p10, 4u);
    std::memcpy(&w11, p11, 4u);
    const uint8x8_t u8_0 = vreinterpret_u8_u64(vcreate_u64(static_cast<uint64_t>(w00) | (static_cast<uint64_t>(w01) << 32u)));
    const uint8x8_t u8_1 = vreinterpret_u8_u64(vcreate_u64(static_cast<uint64_t>(w10) | (static_cast<uint64_t>(w11) << 32u)));
    const uint16x8_t u16_0 = vmovl_u8(u8_0);
    const uint16x8_t u16_1 = vmovl_u8(u8_1);
    const float32x4_t f00 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_0)));
    const float32x4_t f01 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_0)));
    const float32x4_t f10 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_1)));
    const float32x4_t f11 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_1)));
    const float32x4_t r0 = vfmaq_n_f32(f00, vsubq_f32(f01, f00), tx);
    const float32x4_t r1 = vfmaq_n_f32(f10, vsubq_f32(f11, f10), tx);
    const float32x4_t res = vmulq_n_f32(vfmaq_n_f32(r0, vsubq_f32(r1, r0), ty), 1.f / 255.f);
    alignas(16) float out[4];
    vst1q_f32(out, res);
    return luisa::make_float4(out[0], out[1], out[2], out[3]);
#elif defined(__clang__)
    using u8x4 = uint8_t __attribute__((ext_vector_type(4)));
    using f32x4 = float __attribute__((ext_vector_type(4)));
    u8x4 b00, b01, b10, b11;
    std::memcpy(&b00, p00, 4u);
    std::memcpy(&b01, p01, 4u);
    std::memcpy(&b10, p10, 4u);
    std::memcpy(&b11, p11, 4u);
    const f32x4 f00 = __builtin_convertvector(b00, f32x4);
    const f32x4 f01 = __builtin_convertvector(b01, f32x4);
    const f32x4 f10 = __builtin_convertvector(b10, f32x4);
    const f32x4 f11 = __builtin_convertvector(b11, f32x4);
    const f32x4 r0 = f00 + (f01 - f00) * tx;
    const f32x4 r1 = f10 + (f11 - f10) * tx;
    const f32x4 res = (r0 + (r1 - r0) * ty) * (1.f / 255.f);
    return luisa::make_float4(res.x, res.y, res.z, res.w);
#else
    auto v00 = detail::pixel_to_float4<uint8_t, 4u>(reinterpret_cast<const std::byte *>(p00));
    auto v01 = detail::pixel_to_float4<uint8_t, 4u>(reinterpret_cast<const std::byte *>(p01));
    auto v10 = detail::pixel_to_float4<uint8_t, 4u>(reinterpret_cast<const std::byte *>(p10));
    auto v11 = detail::pixel_to_float4<uint8_t, 4u>(reinterpret_cast<const std::byte *>(p11));
    return luisa::lerp(luisa::lerp(v00, v01, tx), luisa::lerp(v10, v11, tx), ty);
#endif
}

[[nodiscard]] inline auto texture_sample_linear(FallbackTextureView view, Sampler::Address address, luisa::float2 uv) noexcept {
    const auto sz = view.size2d();
    const uint w = sz.x;
    const uint h = sz.y;
    const auto size = make_float2(sz);
    const auto storage = view.storage();

    // Fast division-free interior path for REPEAT and EDGE (covers >99% of UV lookups).
    if (w >= 2u && h >= 2u &&
        (address == Sampler::Address::REPEAT || address == Sampler::Address::EDGE)) {
        float ux = uv.x;
        float uy = uv.y;
        if (address == Sampler::Address::REPEAT) {
            ux -= std::floor(ux);
            uy -= std::floor(uy);
        }
        if (ux >= 0.f && ux < 1.f && uy >= 0.f && uy < 1.f) {
            const float px = ux * size.x - 0.5f;
            const float py = uy * size.y - 0.5f;
            if (px >= 0.f && px < size.x - 1.f && py >= 0.f && py < size.y - 1.f) [[likely]] {
                const uint x0 = static_cast<uint>(px);
                const uint y0 = static_cast<uint>(py);
                const float tx = px - static_cast<float>(x0);
                const float ty = py - static_cast<float>(y0);
                const auto *raw = reinterpret_cast<const uint8_t *>(view.data());
                switch (storage) {
                    case PixelStorage::BYTE4: {
                        const size_t stride = static_cast<size_t>(w) << 2u;
                        const uint8_t *r0 = raw + static_cast<size_t>(y0) * stride + (static_cast<size_t>(x0) << 2u);
                        const uint8_t *r1 = r0 + stride;
#if defined(_M_X64) || defined(__x86_64__)
                        const __m128i zero = _mm_setzero_si128();
                        const __m128i u0 = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(r0)), zero);
                        const __m128i u1 = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(r1)), zero);
                        const __m128 f00 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(u0, zero));
                        const __m128 f01 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(u0, zero));
                        const __m128 f10 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(u1, zero));
                        const __m128 f11 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(u1, zero));
                        const __m128 vtx = _mm_set1_ps(tx);
                        const __m128 vty = _mm_set1_ps(ty);
                        const __m128 row0 = _mm_add_ps(f00, _mm_mul_ps(_mm_sub_ps(f01, f00), vtx));
                        const __m128 row1 = _mm_add_ps(f10, _mm_mul_ps(_mm_sub_ps(f11, f10), vtx));
                        const __m128 res = _mm_mul_ps(
                            _mm_add_ps(row0, _mm_mul_ps(_mm_sub_ps(row1, row0), vty)),
                            _mm_set1_ps(1.f / 255.f));
                        alignas(16) float out[4];
                        _mm_store_ps(out, res);
                        return luisa::make_float4(out[0], out[1], out[2], out[3]);
#else
                        return bilinear_rgba8_pair(r0, r0 + 4u, r1, r1 + 4u, tx, ty);
#endif
                    }
                    case PixelStorage::BYTE4_SRGB: {
                        const size_t stride = static_cast<size_t>(w) << 2u;
                        const auto *r0 = reinterpret_cast<const std::byte *>(raw + static_cast<size_t>(y0) * stride + (static_cast<size_t>(x0) << 2u));
                        const auto *r1 = r0 + stride;
                        const auto v00 = detail::pixel_srgb8_to_float4(r0);
                        const auto v01 = detail::pixel_srgb8_to_float4(r0 + 4u);
                        const auto v10 = detail::pixel_srgb8_to_float4(r1);
                        const auto v11 = detail::pixel_srgb8_to_float4(r1 + 4u);
                        return luisa::lerp(luisa::lerp(v00, v01, tx), luisa::lerp(v10, v11, tx), ty);
                    }
                    case PixelStorage::FLOAT4: {
                        const auto *fraw = reinterpret_cast<const luisa::float4 *>(raw);
                        const size_t idx0 = static_cast<size_t>(y0) * w + x0;
                        const size_t idx1 = idx0 + w;
                        return luisa::lerp(
                            luisa::lerp(fraw[idx0], fraw[idx0 + 1u], tx),
                            luisa::lerp(fraw[idx1], fraw[idx1 + 1u], tx), ty);
                    }
                    case PixelStorage::BYTE1: {
                        const size_t idx0 = static_cast<size_t>(y0) * w + x0;
                        const size_t idx1 = idx0 + w;
                        const float r0 = static_cast<float>(raw[idx0]) + (static_cast<float>(raw[idx0 + 1u]) - static_cast<float>(raw[idx0])) * tx;
                        const float r1 = static_cast<float>(raw[idx1]) + (static_cast<float>(raw[idx1 + 1u]) - static_cast<float>(raw[idx1])) * tx;
                        return luisa::make_float4((r0 + (r1 - r0) * ty) * (1.f / 255.f), 0.f, 0.f, 0.f);
                    }
                    case PixelStorage::FLOAT1: {
                        const auto *fraw = reinterpret_cast<const float *>(raw);
                        const size_t idx0 = static_cast<size_t>(y0) * w + x0;
                        const size_t idx1 = idx0 + w;
                        const float r0 = fraw[idx0] + (fraw[idx0 + 1u] - fraw[idx0]) * tx;
                        const float r1 = fraw[idx1] + (fraw[idx1 + 1u] - fraw[idx1]) * tx;
                        return luisa::make_float4(r0 + (r1 - r0) * ty, 0.f, 0.f, 0.f);
                    }
                    case PixelStorage::HALF4: {
                        const size_t stride = static_cast<size_t>(w) << 3u;
                        const auto *r0 = reinterpret_cast<const std::byte *>(raw + static_cast<size_t>(y0) * stride + (static_cast<size_t>(x0) << 3u));
                        const auto *r1 = r0 + stride;
                        const auto v00 = detail::pixel_to_float4<luisa::half, 4u>(r0);
                        const auto v01 = detail::pixel_to_float4<luisa::half, 4u>(r0 + 8u);
                        const auto v10 = detail::pixel_to_float4<luisa::half, 4u>(r1);
                        const auto v11 = detail::pixel_to_float4<luisa::half, 4u>(r1 + 8u);
                        return luisa::lerp(luisa::lerp(v00, v01, tx), luisa::lerp(v10, v11, tx), ty);
                    }
                    default: break;
                }
            }
        }
    }

    luisa::uint2 c0{}, c1{};
    luisa::float2 t{};
    texture_coord_linear_1d(address, uv.x, w, c0.x, c1.x, t.x);
    texture_coord_linear_1d(address, uv.y, h, c0.y, c1.y, t.y);
    if ((c0.x < w) & (c1.x < w) & (c0.y < h) & (c1.y < h)) [[likely]] {
        const auto *raw = reinterpret_cast<const uint8_t *>(view.data());
        switch (storage) {
            case PixelStorage::BYTE4: {
                const size_t stride = static_cast<size_t>(w) << 2u;
                const uint8_t *row0 = raw + static_cast<size_t>(c0.y) * stride;
                const uint8_t *row1 = raw + static_cast<size_t>(c1.y) * stride;
                return bilinear_rgba8_pair(
                    row0 + (static_cast<size_t>(c0.x) << 2u),
                    row0 + (static_cast<size_t>(c1.x) << 2u),
                    row1 + (static_cast<size_t>(c0.x) << 2u),
                    row1 + (static_cast<size_t>(c1.x) << 2u),
                    t.x, t.y);
            }
            case PixelStorage::BYTE4_SRGB: {
                const size_t stride = static_cast<size_t>(w) << 2u;
                const auto *row0 = reinterpret_cast<const std::byte *>(raw + static_cast<size_t>(c0.y) * stride);
                const auto *row1 = reinterpret_cast<const std::byte *>(raw + static_cast<size_t>(c1.y) * stride);
                const auto v00 = detail::pixel_srgb8_to_float4(row0 + (static_cast<size_t>(c0.x) << 2u));
                const auto v01 = detail::pixel_srgb8_to_float4(row0 + (static_cast<size_t>(c1.x) << 2u));
                const auto v10 = detail::pixel_srgb8_to_float4(row1 + (static_cast<size_t>(c0.x) << 2u));
                const auto v11 = detail::pixel_srgb8_to_float4(row1 + (static_cast<size_t>(c1.x) << 2u));
                return luisa::lerp(luisa::lerp(v00, v01, t.x), luisa::lerp(v10, v11, t.x), t.y);
            }
            case PixelStorage::FLOAT4: {
                const auto *fraw = reinterpret_cast<const luisa::float4 *>(raw);
                const auto *row0 = fraw + static_cast<size_t>(c0.y) * w;
                const auto *row1 = fraw + static_cast<size_t>(c1.y) * w;
                return luisa::lerp(luisa::lerp(row0[c0.x], row0[c1.x], t.x),
                                   luisa::lerp(row1[c0.x], row1[c1.x], t.x), t.y);
            }
            case PixelStorage::BYTE1: {
                const auto *row0 = raw + static_cast<size_t>(c0.y) * w;
                const auto *row1 = raw + static_cast<size_t>(c1.y) * w;
                const float r0 = static_cast<float>(row0[c0.x]) + (static_cast<float>(row0[c1.x]) - static_cast<float>(row0[c0.x])) * t.x;
                const float r1 = static_cast<float>(row1[c0.x]) + (static_cast<float>(row1[c1.x]) - static_cast<float>(row1[c0.x])) * t.x;
                return luisa::make_float4((r0 + (r1 - r0) * t.y) * (1.f / 255.f), 0.f, 0.f, 0.f);
            }
            default: break;
        }
    }
    auto handle = reinterpret_cast<const PackedTextureView *>(&view);
    const auto fetch2d = [&](uint x, uint y) noexcept {
        return (x < w && y < h)
                   ? bit_cast<luisa::float4>(luisa_fallback_texture2d_read_float(handle->data, handle->extra, x, y))
                   : luisa::make_float4(0.f);
    };
    auto v00 = fetch2d(c0.x, c0.y);
    auto v01 = fetch2d(c1.x, c0.y);
    auto v10 = fetch2d(c0.x, c1.y);
    auto v11 = fetch2d(c1.x, c1.y);
    return luisa::lerp(luisa::lerp(v00, v01, t.x), luisa::lerp(v10, v11, t.x), t.y);
}

[[nodiscard]] float4 luisa_fallback_bindless_texture2d_sample(const Texture *handle, uint sampler, float u, float v) noexcept {
    const uint64_t t0 = detail_api_profile::enabled() ? detail_api_profile::read_tsc() : 0ull;
    auto tex = reinterpret_cast<const FallbackTexture *>(handle);
    auto s = Sampler::decode(sampler);
    auto view = tex->view(0);
    auto result = s.filter() == Sampler::Filter::POINT ?
                      texture_sample_point(view, s.address(), make_float2(u, v)) :
                      texture_sample_linear(view, s.address(), make_float2(u, v));
    if (detail_api_profile::enabled()) [[unlikely]] {
        const uint64_t dt = detail_api_profile::read_tsc() - t0;
        auto &slot = detail_api_profile::tls_slot;
        slot.c.tex2d_sample_calls += 1;
        slot.c.tex2d_sample_tsc += dt;
        if (slot.in_shadow_cb != 0u) { slot.c.tex2d_in_shadow_cb_tsc += dt; }
    }
    return {result.x, result.y, result.z, result.w};
}

[[nodiscard]] float4 luisa_fallback_bindless_texture2d_sample_level(const Texture *handle, uint sampler, float u, float v, float level) noexcept {
    auto tex = reinterpret_cast<const FallbackTexture *>(handle);
    auto s = Sampler::decode(sampler);
    auto filter = s.filter();
    if (level <= 0.f || tex->mip_levels() == 0u || filter == Sampler::Filter::POINT) {
        return luisa_fallback_bindless_texture2d_sample(handle, sampler, u, v);
    }
    auto level0 = std::min(static_cast<uint32_t>(level), tex->mip_levels() - 1u);
    auto view0 = tex->view(level0);
    auto v0 = texture_sample_linear(view0, s.address(), make_float2(u, v));
    if (level0 == tex->mip_levels() - 1u || filter == Sampler::Filter::LINEAR_POINT) {
        return {v0.x, v0.y, v0.z, v0.w};
    }
    auto view1 = tex->view(level0 + 1);
    auto v1 = texture_sample_linear(view1, s.address(), make_float2(u, v));
    auto result = luisa::lerp(v0, v1, luisa::fract(level));
    return {result.x, result.y, result.z, result.w};
}

// TODO: implement
[[nodiscard]] float4 luisa_fallback_bindless_texture2d_sample_grad(const Texture *handle, uint sampler, float u, float v, float dudx, float dudy, float dvdx, float dvdy) noexcept {
    return luisa_fallback_bindless_texture2d_sample(handle, sampler, u, v);
}

// TODO: implement
[[nodiscard]] float4 luisa_fallback_bindless_texture2d_sample_grad_level(const Texture *handle, uint sampler, float u, float v, float dudx, float dudy, float dvdx, float dvdy, float level) noexcept {
    return luisa_fallback_bindless_texture2d_sample_level(handle, sampler, u, v, level);
}

[[nodiscard]] inline auto texture_sample_point(FallbackTextureView view, Sampler::Address address, luisa::float3 uv) noexcept {
    auto size = make_float3(view.size3d());
    auto c = make_uint3(texture_coord_point(address, uv, size));
    auto handle = reinterpret_cast<const PackedTextureView *>(&view);
    return bit_cast<luisa::float4>(luisa_fallback_texture3d_read_float(handle->data, handle->extra, c.x, c.y, c.z));
}

[[nodiscard]] inline auto texture_sample_linear(FallbackTextureView view, Sampler::Address address, luisa::float3 uvw) noexcept {
    const auto sz = view.size3d();
    luisa::uint3 c0{}, c1{};
    luisa::float3 t{};
    texture_coord_linear_1d(address, uvw.x, sz.x, c0.x, c1.x, t.x);
    texture_coord_linear_1d(address, uvw.y, sz.y, c0.y, c1.y, t.y);
    texture_coord_linear_1d(address, uvw.z, sz.z, c0.z, c1.z, t.z);
    auto handle = reinterpret_cast<const PackedTextureView *>(&view);
    const auto fetch3d = [&](uint x, uint y, uint z) noexcept {
        return (x < sz.x && y < sz.y && z < sz.z)
                   ? bit_cast<luisa::float4>(luisa_fallback_texture3d_read_float(handle->data, handle->extra, x, y, z))
                   : luisa::make_float4(0.f);
    };
    auto v000 = fetch3d(c0.x, c0.y, c0.z);
    auto v001 = fetch3d(c1.x, c0.y, c0.z);
    auto v010 = fetch3d(c0.x, c1.y, c0.z);
    auto v011 = fetch3d(c1.x, c1.y, c0.z);
    auto v100 = fetch3d(c0.x, c0.y, c1.z);
    auto v101 = fetch3d(c1.x, c0.y, c1.z);
    auto v110 = fetch3d(c0.x, c1.y, c1.z);
    auto v111 = fetch3d(c1.x, c1.y, c1.z);
    return luisa::lerp(luisa::lerp(luisa::lerp(v000, v001, t.x),
                                   luisa::lerp(v010, v011, t.x), t.y),
                       luisa::lerp(luisa::lerp(v100, v101, t.x),
                                   luisa::lerp(v110, v111, t.x), t.y),
                       t.z);
}

[[nodiscard]] float4 luisa_fallback_bindless_texture3d_sample(const Texture *handle, uint sampler, float u, float v, float w) noexcept {
    const uint64_t t0 = detail_api_profile::enabled() ? detail_api_profile::read_tsc() : 0ull;
    auto tex = reinterpret_cast<const FallbackTexture *>(handle);
    auto s = Sampler::decode(sampler);
    auto view = tex->view(0);
    auto result = s.filter() == Sampler::Filter::POINT ?
                      texture_sample_point(view, s.address(), make_float3(u, v, w)) :
                      texture_sample_linear(view, s.address(), make_float3(u, v, w));
    if (detail_api_profile::enabled()) [[unlikely]] {
        auto &slot = detail_api_profile::tls_slot;
        slot.c.tex3d_sample_calls += 1;
        slot.c.tex3d_sample_tsc += detail_api_profile::read_tsc() - t0;
    }
    return {result.x, result.y, result.z, result.w};
}

[[nodiscard]] float4 luisa_fallback_bindless_texture3d_sample_level(const Texture *handle, uint sampler, float u, float v, float w, float level) noexcept {
    auto tex = reinterpret_cast<const FallbackTexture *>(handle);
    auto s = Sampler::decode(sampler);
    auto filter = s.filter();
    if (level <= 0.f || tex->mip_levels() == 0u || filter == Sampler::Filter::POINT) {
        return luisa_fallback_bindless_texture3d_sample(handle, sampler, u, v, w);
    }
    auto level0 = std::min(static_cast<uint32_t>(level), tex->mip_levels() - 1u);
    auto view0 = tex->view(level0);
    auto v0 = texture_sample_linear(view0, s.address(), make_float3(u, v, w));
    if (level0 == tex->mip_levels() - 1u || filter == Sampler::Filter::LINEAR_POINT) {
        return {v0.x, v0.y, v0.z, v0.w};
    }
    auto view1 = tex->view(level0 + 1);
    auto v1 = texture_sample_linear(view1, s.address(), make_float3(u, v, w));
    auto result = luisa::lerp(v0, v1, luisa::fract(level));
    return {result.x, result.y, result.z, result.w};
}

// TODO: implement
[[nodiscard]] float4 luisa_fallback_bindless_texture3d_sample_grad(const Texture *handle, uint sampler, float u, float v, float w, float dudx, float dvdx, float dwdx, float dudy, float dvdy, float dwdy) noexcept {
    return luisa_fallback_bindless_texture3d_sample(handle, sampler, u, v, w);
}

// TODO: implement
[[nodiscard]] float4 luisa_fallback_bindless_texture3d_sample_grad_level(const Texture *handle, uint sampler, float u, float v, float w, float dudx, float dvdx, float dwdx, float dudy, float dvdy, float dwdy, float level) noexcept {
    return luisa_fallback_bindless_texture3d_sample_level(handle, sampler, u, v, w, level);
}

[[nodiscard]] float4 luisa_fallback_bindless_texture2d_read_level(const Texture *handle, uint x, uint y, uint level) noexcept {
    auto texture = reinterpret_cast<const FallbackTexture *>(handle);
    auto levels = texture->mip_levels();
    LUISA_ASSUME(levels > 0);
    if (level >= levels) { return {}; }
    auto view = texture->view(level);
    auto packed = reinterpret_cast<const PackedTextureView *>(&view);
    return luisa_fallback_texture2d_read_float(packed->data, packed->extra, x, y);
}

[[nodiscard]] float4 luisa_fallback_bindless_texture2d_read(const Texture *handle, uint x, uint y) noexcept {
    return luisa_fallback_bindless_texture2d_read_level(handle, x, y, 0u);
}

[[nodiscard]] float4 luisa_fallback_bindless_texture3d_read_level(const Texture *handle, uint x, uint y, uint z, uint level) noexcept {
    auto texture = reinterpret_cast<const FallbackTexture *>(handle);
    auto levels = texture->mip_levels();
    LUISA_ASSUME(levels > 0);
    if (level >= levels) { return {}; }
    auto view = texture->view(level);
    auto packed = reinterpret_cast<const PackedTextureView *>(&view);
    return luisa_fallback_texture3d_read_float(packed->data, packed->extra, x, y, z);
}

[[nodiscard]] float4 luisa_fallback_bindless_texture3d_read(const Texture *handle, uint x, uint y, uint z) noexcept {
    return luisa_fallback_bindless_texture3d_read_level(handle, x, y, z, 0u);
}

struct alignas(16) RayQueryObject {
    AccelView accel;
    RayQueryCandidate candidate;
    RTCRayHit ray_hit;
    Ray world_ray;
    const RTCRay *active_rtc_ray;
};

struct alignas(16) RayQueryContext {
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    RTCIntersectContext rtc_ctx;
#else
    RTCRayQueryContext rtc_ctx;
#endif
    RayQueryObject *q;
};

struct RayQueryContextEx {
    RayQueryContext base;
    const void *capture;
    const uint8_t *instance_flags;
    float current_t;
    float original_tnear;
    bool object_ray_dirty;
    bool terminate_on_surface_commit;
    uint excl_inst0;
    uint excl_prim0;
    uint excl_inst1;
    uint excl_prim1;
    RayQueryOnSurfaceFunc *on_surface;
    RayQueryOnProceduralFunc *on_procedural;
};

[[nodiscard]] static inline uint resolve_top_level_hit_inst(const RTCHit *hit) noexcept {
    return hit->instID[0] != RTC_INVALID_GEOMETRY_ID ? hit->instID[0] : hit->geomID;
}

// possibly a curve hit, we should modify hit.v to -1
static void ray_trace_fix_hit_for_curves(RTCScene scene, RTCRayHit *rh) noexcept {
    const auto inst = resolve_top_level_hit_inst(&rh->hit);
    rh->hit.instID[0] = inst;
    if (inst != RTC_INVALID_GEOMETRY_ID && rh->hit.v == 0.f) {
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
        auto flags = reinterpret_cast<uint64_t>(rtcGetGeometryUserData(rtcGetGeometry(scene, inst)));
#else
        auto flags = reinterpret_cast<uint64_t>(rtcGetGeometryUserDataFromScene(scene, inst));
#endif
        if (flags & luisa_fallback_embree_accel_user_data_flags_curve) {
            rh->hit.v = -1.f;
        }
    }
}

void luisa_fallback_accel_trace_closest(void *handle, EmbreeRayHit *ray_hit) noexcept {
    const uint64_t t0 = detail_api_profile::enabled() ? detail_api_profile::read_tsc() : 0ull;
    auto scene = static_cast<RTCScene>(handle);
    auto rh = reinterpret_cast<RTCRayHit *>(ray_hit);

    RayQueryContext ctx;
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    rtcInitIntersectContext(&ctx.rtc_ctx);
#else
    rtcInitRayQueryContext(&ctx.rtc_ctx);
#endif
    ctx.q = nullptr;

#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    rtcIntersect1(scene, &ctx.rtc_ctx, rh);
#else
    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    args.context = &ctx.rtc_ctx;
    rtcIntersect1(scene, rh, &args);
#endif
    ray_trace_fix_hit_for_curves(scene, rh);
    if (detail_api_profile::enabled()) [[unlikely]] {
        auto &c = detail_api_profile::tls_slot.c;
        c.trace_closest_calls += 1;
        c.trace_closest_tsc += detail_api_profile::read_tsc() - t0;
    }
}

void luisa_fallback_accel_trace_any(void *handle, EmbreeRay *ray) noexcept {
    const uint64_t t0 = detail_api_profile::enabled() ? detail_api_profile::read_tsc() : 0ull;
    auto scene = static_cast<RTCScene>(handle);
    auto r = reinterpret_cast<RTCRay *>(ray);

    RayQueryContext ctx;
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    rtcInitIntersectContext(&ctx.rtc_ctx);
#else
    rtcInitRayQueryContext(&ctx.rtc_ctx);
#endif
    ctx.q = nullptr;

#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    rtcOccluded1(scene, &ctx.rtc_ctx, r);
#else
    RTCOccludedArguments args;
    rtcInitOccludedArguments(&args);
    args.context = &ctx.rtc_ctx;
    rtcOccluded1(scene, r, &args);
#endif
    if (detail_api_profile::enabled()) [[unlikely]] {
        auto &c = detail_api_profile::tls_slot.c;
        c.trace_any_calls += 1;
        c.trace_any_tsc += detail_api_profile::read_tsc() - t0;
    }
}

size_t luisa_fallback_ray_query_object_size() noexcept {
    return sizeof(RayQueryObject);
}

size_t luisa_fallback_ray_query_object_alignment() noexcept {
    return alignof(RayQueryObject);
}

static void ray_query_update_current_t(RayQueryContextEx *ctx, float new_t) noexcept {
    ctx->current_t = std::min(ctx->current_t, new_t);
    ctx->base.q->world_ray.t_max = ctx->current_t;
}

// Luisa ray queries use a closed lower bound, as do the hardware backends.
// Embree's triangle intersector may reject an exactly reconstructed tnear
// hit before its filter callback runs. Widen only the internal traversal
// interval by one representable value, then reject candidates below the
// original bound in the filter. The smallest-normal fallback keeps the
// widening effective when FTZ/DAZ would erase a subnormal nextafter result.
[[nodiscard]] static float ray_query_embree_tnear(float tnear) noexcept {
    constexpr uint32_t exp_mask = 0x7f800000u;
    const uint32_t bits = std::bit_cast<uint32_t>(tnear);
    if ((bits & exp_mask) == exp_mask) [[unlikely]] { return tnear; }
    if (tnear > std::numeric_limits<float>::min()) [[likely]] {
        return std::bit_cast<float>(bits - 1u);
    }
    if (tnear > -std::numeric_limits<float>::min()) [[likely]] {
        return -std::numeric_limits<float>::min();
    }
    return std::bit_cast<float>(bits + 1u);
}

[[nodiscard]] static bool ray_query_candidate_in_range(
    const RayQueryContextEx *ctx, float t) noexcept {
    return t >= ctx->original_tnear;
}

static void ray_query_decode_surface_candidate(
    RayQueryCandidate *out, const RTCRay *ray, const RTCHit *hit, uint resolved_inst) noexcept {
    out->inst = resolved_inst;
    out->prim = hit->primID;
    out->bary = {hit->u, hit->v};
    out->t = ray->tfar;
    out->committed = false;
    out->terminated = false;
}

static void ray_query_decode_procedural_candidate(RayQueryCandidate *out, const RayQueryContext *ctx, const RTCRay *ray, uint prim) noexcept {
    out->inst = ctx->rtc_ctx.instID[0];
    out->prim = prim;
    out->committed = false;
    out->terminated = false;
}

[[nodiscard]] static inline uint64_t ray_query_surface_instance_flags(
    const RayQueryContextEx *ctx, uint resolved_inst) noexcept {
    LUISA_DEBUG_ASSERT(
        resolved_inst != RTC_INVALID_GEOMETRY_ID,
        "Ray query surface hit is missing its top-level instance/geometry ID.");
    if (ctx->instance_flags != nullptr) [[likely]] {
        return ctx->instance_flags[resolved_inst];
    }
    const auto *instances = *ctx->base.q->accel.instances;
    const auto &desc = instances[resolved_inst];
    return (desc.opaque ? luisa_fallback_embree_accel_user_data_flags_opaque : 0u) |
           (desc.is_curve ? luisa_fallback_embree_accel_user_data_flags_curve : 0u);
}

static inline void ray_query_set_active_rtc_ray(
    RayQueryObject *q, const RTCRay *ray) noexcept {
    q->active_rtc_ray = ray;
}

void luisa_fallback_ray_query_terminate_ray(RTCRay *ray) noexcept {
    ray->tfar = -std::numeric_limits<float>::infinity();
}

void luisa_fallback_ray_query_procedural_intersect_function(const RayQueryIntersectFunctionArguments *args_in) noexcept {
    auto args = reinterpret_cast<const RTCIntersectFunctionNArguments *>(args_in);
    LUISA_DEBUG_ASSERT(args->N == 1u, "Only single ray is support.");
    LUISA_DEBUG_ASSERT(args->valid[0] == -1, "Only valid ray is support.");
    args->valid[0] = 0;
    auto ctx = reinterpret_cast<RayQueryContextEx *>(args->context);
    if (auto q = ctx->base.q) {
        if (auto on_procedural = ctx->on_procedural) {
            auto candidate = &q->candidate;
            auto ray_hit = reinterpret_cast<RTCRayHit *>(args->rayhit);
            ray_query_decode_procedural_candidate(candidate, &ctx->base, &ray_hit->ray, args->primID);
            ray_query_set_active_rtc_ray(q, &ray_hit->ray);
            const auto embree_tnear = ray_hit->ray.tnear;
            ray_hit->ray.tnear = ctx->original_tnear;
            on_procedural(reinterpret_cast<LC_RayQueryObject *>(q), ctx->capture);
            ray_hit->ray.tnear = embree_tnear;
            if (candidate->committed &&
                ray_query_candidate_in_range(ctx, candidate->t) &&
                candidate->t <= ray_hit->ray.tfar) {
                args->valid[0] = -1;
                ray_hit->ray.tfar = candidate->t;
                ray_query_update_current_t(ctx, candidate->t);
                candidate->committed_hit_type = HitType::Procedural;
                ray_hit->hit = {
                    .Ng_x = 0.f,
                    .Ng_y = 0.f,
                    .Ng_z = 0.f,
                    .u = -1.f,
                    .v = -1.f,
                    .primID = args->primID,
                    .geomID = args->geomID,
                    .instID = {candidate->inst},
                };
            }
            if (candidate->terminated) {
                luisa_fallback_ray_query_terminate_ray(&ray_hit->ray);
            }
        }
    }
}

void luisa_fallback_ray_query_procedural_occluded_function(const RayQueryOccludedFunctionArguments *args_in) noexcept {
    auto args = reinterpret_cast<const RTCOccludedFunctionNArguments *>(args_in);
    LUISA_DEBUG_ASSERT(args->N == 1u, "Only single ray is support.");
    LUISA_DEBUG_ASSERT(args->valid[0] == -1, "Only valid ray is support.");
    args->valid[0] = 0;
    auto ctx = reinterpret_cast<RayQueryContextEx *>(args->context);
    if (auto q = ctx->base.q) {
        if (auto on_procedural = ctx->on_procedural) {
            auto candidate = &q->candidate;
            auto ray = reinterpret_cast<RTCRay *>(args->ray);
            ray_query_decode_procedural_candidate(candidate, &ctx->base, ray, args->primID);
            ray_query_set_active_rtc_ray(q, ray);
            const auto embree_tnear = ray->tnear;
            ray->tnear = ctx->original_tnear;
            on_procedural(reinterpret_cast<LC_RayQueryObject *>(q), ctx->capture);
            ray->tnear = embree_tnear;
            if (candidate->committed &&
                ray_query_candidate_in_range(ctx, candidate->t) &&
                candidate->t <= ray->tfar) {
                args->valid[0] = -1;
                ray->tfar = candidate->t;
                candidate->committed_hit_type = HitType::Procedural;
                q->ray_hit.hit = {
                    .Ng_z = candidate->t,
                    .u = -1.f,
                    .v = -1.f,
                    .primID = candidate->prim,
                    .instID = {candidate->inst},
                };
            }
            if (candidate->terminated) {
                luisa_fallback_ray_query_terminate_ray(ray);
            }
        }
    }
}

static void luisa_fallback_ray_query_surface_intersect_filter_function(const RTCFilterFunctionNArguments *args) noexcept {
    LUISA_DEBUG_ASSERT(args->N == 1u, "Only single ray is support.");
    LUISA_DEBUG_ASSERT(args->valid[0] == -1, "Only valid ray is support.");
    auto ctx = reinterpret_cast<RayQueryContextEx *>(args->context);
    auto ray = reinterpret_cast<RTCRay *>(args->ray);
    auto hit = reinterpret_cast<RTCHit *>(args->hit);
    const auto resolved_inst = resolve_top_level_hit_inst(hit);
    auto flags = ray_query_surface_instance_flags(ctx, resolved_inst);
    if (flags & luisa_fallback_embree_accel_user_data_flags_curve) {// curve
        hit->v = -1.f;
    }
    if (!ray_query_candidate_in_range(ctx, ray->tfar)) {
        args->valid[0] = 0;
        return;
    }
    if (flags & luisa_fallback_embree_accel_user_data_flags_opaque) {// opaque, always commit
        ray_query_update_current_t(ctx, ray->tfar);
        ctx->base.q->candidate.committed_hit_type = HitType::Surface;
    } else if (auto on_surface = ctx->on_surface) {
        auto q = ctx->base.q;
        auto candidate = &q->candidate;
        ray_query_decode_surface_candidate(candidate, ray, hit, resolved_inst);
        const uint64_t cb_t0 = detail_api_profile::enabled() ? detail_api_profile::read_tsc() : 0ull;
        on_surface(reinterpret_cast<LC_RayQueryObject *>(q), ctx->capture);
        if (detail_api_profile::enabled()) [[unlikely]] {
            auto &c = detail_api_profile::tls_slot.c;
            c.rq_all_callback_calls += 1;
            c.rq_all_callback_tsc += detail_api_profile::read_tsc() - cb_t0;
        }
        if (candidate->committed) {
            ray_query_update_current_t(ctx, ray->tfar);
            candidate->committed_hit_type = HitType::Surface;
        } else {
            args->valid[0] = 0;
        }
        if (candidate->terminated) {
            luisa_fallback_ray_query_terminate_ray(ray);
        }
    } else {
        args->valid[0] = 0;
    }
}

static void luisa_fallback_ray_query_surface_occluded_filter_function(const RTCFilterFunctionNArguments *args) noexcept {
    LUISA_DEBUG_ASSERT(args->N == 1u, "Only single ray is support.");
    LUISA_DEBUG_ASSERT(args->valid[0] == -1, "Only valid ray is support.");
    static constexpr auto record_hit_data = [](RayQueryObject *q, const RTCRay *ray, const RTCHit *hit, uint resolved_inst) noexcept {
        q->candidate.committed_hit_type = HitType::Surface;
        q->ray_hit.hit.Ng_x = 0.f;
        q->ray_hit.hit.Ng_y = 0.f;
        q->ray_hit.hit.Ng_z = ray->tfar;
        q->ray_hit.hit.u = hit->u;
        q->ray_hit.hit.v = hit->v;
        q->ray_hit.hit.primID = hit->primID;
        q->ray_hit.hit.geomID = hit->geomID;
        q->ray_hit.hit.instID[0] = resolved_inst;
    };
    auto ctx = reinterpret_cast<RayQueryContextEx *>(args->context);
    auto ray = reinterpret_cast<RTCRay *>(args->ray);
    auto hit = reinterpret_cast<RTCHit *>(args->hit);
    const auto resolved_inst = resolve_top_level_hit_inst(hit);
    auto flags = ray_query_surface_instance_flags(ctx, resolved_inst);
    if (flags & luisa_fallback_embree_accel_user_data_flags_curve) {// curve
        hit->v = -1.f;
    }
    if (!ray_query_candidate_in_range(ctx, ray->tfar)) {
        args->valid[0] = 0;
        return;
    }
    auto q = ctx->base.q;
    if (flags & luisa_fallback_embree_accel_user_data_flags_opaque) {// opaque
        record_hit_data(q, ray, hit, resolved_inst);
    } else if (auto on_surface = ctx->on_surface) {
        auto candidate = &q->candidate;
        ray_query_decode_surface_candidate(candidate, ray, hit, resolved_inst);
        uint64_t cb_t0 = 0ull;
        if (detail_api_profile::enabled()) [[unlikely]] {
            detail_api_profile::tls_slot.in_shadow_cb += 1u;
            cb_t0 = detail_api_profile::read_tsc();
        }
        on_surface(reinterpret_cast<LC_RayQueryObject *>(q), ctx->capture);
        if (detail_api_profile::enabled()) [[unlikely]] {
            auto &slot = detail_api_profile::tls_slot;
            slot.c.rq_any_callback_calls += 1;
            slot.c.rq_any_callback_tsc += detail_api_profile::read_tsc() - cb_t0;
            slot.in_shadow_cb -= 1u;
        }
        if (candidate->committed) {
            record_hit_data(q, ray, hit, resolved_inst);
        } else {
            args->valid[0] = 0;
        }
        if (candidate->terminated) {
            luisa_fallback_ray_query_terminate_ray(ray);
        }
    } else {
        args->valid[0] = 0;
    }
}

[[nodiscard]] static inline const uint8_t *ray_query_resolve_instance_flags(const AccelView &accel) noexcept {
    // FallbackAccelRuntimeHeader stores {Instance *instances, const uint8_t *instance_flags}
    // contiguously and passes &header.instances as AccelView::instances.
    return accel.instances != nullptr
               ? reinterpret_cast<const uint8_t *const *>(accel.instances)[1]
               : nullptr;
}

void luisa_fallback_ray_query_pipeline_all(LC_RayQueryObject *query_object, const void *capture, RayQueryOnSurfaceFunc *on_surface, RayQueryOnProceduralFunc *on_procedural) noexcept {
    const uint64_t rq_t0 = detail_api_profile::enabled() ? detail_api_profile::read_tsc() : 0ull;
    auto q = reinterpret_cast<RayQueryObject *>(query_object);
    auto scene = static_cast<RTCScene>(q->accel.embree_scene);

    RayQueryContextEx ctx;
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    rtcInitIntersectContext(&ctx.base.rtc_ctx);
#else
    rtcInitRayQueryContext(&ctx.base.rtc_ctx);
#endif
    ctx.base.q = q;
    ctx.capture = capture;
    ctx.instance_flags = ray_query_resolve_instance_flags(q->accel);
    ctx.current_t = q->world_ray.t_max;
    ctx.original_tnear = q->ray_hit.ray.tnear;
    ctx.object_ray_dirty = false;
    ctx.terminate_on_surface_commit = false;
    ctx.excl_inst0 = RTC_INVALID_GEOMETRY_ID;
    ctx.excl_prim0 = RTC_INVALID_GEOMETRY_ID;
    ctx.excl_inst1 = RTC_INVALID_GEOMETRY_ID;
    ctx.excl_prim1 = RTC_INVALID_GEOMETRY_ID;
    ctx.on_surface = on_surface;
    ctx.on_procedural = on_procedural;
    q->ray_hit.ray.tnear =
        ray_query_embree_tnear(ctx.original_tnear);

#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    ctx.base.rtc_ctx.filter = luisa_fallback_ray_query_surface_intersect_filter_function;
    rtcIntersect1(scene, &ctx.base.rtc_ctx, &q->ray_hit);
#else
    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    args.context = &ctx.base.rtc_ctx;
    args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
    args.filter = luisa_fallback_ray_query_surface_intersect_filter_function;
    rtcIntersect1(scene, &q->ray_hit, &args);
#endif
    q->active_rtc_ray = nullptr;
    q->ray_hit.ray.tnear = ctx.original_tnear;
    q->ray_hit.hit.Ng_z = ctx.current_t;
    const auto committed_inst = resolve_top_level_hit_inst(&q->ray_hit.hit);
    q->ray_hit.hit.instID[0] = committed_inst;
    if (committed_inst != RTC_INVALID_GEOMETRY_ID && q->ray_hit.hit.v == 0.f) {
        const bool is_curve = ctx.instance_flags != nullptr
            ? ((ctx.instance_flags[committed_inst] & luisa_fallback_embree_accel_user_data_flags_curve) != 0u)
            : static_cast<bool>((*q->accel.instances)[committed_inst].is_curve);
        if (is_curve) {
            q->ray_hit.hit.v = -1.f;
        }
    }
    if (detail_api_profile::enabled()) [[unlikely]] {
        auto &c = detail_api_profile::tls_slot.c;
        c.rq_all_calls += 1;
        c.rq_all_total_tsc += detail_api_profile::read_tsc() - rq_t0;
    }
}

void luisa_fallback_ray_query_pipeline_any(LC_RayQueryObject *query_object, const void *capture, RayQueryOnSurfaceFunc *on_surface, RayQueryOnProceduralFunc *on_procedural) noexcept {
    const uint64_t rq_t0 = detail_api_profile::enabled() ? detail_api_profile::read_tsc() : 0ull;
    auto q = reinterpret_cast<RayQueryObject *>(query_object);
    auto scene = static_cast<RTCScene>(q->accel.embree_scene);

    RayQueryContextEx ctx;
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    rtcInitIntersectContext(&ctx.base.rtc_ctx);
#else
    rtcInitRayQueryContext(&ctx.base.rtc_ctx);
#endif
    ctx.base.q = q;
    ctx.capture = capture;
    ctx.instance_flags = ray_query_resolve_instance_flags(q->accel);
    ctx.current_t = q->world_ray.t_max;
    ctx.original_tnear = q->ray_hit.ray.tnear;
    ctx.object_ray_dirty = false;
    ctx.terminate_on_surface_commit = false;
    ctx.excl_inst0 = RTC_INVALID_GEOMETRY_ID;
    ctx.excl_prim0 = RTC_INVALID_GEOMETRY_ID;
    ctx.excl_inst1 = RTC_INVALID_GEOMETRY_ID;
    ctx.excl_prim1 = RTC_INVALID_GEOMETRY_ID;
    ctx.on_surface = on_surface;
    ctx.on_procedural = on_procedural;
    q->ray_hit.ray.tnear =
        ray_query_embree_tnear(ctx.original_tnear);

#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    ctx.base.rtc_ctx.filter = luisa_fallback_ray_query_surface_occluded_filter_function;
    rtcOccluded1(scene, &ctx.base.rtc_ctx, &q->ray_hit.ray);
#else
    RTCOccludedArguments args;
    rtcInitOccludedArguments(&args);
    args.context = &ctx.base.rtc_ctx;
    args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
    args.filter = luisa_fallback_ray_query_surface_occluded_filter_function;
    rtcOccluded1(scene, &q->ray_hit.ray, &args);
#endif
    q->active_rtc_ray = nullptr;
    q->ray_hit.ray.tnear = ctx.original_tnear;
    if (detail_api_profile::enabled()) [[unlikely]] {
        auto &c = detail_api_profile::tls_slot.c;
        c.rq_any_calls += 1;
        c.rq_any_total_tsc += detail_api_profile::read_tsc() - rq_t0;
    }
}

static void luisa_fallback_ray_query_surface_intersect_excl_filter_function(const RTCFilterFunctionNArguments *args) noexcept {
    LUISA_DEBUG_ASSERT(args->N == 1u, "Only single ray is support.");
    LUISA_DEBUG_ASSERT(args->valid[0] == -1, "Only valid ray is support.");
    auto ctx = reinterpret_cast<RayQueryContextEx *>(args->context);
    auto ray = reinterpret_cast<RTCRay *>(args->ray);
    auto hit = reinterpret_cast<RTCHit *>(args->hit);
    const auto resolved_inst = resolve_top_level_hit_inst(hit);
    const auto flags = ray_query_surface_instance_flags(ctx, resolved_inst);
    if (flags & luisa_fallback_embree_accel_user_data_flags_curve) [[unlikely]] {
        hit->v = -1.f;
    }
    if (!ray_query_candidate_in_range(ctx, ray->tfar)) [[unlikely]] {
        args->valid[0] = 0;
        return;
    }
    if ((flags & luisa_fallback_embree_accel_user_data_flags_opaque) == 0u) {
        const uint prim = hit->primID;
        if ((resolved_inst == ctx->excl_inst0 && prim == ctx->excl_prim0) ||
            (resolved_inst == ctx->excl_inst1 && prim == ctx->excl_prim1)) {
            args->valid[0] = 0;
            return;
        }
    }
    ray_query_update_current_t(ctx, ray->tfar);
    ctx->base.q->candidate.committed_hit_type = HitType::Surface;
    if ((flags & luisa_fallback_embree_accel_user_data_flags_opaque) == 0u &&
        ctx->terminate_on_surface_commit) [[unlikely]] {
        luisa_fallback_ray_query_terminate_ray(ray);
    }
}

static void luisa_fallback_ray_query_surface_occluded_excl_filter_function(const RTCFilterFunctionNArguments *args) noexcept {
    LUISA_DEBUG_ASSERT(args->N == 1u, "Only single ray is support.");
    LUISA_DEBUG_ASSERT(args->valid[0] == -1, "Only valid ray is support.");
    auto ctx = reinterpret_cast<RayQueryContextEx *>(args->context);
    auto ray = reinterpret_cast<RTCRay *>(args->ray);
    auto hit = reinterpret_cast<RTCHit *>(args->hit);
    const auto resolved_inst = resolve_top_level_hit_inst(hit);
    const auto flags = ray_query_surface_instance_flags(ctx, resolved_inst);
    if (flags & luisa_fallback_embree_accel_user_data_flags_curve) [[unlikely]] {
        hit->v = -1.f;
    }
    if (!ray_query_candidate_in_range(ctx, ray->tfar)) [[unlikely]] {
        args->valid[0] = 0;
        return;
    }
    auto q = ctx->base.q;
    if ((flags & luisa_fallback_embree_accel_user_data_flags_opaque) == 0u) {
        const uint prim = hit->primID;
        if ((resolved_inst == ctx->excl_inst0 && prim == ctx->excl_prim0) ||
            (resolved_inst == ctx->excl_inst1 && prim == ctx->excl_prim1)) {
            args->valid[0] = 0;
            return;
        }
        q->world_ray.t_max = ray->tfar;
    }
    q->candidate.committed_hit_type = HitType::Surface;
    q->ray_hit.hit.Ng_x = 0.f;
    q->ray_hit.hit.Ng_y = 0.f;
    q->ray_hit.hit.Ng_z = ray->tfar;
    q->ray_hit.hit.u = hit->u;
    q->ray_hit.hit.v = hit->v;
    q->ray_hit.hit.primID = hit->primID;
    q->ray_hit.hit.geomID = hit->geomID;
    q->ray_hit.hit.instID[0] = resolved_inst;
    if ((flags & luisa_fallback_embree_accel_user_data_flags_opaque) == 0u &&
        ctx->terminate_on_surface_commit) {
        luisa_fallback_ray_query_terminate_ray(ray);
    }
}

void luisa_fallback_ray_query_pipeline_all_excl(
    LC_RayQueryObject *query_object,
    uint excl_inst0, uint excl_prim0,
    uint excl_inst1, uint excl_prim1,
    uint flags,
    const void *capture,
    RayQueryOnProceduralFunc *on_procedural) noexcept {
    const uint64_t rq_t0 = detail_api_profile::enabled() ? detail_api_profile::read_tsc() : 0ull;
    auto q = reinterpret_cast<RayQueryObject *>(query_object);
    auto scene = static_cast<RTCScene>(q->accel.embree_scene);

    RayQueryContextEx ctx;
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    rtcInitIntersectContext(&ctx.base.rtc_ctx);
#else
    rtcInitRayQueryContext(&ctx.base.rtc_ctx);
#endif
    ctx.base.q = q;
    ctx.capture = capture;
    ctx.instance_flags = ray_query_resolve_instance_flags(q->accel);
    ctx.current_t = q->world_ray.t_max;
    ctx.original_tnear = q->ray_hit.ray.tnear;
    ctx.object_ray_dirty = false;
    ctx.terminate_on_surface_commit = (flags & 1u) != 0u;
    ctx.excl_inst0 = excl_inst0;
    ctx.excl_prim0 = excl_prim0;
    ctx.excl_inst1 = excl_inst1;
    ctx.excl_prim1 = excl_prim1;
    ctx.on_surface = nullptr;
    ctx.on_procedural = on_procedural;
    q->ray_hit.ray.tnear = ray_query_embree_tnear(ctx.original_tnear);

#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    ctx.base.rtc_ctx.filter = luisa_fallback_ray_query_surface_intersect_excl_filter_function;
    rtcIntersect1(scene, &ctx.base.rtc_ctx, &q->ray_hit);
#else
    const bool no_filter_fast_path =
        on_procedural == nullptr &&
        !ctx.terminate_on_surface_commit &&
        excl_inst0 == RTC_INVALID_GEOMETRY_ID &&
        excl_inst1 == RTC_INVALID_GEOMETRY_ID &&
        ctx.original_tnear == 0.f;
    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    args.context = &ctx.base.rtc_ctx;
    if (no_filter_fast_path) {
        q->ray_hit.ray.tnear = 0.f;
        rtcIntersect1(scene, &q->ray_hit, &args);
        const auto committed_inst = resolve_top_level_hit_inst(&q->ray_hit.hit);
        q->ray_hit.hit.instID[0] = committed_inst;
        if (committed_inst != RTC_INVALID_GEOMETRY_ID) {
            ctx.current_t = q->ray_hit.ray.tfar;
            q->world_ray.t_max = ctx.current_t;
            q->candidate.committed_hit_type = HitType::Surface;
            if (q->ray_hit.hit.v == 0.f) {
                const bool is_curve = ctx.instance_flags != nullptr
                    ? ((ctx.instance_flags[committed_inst] & luisa_fallback_embree_accel_user_data_flags_curve) != 0u)
                    : static_cast<bool>((*q->accel.instances)[committed_inst].is_curve);
                if (is_curve) {
                    q->ray_hit.hit.v = -1.f;
                }
            }
        }
        q->active_rtc_ray = nullptr;
        q->ray_hit.hit.Ng_z = ctx.current_t;
        if (detail_api_profile::enabled()) [[unlikely]] {
            auto &c = detail_api_profile::tls_slot.c;
            c.rq_all_calls += 1;
            c.rq_all_total_tsc += detail_api_profile::read_tsc() - rq_t0;
        }
        return;
    }
    args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
    args.filter = luisa_fallback_ray_query_surface_intersect_excl_filter_function;
    rtcIntersect1(scene, &q->ray_hit, &args);
#endif
    q->active_rtc_ray = nullptr;
    q->ray_hit.ray.tnear = ctx.original_tnear;
    q->ray_hit.hit.Ng_z = ctx.current_t;
    const auto committed_inst = resolve_top_level_hit_inst(&q->ray_hit.hit);
    q->ray_hit.hit.instID[0] = committed_inst;
    if (committed_inst != RTC_INVALID_GEOMETRY_ID && q->ray_hit.hit.v == 0.f) {
        const bool is_curve = ctx.instance_flags != nullptr
            ? ((ctx.instance_flags[committed_inst] & luisa_fallback_embree_accel_user_data_flags_curve) != 0u)
            : static_cast<bool>((*q->accel.instances)[committed_inst].is_curve);
        if (is_curve) {
            q->ray_hit.hit.v = -1.f;
        }
    }
    if (detail_api_profile::enabled()) [[unlikely]] {
        auto &c = detail_api_profile::tls_slot.c;
        c.rq_all_calls += 1;
        c.rq_all_total_tsc += detail_api_profile::read_tsc() - rq_t0;
    }
}

void luisa_fallback_ray_query_pipeline_any_excl(
    LC_RayQueryObject *query_object,
    uint excl_inst0, uint excl_prim0,
    uint excl_inst1, uint excl_prim1,
    uint flags,
    const void *capture,
    RayQueryOnProceduralFunc *on_procedural) noexcept {
    const uint64_t rq_t0 = detail_api_profile::enabled() ? detail_api_profile::read_tsc() : 0ull;
    auto q = reinterpret_cast<RayQueryObject *>(query_object);
    auto scene = static_cast<RTCScene>(q->accel.embree_scene);

    RayQueryContextEx ctx;
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    rtcInitIntersectContext(&ctx.base.rtc_ctx);
#else
    rtcInitRayQueryContext(&ctx.base.rtc_ctx);
#endif
    ctx.base.q = q;
    ctx.capture = capture;
    ctx.instance_flags = ray_query_resolve_instance_flags(q->accel);
    ctx.current_t = q->world_ray.t_max;
    ctx.original_tnear = q->ray_hit.ray.tnear;
    ctx.object_ray_dirty = false;
    ctx.terminate_on_surface_commit = (flags & 1u) != 0u;
    ctx.excl_inst0 = excl_inst0;
    ctx.excl_prim0 = excl_prim0;
    ctx.excl_inst1 = excl_inst1;
    ctx.excl_prim1 = excl_prim1;
    ctx.on_surface = nullptr;
    ctx.on_procedural = on_procedural;
    q->ray_hit.ray.tnear = ray_query_embree_tnear(ctx.original_tnear);

#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    ctx.base.rtc_ctx.filter = luisa_fallback_ray_query_surface_occluded_excl_filter_function;
    rtcOccluded1(scene, &ctx.base.rtc_ctx, &q->ray_hit.ray);
#else
    RTCOccludedArguments args;
    rtcInitOccludedArguments(&args);
    args.context = &ctx.base.rtc_ctx;
    args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
    args.filter = luisa_fallback_ray_query_surface_occluded_excl_filter_function;
    rtcOccluded1(scene, &q->ray_hit.ray, &args);
#endif
    q->active_rtc_ray = nullptr;
    q->ray_hit.ray.tnear = ctx.original_tnear;
    if (detail_api_profile::enabled()) [[unlikely]] {
        auto &c = detail_api_profile::tls_slot.c;
        c.rq_any_calls += 1;
        c.rq_any_total_tsc += detail_api_profile::read_tsc() - rq_t0;
    }
}

}// namespace luisa::compute::fallback::api
