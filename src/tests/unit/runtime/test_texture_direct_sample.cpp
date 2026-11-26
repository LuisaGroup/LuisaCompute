// Deterministic CUDA/GPU direct texture sampling test.
//
// Verifies hardware TMU texture direct sampling across 2D and 3D textures:
//   - sample / sample_level / sample_grad / sample_grad_level
//   - all 4 filters x 4 address modes (compile-time static & runtime dynamic)
//   - mipmap level selection & trilinear interpolation
//   - mixed read/write + sample on the same texture
//   - sample-only textures (including BC1, BYTE4_SRGB, BC7_SRGB)
//   - non-zero mip views (image.view(level), volume.view(level))
//   - Callable parameter narrowing & dynamic fallback
//   - shader cache save/reload preservation of texture sampler masks
//   - zero CUtexObject creation and zero sampler-table allocation guarantees on CUDA

#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <luisa/luisa-compute.h>

#ifdef LUISA_TEST_CUDA_WHITEBOX
#include "cuda_texture.h"
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

template<typename TexExpr, typename... Args>
[[nodiscard]] Var<float4> call_direct_sample(CallOp op, TexExpr tex, Args &&...args) noexcept {
    auto f = luisa::compute::detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), op,
        {tex.expression(), luisa::compute::detail::extract_expression(std::forward<Args>(args))...}));
}

[[nodiscard]] Var<float4> tex2d_sample_static(
    Expr<Image<float>> img, Expr<float2> uv,
    SamplerFilter filter, SamplerAddress address) noexcept {
    auto f = luisa::compute::detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE,
        {img.expression(), uv.expression(),
         f->literal(Type::of<uint>(), static_cast<uint>(filter)),
         f->literal(Type::of<uint>(), static_cast<uint>(address))}));
}

[[nodiscard]] Var<float4> tex2d_sample_dynamic(
    Expr<Image<float>> img, Expr<float2> uv,
    Expr<uint> filter, Expr<uint> address) noexcept {
    return call_direct_sample(CallOp::TEXTURE2D_SAMPLE, img, uv, filter, address);
}

[[nodiscard]] Var<float4> tex2d_sample_level_static(
    Expr<Image<float>> img, Expr<float2> uv, Expr<float> level,
    SamplerFilter filter, SamplerAddress address) noexcept {
    auto f = luisa::compute::detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE_LEVEL,
        {img.expression(), uv.expression(), level.expression(),
         f->literal(Type::of<uint>(), static_cast<uint>(filter)),
         f->literal(Type::of<uint>(), static_cast<uint>(address))}));
}

[[nodiscard]] Var<float4> tex2d_sample_level_dynamic(
    Expr<Image<float>> img, Expr<float2> uv, Expr<float> level,
    Expr<uint> filter, Expr<uint> address) noexcept {
    return call_direct_sample(CallOp::TEXTURE2D_SAMPLE_LEVEL, img, uv, level, filter, address);
}

[[nodiscard]] Var<float4> tex2d_sample_grad_static(
    Expr<Image<float>> img, Expr<float2> uv, Expr<float2> ddx, Expr<float2> ddy,
    SamplerFilter filter, SamplerAddress address) noexcept {
    auto f = luisa::compute::detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE_GRAD,
        {img.expression(), uv.expression(), ddx.expression(), ddy.expression(),
         f->literal(Type::of<uint>(), static_cast<uint>(filter)),
         f->literal(Type::of<uint>(), static_cast<uint>(address))}));
}

[[nodiscard]] Var<float4> tex2d_sample_grad_level_static(
    Expr<Image<float>> img, Expr<float2> uv, Expr<float2> ddx, Expr<float2> ddy, Expr<float> mip_clamp,
    SamplerFilter filter, SamplerAddress address) noexcept {
    auto f = luisa::compute::detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL,
        {img.expression(), uv.expression(), ddx.expression(), ddy.expression(), mip_clamp.expression(),
         f->literal(Type::of<uint>(), static_cast<uint>(filter)),
         f->literal(Type::of<uint>(), static_cast<uint>(address))}));
}

[[nodiscard]] Var<float4> tex3d_sample_static(
    Expr<Volume<float>> vol, Expr<float3> uvw,
    SamplerFilter filter, SamplerAddress address) noexcept {
    auto f = luisa::compute::detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE3D_SAMPLE,
        {vol.expression(), uvw.expression(),
         f->literal(Type::of<uint>(), static_cast<uint>(filter)),
         f->literal(Type::of<uint>(), static_cast<uint>(address))}));
}

[[nodiscard]] Var<float4> tex3d_sample_dynamic(
    Expr<Volume<float>> vol, Expr<float3> uvw,
    Expr<uint> filter, Expr<uint> address) noexcept {
    return call_direct_sample(CallOp::TEXTURE3D_SAMPLE, vol, uvw, filter, address);
}

[[nodiscard]] Var<float4> tex3d_sample_level_static(
    Expr<Volume<float>> vol, Expr<float3> uvw, Expr<float> level,
    SamplerFilter filter, SamplerAddress address) noexcept {
    auto f = luisa::compute::detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE3D_SAMPLE_LEVEL,
        {vol.expression(), uvw.expression(), level.expression(),
         f->literal(Type::of<uint>(), static_cast<uint>(filter)),
         f->literal(Type::of<uint>(), static_cast<uint>(address))}));
}

[[nodiscard]] Var<float4> tex3d_sample_grad_static(
    Expr<Volume<float>> vol, Expr<float3> uvw, Expr<float3> ddx, Expr<float3> ddy,
    SamplerFilter filter, SamplerAddress address) noexcept {
    auto f = luisa::compute::detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE3D_SAMPLE_GRAD,
        {vol.expression(), uvw.expression(), ddx.expression(), ddy.expression(),
         f->literal(Type::of<uint>(), static_cast<uint>(filter)),
         f->literal(Type::of<uint>(), static_cast<uint>(address))}));
}

[[nodiscard]] Var<float4> tex3d_sample_grad_level_static(
    Expr<Volume<float>> vol, Expr<float3> uvw, Expr<float3> ddx, Expr<float3> ddy, Expr<float> mip_clamp,
    SamplerFilter filter, SamplerAddress address) noexcept {
    auto f = luisa::compute::detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL,
        {vol.expression(), uvw.expression(), ddx.expression(), ddy.expression(), mip_clamp.expression(),
         f->literal(Type::of<uint>(), static_cast<uint>(filter)),
         f->literal(Type::of<uint>(), static_cast<uint>(address))}));
}

[[nodiscard]] float max_error(float4 a, float4 b) noexcept {
    auto d = abs(a - b);
    return std::max(std::max(d.x, d.y), std::max(d.z, d.w));
}

[[nodiscard]] float srgb_u8_to_linear(uint8_t v) noexcept {
    auto c = static_cast<float>(v) / 255.0f;
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

[[nodiscard]] std::array<uint8_t, 16u> make_bc7_mode6_solid_block(
    uint8_t r7, uint8_t g7, uint8_t b7, uint8_t a7, uint8_t pbit) noexcept {
    // BC7 Mode 6: bit6=1 (bits [0..6] = 0b1000000), 7-bit endpoints R0,R1,G0,G1,B0,B1,A0,A1,
    // followed by 1-bit P0, P1, and all-zero 4-bit texel indices -> every texel decodes to
    // ((R0 << 1) | P0, (G0 << 1) | P0, (B0 << 1) | P0, (A0 << 1) | P0).
    uint64_t lo = 0ull;
    uint64_t hi = 0ull;
    auto push_bits = [&](uint32_t offset, uint32_t width, uint32_t value) noexcept {
        for (uint32_t i = 0u; i < width; ++i) {
            const auto bit = static_cast<uint64_t>((value >> i) & 1u);
            const auto pos = offset + i;
            if (pos < 64u) {
                lo |= bit << pos;
            } else {
                hi |= bit << (pos - 64u);
            }
        }
    };
    push_bits(0u, 7u, 0x40u);
    push_bits(7u, 7u, r7);
    push_bits(14u, 7u, r7);
    push_bits(21u, 7u, g7);
    push_bits(28u, 7u, g7);
    push_bits(35u, 7u, b7);
    push_bits(42u, 7u, b7);
    push_bits(49u, 7u, a7);
    push_bits(56u, 7u, a7);
    push_bits(63u, 1u, pbit);
    push_bits(64u, 1u, pbit);
    std::array<uint8_t, 16u> bytes{};
    std::memcpy(bytes.data(), &lo, 8u);
    std::memcpy(bytes.data() + 8u, &hi, 8u);
    return bytes;
}

[[nodiscard]] int positive_mod(int x, int m) noexcept {
    auto r = x % m;
    return r < 0 ? r + m : r;
}

[[nodiscard]] float4 host_texel_4x4(uint32_t x, uint32_t y) noexcept {
    return make_float4(
        static_cast<float>(x + 1u) * (1.0f / 8.0f),
        static_cast<float>(y + 1u) * (1.0f / 8.0f),
        static_cast<float>(x + 4u * y) * (1.0f / 32.0f),
        1.0f);
}

[[nodiscard]] float4 host_fetch_4x4(int x, int y, Sampler::Address addr) noexcept {
    constexpr int size = 4;
    auto resolve = [addr](int v, bool &ok) noexcept {
        switch (addr) {
            case Sampler::Address::EDGE: return std::clamp(v, 0, size - 1);
            case Sampler::Address::REPEAT: return positive_mod(v, size);
            case Sampler::Address::MIRROR: {
                auto p = positive_mod(v, 2 * size);
                return p < size ? p : 2 * size - 1 - p;
            }
            case Sampler::Address::ZERO:
                if (v < 0 || v >= size) { ok = false; }
                return std::clamp(v, 0, size - 1);
        }
        return 0;
    };
    bool ok = true;
    x = resolve(x, ok);
    y = resolve(y, ok);
    return ok ? host_texel_4x4(static_cast<uint32_t>(x), static_cast<uint32_t>(y)) : make_float4(0.0f);
}

[[nodiscard]] float4 host_sample_4x4(float2 uv, Sampler::Filter filter, Sampler::Address addr) noexcept {
    constexpr float size = 4.0f;
    if (filter == Sampler::Filter::POINT) {
        auto ix = static_cast<int>(std::floor(uv.x * size));
        auto iy = static_cast<int>(std::floor(uv.y * size));
        return host_fetch_4x4(ix, iy, addr);
    }
    auto px = uv.x * size - 0.5f;
    auto py = uv.y * size - 0.5f;
    auto x0 = static_cast<int>(std::floor(px));
    auto y0 = static_cast<int>(std::floor(py));
    auto tx = px - static_cast<float>(x0);
    auto ty = py - static_cast<float>(y0);
    auto v00 = host_fetch_4x4(x0, y0, addr);
    auto v10 = host_fetch_4x4(x0 + 1, y0, addr);
    auto v01 = host_fetch_4x4(x0, y0 + 1, addr);
    auto v11 = host_fetch_4x4(x0 + 1, y0 + 1, addr);
    return lerp(lerp(v00, v10, tx), lerp(v01, v11, tx), ty);
}

[[nodiscard]] uint32_t cuda_tex_count(const auto &res) noexcept {
#ifdef LUISA_TEST_CUDA_WHITEBOX
    return reinterpret_cast<const cuda::CUDATexture *>(res.handle())->created_tex_object_count();
#else
    static_cast<void>(res);
    return 0u;
#endif
}

[[nodiscard]] bool cuda_has_table(const auto &res) noexcept {
#ifdef LUISA_TEST_CUDA_WHITEBOX
    return reinterpret_cast<const cuda::CUDATexture *>(res.handle())->has_sampler_table();
#else
    static_cast<void>(res);
    return false;
#endif
}

}// namespace

void test_texture_direct_sample(Device &device) {
    const bool is_cuda = device.backend_name() == "cuda";
    auto stream = device.create_stream();
    constexpr float tol = 1.0e-3f;

    // 1. White-box resource overhead check:
    //    - read/write-only texture creates 0 CUtexObject and no sampler table
    //    - single-static-sampler creates 1 CUtexObject and no sampler table
    //    - multi-static-sampler creates exact N CUtexObjects + sampler table
    //    - dynamic sampler creates 16 CUtexObjects + sampler table
    {
        auto rw_img = device.create_image<float>(PixelStorage::FLOAT4, make_uint2(4u));
        auto single_img = device.create_image<float>(PixelStorage::FLOAT4, make_uint2(4u));
        auto dual_img = device.create_image<float>(PixelStorage::FLOAT4, make_uint2(4u));
        auto dyn_img = device.create_image<float>(PixelStorage::FLOAT4, make_uint2(4u));
        auto mixed_img = device.create_image<float>(PixelStorage::FLOAT4, make_uint2(4u));

        luisa::vector<float4> src(16u, make_float4(0.25f, 0.5f, 0.75f, 1.0f));
        src[5u] = make_float4(0.9f, 0.2f, 0.1f, 1.0f);
        auto out = device.create_buffer<float4>(6u);
        auto rw_out = device.create_buffer<float4>(1u);
        auto mixed_copy = luisa::vector<float4>(16u);
        auto dyn_params = device.create_buffer<uint2>(1u);
        uint2 host_dyn{static_cast<uint>(SamplerFilter::POINT), static_cast<uint>(SamplerAddress::EDGE)};

        Callable sample_in_callable_static = [](ImageFloat tex, Float2 uv) noexcept {
            return tex2d_sample_static(tex, uv, SamplerFilter::LINEAR_POINT, SamplerAddress::REPEAT);
        };
        Callable sample_in_callable_dyn = [](ImageFloat tex, Float2 uv, UInt f, UInt a) noexcept {
            return tex2d_sample_dynamic(tex, uv, f, a);
        };

        Kernel1D k_overhead = [&](ImageFloat rw, ImageFloat single, ImageFloat dual,
                                  ImageFloat dyn, ImageFloat mixed,
                                  BufferUInt2 dparams, BufferFloat4 res, BufferFloat4 rw_res) noexcept {
            auto p = rw.read(make_uint2(1u, 1u));
            rw.write(make_uint2(0u, 0u), p + make_float4(0.1f));
            rw_res.write(0u, rw.read(make_uint2(1u, 1u)));

            res.write(0u, sample_in_callable_static(single, make_float2(0.375f, 0.375f)));
            res.write(1u, tex2d_sample_static(dual, make_float2(0.375f, 0.375f), SamplerFilter::POINT, SamplerAddress::EDGE));
            res.write(2u, tex2d_sample_static(dual, make_float2(0.375f, 0.375f), SamplerFilter::LINEAR_LINEAR, SamplerAddress::MIRROR));

            auto dp = dparams.read(0u);
            res.write(3u, sample_in_callable_dyn(dyn, make_float2(0.375f, 0.375f), dp.x, dp.y));

            auto mv = mixed.read(make_uint2(1u, 1u));
            mixed.write(make_uint2(2u, 2u), mv * 0.5f);
            res.write(4u, tex2d_sample_static(mixed, make_float2(0.375f, 0.375f), SamplerFilter::POINT, SamplerAddress::EDGE));
            res.write(5u, mv);
        };

        std::array<float4, 6u> res_host{};
        float4 rw_host{};
        auto shader = device.compile(k_overhead);
        stream << rw_img.copy_from(src.data())
               << single_img.copy_from(src.data())
               << dual_img.copy_from(src.data())
               << dyn_img.copy_from(src.data())
               << mixed_img.copy_from(src.data())
               << dyn_params.copy_from(&host_dyn)
               << shader(rw_img, single_img, dual_img, dyn_img, mixed_img, dyn_params, out, rw_out).dispatch(1u)
               << out.copy_to(res_host.data())
               << rw_out.copy_to(&rw_host)
               << mixed_img.copy_to(mixed_copy.data())
               << synchronize();

        expect(max_error(rw_host, src[5u]) < tol);
        expect(max_error(res_host[0u], src[5u]) < tol);
        expect(max_error(res_host[1u], src[5u]) < tol);
        expect(max_error(res_host[3u], src[5u]) < tol);
        expect(max_error(res_host[4u], src[5u]) < tol);
        expect(max_error(res_host[5u], src[5u]) < tol);
        expect(max_error(mixed_copy[2u * 4u + 2u], src[5u] * 0.5f) < tol);

        if (is_cuda) {
            expect(eq(cuda_tex_count(rw_img), 0u));
            expect(!cuda_has_table(rw_img));
            expect(eq(cuda_tex_count(single_img), 1u));
            expect(!cuda_has_table(single_img));
            expect(eq(cuda_tex_count(dual_img), 2u));
            expect(cuda_has_table(dual_img));
            expect(eq(cuda_tex_count(dyn_img), 16u));
            expect(cuda_has_table(dyn_img));
            expect(eq(cuda_tex_count(mixed_img), 1u));
            expect(!cuda_has_table(mixed_img));
        }
    }

    // 2. All 4 filters x 4 address modes (16 static + 16 dynamic) for 2D & 3D
    {
        auto tex2d = device.create_image<float>(PixelStorage::FLOAT4, make_uint2(4u));
        luisa::vector<float4> src2(16u);
        for (uint32_t y = 0u; y < 4u; ++y) {
            for (uint32_t x = 0u; x < 4u; ++x) {
                src2[y * 4u + x] = host_texel_4x4(x, y);
            }
        }
        auto vol3d = device.create_volume<float>(PixelStorage::FLOAT4, make_uint3(2u));
        std::array<float4, 8u> src3{};
        for (uint32_t i = 0u; i < 8u; ++i) {
            src3[i] = make_float4(
                static_cast<float>(i + 1u) * 0.1f,
                static_cast<float>(i + 2u) * 0.1f,
                static_cast<float>(i + 3u) * 0.1f,
                1.0f);
        }

        constexpr uint32_t combos = 16u;
        auto out2_static = device.create_buffer<float4>(combos);
        auto out2_dyn = device.create_buffer<float4>(combos);
        auto out3_static = device.create_buffer<float4>(combos);
        auto out3_dyn = device.create_buffer<float4>(combos);

        constexpr auto test_uv = make_float2(-0.40f, 1.40f);
        constexpr auto test_uvw = make_float3(-0.25f, 0.25f, 0.75f);

        Kernel1D k_static_all = [&](ImageFloat i2, VolumeFloat v3,
                                    BufferFloat4 o2, BufferFloat4 o3) noexcept {
            auto run_pair = [&](uint32_t idx, SamplerFilter f, SamplerAddress a) noexcept {
                o2.write(idx, tex2d_sample_static(i2, test_uv, f, a));
                o3.write(idx, tex3d_sample_static(v3, test_uvw, f, a));
            };
            constexpr std::array filters{
                SamplerFilter::POINT, SamplerFilter::LINEAR_POINT,
                SamplerFilter::LINEAR_LINEAR, SamplerFilter::ANISOTROPIC};
            constexpr std::array addrs{
                SamplerAddress::EDGE, SamplerAddress::REPEAT,
                SamplerAddress::MIRROR, SamplerAddress::ZERO};
            for (uint32_t fi = 0u; fi < 4u; ++fi) {
                for (uint32_t ai = 0u; ai < 4u; ++ai) {
                    run_pair(fi * 4u + ai, filters[fi], addrs[ai]);
                }
            }
        };

        Kernel1D k_dyn_all = [&](ImageFloat i2, VolumeFloat v3,
                                 BufferFloat4 o2, BufferFloat4 o3) noexcept {
            auto idx = dispatch_x();
            auto f = idx / 4u;
            auto a = idx % 4u;
            o2.write(idx, tex2d_sample_dynamic(i2, test_uv, f, a));
            o3.write(idx, tex3d_sample_dynamic(v3, test_uvw, f, a));
        };

        auto s_static = device.compile(k_static_all);
        auto s_dyn = device.compile(k_dyn_all);
        std::array<float4, combos> h2s{}, h2d{}, h3s{}, h3d{};
        stream << tex2d.copy_from(src2.data())
               << vol3d.copy_from(src3.data())
               << s_static(tex2d, vol3d, out2_static, out3_static).dispatch(1u)
               << s_dyn(tex2d, vol3d, out2_dyn, out3_dyn).dispatch(combos)
               << out2_static.copy_to(h2s.data())
               << out2_dyn.copy_to(h2d.data())
               << out3_static.copy_to(h3s.data())
               << out3_dyn.copy_to(h3d.data())
               << synchronize();

        for (uint32_t fi = 0u; fi < 4u; ++fi) {
            for (uint32_t ai = 0u; ai < 4u; ++ai) {
                const uint32_t idx = fi * 4u + ai;
                const auto f = static_cast<Sampler::Filter>(fi);
                const auto a = static_cast<Sampler::Address>(ai);
                const auto ref2 = host_sample_4x4(test_uv, f, a);
                expect(max_error(h2s[idx], ref2) < tol) << "2D static mismatch idx=" << idx;
                expect(max_error(h2d[idx], ref2) < tol) << "2D dyn mismatch idx=" << idx;
                expect(max_error(h3s[idx], h3d[idx]) < tol) << "3D static/dyn mismatch idx=" << idx;
            }
        }
    }

    // 3. Mipmap sample, sample_level, sample_grad, sample_grad_level + non-zero mip views (2D & 3D)
    {
        constexpr uint32_t mips2 = 4u;
        auto img = device.create_image<float>(PixelStorage::FLOAT4, make_uint2(8u), mips2);
        const std::array<float4, 4u> mip2_colors{
            make_float4(1.0f, 0.0f, 0.0f, 1.0f),
            make_float4(0.0f, 1.0f, 0.0f, 1.0f),
            make_float4(0.0f, 0.0f, 1.0f, 1.0f),
            make_float4(1.0f, 1.0f, 0.0f, 1.0f),
        };
        for (uint32_t l = 0u; l < mips2; ++l) {
            uint32_t s = 8u >> l;
            luisa::vector<float4> buf(static_cast<size_t>(s) * s, mip2_colors[l]);
            stream << img.view(l).copy_from(buf.data());
        }

        constexpr uint32_t mips3 = 3u;
        auto vol = device.create_volume<float>(PixelStorage::FLOAT4, make_uint3(4u), mips3);
        const std::array<float4, 3u> mip3_colors{
            make_float4(0.8f, 0.2f, 0.1f, 1.0f),
            make_float4(0.1f, 0.8f, 0.2f, 1.0f),
            make_float4(0.2f, 0.1f, 0.8f, 1.0f),
        };
        for (uint32_t l = 0u; l < mips3; ++l) {
            uint32_t s = 4u >> l;
            luisa::vector<float4> buf(static_cast<size_t>(s) * s * s, mip3_colors[l]);
            stream << vol.view(l).copy_from(buf.data());
        }

        auto out2 = device.create_buffer<float4>(14u);
        auto out3 = device.create_buffer<float4>(10u);

        Kernel1D k_mips = [&](ImageFloat i0, ImageFloat iv1,
                              VolumeFloat v0, VolumeFloat vv1,
                              BufferFloat4 o2, BufferFloat4 o3) noexcept {
            auto uv = make_float2(0.5f, 0.5f);
            auto uvw = make_float3(0.5f, 0.5f, 0.5f);
            // 2D full view
            o2.write(0u, tex2d_sample_static(i0, uv, SamplerFilter::LINEAR_LINEAR, SamplerAddress::EDGE));
            o2.write(1u, tex2d_sample_level_static(i0, uv, 0.0f, SamplerFilter::POINT, SamplerAddress::EDGE));
            o2.write(2u, tex2d_sample_level_static(i0, uv, 1.0f, SamplerFilter::POINT, SamplerAddress::EDGE));
            o2.write(3u, tex2d_sample_level_static(i0, uv, 2.0f, SamplerFilter::POINT, SamplerAddress::EDGE));
            o2.write(4u, tex2d_sample_level_static(i0, uv, 3.0f, SamplerFilter::POINT, SamplerAddress::EDGE));
            o2.write(5u, tex2d_sample_level_static(i0, uv, 0.5f, SamplerFilter::LINEAR_LINEAR, SamplerAddress::EDGE));
            o2.write(6u, tex2d_sample_level_static(i0, uv, 0.4f, SamplerFilter::LINEAR_POINT, SamplerAddress::EDGE));
            o2.write(7u, tex2d_sample_level_static(i0, uv, 0.6f, SamplerFilter::LINEAR_POINT, SamplerAddress::EDGE));
            o2.write(8u, tex2d_sample_grad_static(i0, uv, make_float2(0.125f, 0.0f), make_float2(0.0f, 0.125f), SamplerFilter::LINEAR_LINEAR, SamplerAddress::EDGE));
            o2.write(9u, tex2d_sample_grad_static(i0, uv, make_float2(0.5f, 0.0f), make_float2(0.0f, 0.5f), SamplerFilter::LINEAR_LINEAR, SamplerAddress::EDGE));
            o2.write(10u, tex2d_sample_grad_level_static(i0, uv, make_float2(0.5f, 0.0f), make_float2(0.0f, 0.5f), 0.0f, SamplerFilter::LINEAR_LINEAR, SamplerAddress::EDGE));
            // 2D view(1)
            o2.write(11u, tex2d_sample_static(iv1, uv, SamplerFilter::LINEAR_LINEAR, SamplerAddress::EDGE));
            o2.write(12u, tex2d_sample_level_static(iv1, uv, 1.0f, SamplerFilter::LINEAR_LINEAR, SamplerAddress::EDGE));
            o2.write(13u, tex2d_sample_grad_static(iv1, uv, make_float2(0.25f, 0.0f), make_float2(0.0f, 0.25f), SamplerFilter::LINEAR_LINEAR, SamplerAddress::EDGE));

            // 3D full view
            o3.write(0u, tex3d_sample_static(v0, uvw, SamplerFilter::LINEAR_LINEAR, SamplerAddress::EDGE));
            o3.write(1u, tex3d_sample_level_static(v0, uvw, 0.0f, SamplerFilter::POINT, SamplerAddress::EDGE));
            o3.write(2u, tex3d_sample_level_static(v0, uvw, 1.0f, SamplerFilter::POINT, SamplerAddress::EDGE));
            o3.write(3u, tex3d_sample_level_static(v0, uvw, 2.0f, SamplerFilter::POINT, SamplerAddress::EDGE));
            o3.write(4u, tex3d_sample_level_static(v0, uvw, 0.5f, SamplerFilter::LINEAR_LINEAR, SamplerAddress::EDGE));
            o3.write(5u, tex3d_sample_grad_static(v0, uvw, make_float3(0.25f, 0.0f, 0.0f), make_float3(0.0f, 0.25f, 0.0f), SamplerFilter::LINEAR_LINEAR, SamplerAddress::EDGE));
            o3.write(6u, tex3d_sample_grad_static(v0, uvw, make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), SamplerFilter::LINEAR_LINEAR, SamplerAddress::EDGE));
            o3.write(7u, tex3d_sample_grad_level_static(v0, uvw, make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), 0.0f, SamplerFilter::LINEAR_LINEAR, SamplerAddress::EDGE));
            // 3D view(1)
            o3.write(8u, tex3d_sample_static(vv1, uvw, SamplerFilter::LINEAR_LINEAR, SamplerAddress::EDGE));
            o3.write(9u, tex3d_sample_level_static(vv1, uvw, 1.0f, SamplerFilter::LINEAR_LINEAR, SamplerAddress::EDGE));
        };

        auto s_mips = device.compile(k_mips);
        std::array<float4, 14u> h2{};
        std::array<float4, 10u> h3{};
        stream << s_mips(img, img.view(1u), vol, vol.view(1u), out2, out3).dispatch(1u)
               << out2.copy_to(h2.data())
               << out3.copy_to(h3.data())
               << synchronize();

        expect(max_error(h2[0u], mip2_colors[0u]) < tol);
        expect(max_error(h2[1u], mip2_colors[0u]) < tol);
        expect(max_error(h2[2u], mip2_colors[1u]) < tol);
        expect(max_error(h2[3u], mip2_colors[2u]) < tol);
        expect(max_error(h2[4u], mip2_colors[3u]) < tol);
        expect(max_error(h2[5u], lerp(mip2_colors[0u], mip2_colors[1u], 0.5f)) < 0.01f);
        expect(max_error(h2[6u], mip2_colors[0u]) < tol);
        expect(max_error(h2[7u], mip2_colors[1u]) < tol);
        expect(max_error(h2[8u], mip2_colors[0u]) < tol);
        expect(max_error(h2[9u], mip2_colors[2u]) < tol);
        expect(max_error(h2[10u], mip2_colors[2u]) < tol);
        expect(max_error(h2[11u], mip2_colors[1u]) < tol);
        expect(max_error(h2[12u], mip2_colors[2u]) < tol);
        // On view(1) with ddx = (0.25, 0) (1 texel of view(1) == mip 1), exp2(1) * 0.25 * 8 = 4 -> physical mip 2
        expect(max_error(h2[13u], mip2_colors[2u]) < tol);

        expect(max_error(h3[0u], mip3_colors[0u]) < tol);
        expect(max_error(h3[1u], mip3_colors[0u]) < tol);
        expect(max_error(h3[2u], mip3_colors[1u]) < tol);
        expect(max_error(h3[3u], mip3_colors[2u]) < tol);
        expect(max_error(h3[4u], lerp(mip3_colors[0u], mip3_colors[1u], 0.5f)) < 0.01f);
        expect(max_error(h3[5u], mip3_colors[0u]) < tol);
        expect(max_error(h3[6u], mip3_colors[2u]) < tol);
        expect(max_error(h3[7u], mip3_colors[2u]) < tol);
        expect(max_error(h3[8u], mip3_colors[1u]) < tol);
        expect(max_error(h3[9u], mip3_colors[2u]) < tol);
    }

    // 4. BC1, BYTE4_SRGB, BC7_SRGB hardware decompression & sRGB decode (sample-only!)
    {
        auto bc1_img = device.create_image<float>(PixelStorage::BC1, make_uint2(4u, 4u));
        // BC1 block: c0=0xF800 (red 31,0,0), c1=0x0000, idx=0
        const std::array<uint8_t, 8u> bc1_block{0x00u, 0xF8u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u};

        auto srgb_img = device.create_image<float>(PixelStorage::BYTE4_SRGB, make_uint2(4u, 4u));
        std::array<uint8_t, 4u * 4u * 4u> srgb_bytes{};
        for (size_t i = 0u; i < 16u; ++i) {
            srgb_bytes[i * 4u + 0u] = 128u;
            srgb_bytes[i * 4u + 1u] = 64u;
            srgb_bytes[i * 4u + 2u] = 255u;
            srgb_bytes[i * 4u + 3u] = 255u;
        }

        auto bc7_srgb_img = device.create_image<float>(PixelStorage::BC7_SRGB, make_uint2(4u, 4u));
        const auto bc7_block = make_bc7_mode6_solid_block(64u, 32u, 0u, 127u, 0u);

        auto out = device.create_buffer<float4>(3u);
        Kernel1D k_compressed = [&](ImageFloat bc1, ImageFloat srgb, ImageFloat bc7, BufferFloat4 res) noexcept {
            auto uv = make_float2(0.5f, 0.5f);
            res.write(0u, tex2d_sample_static(bc1, uv, SamplerFilter::LINEAR_POINT, SamplerAddress::EDGE));
            res.write(1u, tex2d_sample_static(srgb, uv, SamplerFilter::LINEAR_POINT, SamplerAddress::EDGE));
            res.write(2u, tex2d_sample_static(bc7, uv, SamplerFilter::LINEAR_POINT, SamplerAddress::EDGE));
        };

        auto shader = device.compile(k_compressed);
        std::array<float4, 3u> h_out{};
        stream << bc1_img.copy_from(bc1_block.data())
               << srgb_img.copy_from(srgb_bytes.data())
               << bc7_srgb_img.copy_from(bc7_block.data())
               << shader(bc1_img, srgb_img, bc7_srgb_img, out).dispatch(1u)
               << out.copy_to(h_out.data())
               << synchronize();

        expect(max_error(h_out[0u], make_float4(1.0f, 0.0f, 0.0f, 1.0f)) < 1.0e-2f);
        const float4 srgb_expected{srgb_u8_to_linear(128u), srgb_u8_to_linear(64u), 1.0f, 1.0f};
        expect(max_error(h_out[1u], srgb_expected) < 1.0e-2f);
        const float4 bc7_expected{srgb_u8_to_linear(128u), srgb_u8_to_linear(64u), 0.0f, 254.0f / 255.0f};
        expect(max_error(h_out[2u], bc7_expected) < 1.5e-2f);

        if (is_cuda) {
            expect(eq(cuda_tex_count(bc1_img), 1u));
            expect(!cuda_has_table(bc1_img));
            expect(eq(cuda_tex_count(srgb_img), 1u));
            expect(!cuda_has_table(srgb_img));
            expect(eq(cuda_tex_count(bc7_srgb_img), 1u));
            expect(!cuda_has_table(bc7_srgb_img));
        }
    }

    // 5. Shader cache save + reload roundtrip (verifies CUDAShaderMetadata::texture_sampler_masks)
    {
        auto img = device.create_image<float>(PixelStorage::FLOAT4, make_uint2(4u));
        luisa::vector<float4> src(16u, make_float4(0.3f, 0.6f, 0.9f, 1.0f));
        auto out = device.create_buffer<float4>(1u);
        Kernel1D k_cached = [&](ImageFloat tex, BufferFloat4 res) noexcept {
            res.write(0u, tex2d_sample_static(tex, make_float2(0.5f, 0.5f), SamplerFilter::LINEAR_POINT, SamplerAddress::REPEAT));
        };
        constexpr luisa::string_view cache_name = "test_direct_sample_cache_v1";
        expect(device.compile_to(k_cached, cache_name));
        auto reloaded = device.load_shader<1, Image<float>, Buffer<float4>>(cache_name);
        float4 reloaded_val{};
        stream << img.copy_from(src.data())
               << reloaded(img, out).dispatch(1u)
               << out.copy_to(&reloaded_val)
               << synchronize();
        expect(max_error(reloaded_val, src[0u]) < tol);
        if (is_cuda) {
            expect(eq(cuda_tex_count(img), 1u));
            expect(!cuda_has_table(img));
        }
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    test_texture_direct_sample(dc->device);
}
