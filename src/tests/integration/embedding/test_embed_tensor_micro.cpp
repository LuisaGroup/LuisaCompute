// Phase-0 tensor microtest (tensor_optimize.md Phase 0: "新增 tensor microtest：
// float4 alias、对齐/tail、DOT、REDUCE_SUM、FMA、length_squared").
//
// Coverage (each family vs an in-file scalar CPU reference, worst error reported):
//   (a) float4 view round-trip: Buffer<float> alias to BufferView<float4>
//       (read AND write through a Kernel1D with BufferVar<float4> args);
//       bit-exact gate.
//   (b) alignment / tail semantics: count=10 float buffer viewed as float4
//       truncates to floor(40/16) = 2 vec4s (bytes 0..31); the remaining 2
//       scalars (elements 8,9) are handled by a second kernel through an
//       aligned subview. Also checks subview byte-offset alignment rules
//       (alignof(float4) == 16; offset 0 is always aligned).
//   (c) DOT: dot(a4,b4) vs CPU 4-lane sum over many random cases.
//   (d) FMA: fma(a4,b4,c4) vs CPU per-lane a*b+c.
//   (e) REDUCE_SUM: reduce_sum(v4) vs CPU lane sum; length_squared(v4) vs
//       CPU sum of squares (same expression as dot(v,v)).
//   (f) make_float2 / make_float3 / make_float4 construction + component
//       access (.x/.y/.z/.w) round-trip.
//
// AST checks (tensor_optimize.md §8.5, positive part): the built Function for
// the DOT kernel must directly call CallOp::DOT; the FMA kernel must call
// CallOp::FMA; the reduce kernels CallOp::REDUCE_SUM / CallOp::LENGTH_SQUARED;
// and none of these tensor-path kernels may use cooperative operations
// (negative AST assert; the coop side is test_embed_coop_probe.cpp).
//
// Gates: float4 IO bit-exact (compare via luisa::bit_cast<uint32_t>, so NaN
// payloads compare too); DOT/FMA/REDUCE/length_squared within 1e-6 relative
// (gate = 1e-6f * max(1, |ref|)); worst error recorded and printed per family.
// No perf claims in Phase 0.
//
// Run (device test, backend = argv[1]; runs on ANY installed backend including
// fallback):
//   xmake run test_embed_tensor_micro dx
//   xmake run test_embed_tensor_micro fallback
//   xmake run test_embed_tensor_micro vk
//   xmake run test_embed_tensor_micro cuda
//   (CTest runs it on fallback with LUISA_ENABLE_VALIDATION=1)
#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>
#include <luisa/ast/function.h>
#include <luisa/ast/op.h>

#include <cmath>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Deterministic seeded LCG (host-side only; no <random>, no RTTI) - same
// generator as the other embedding kernel tests, so ALL backends and the CPU
// oracle consume identical data.
struct Lcg {
    uint32_t state = 0x12345678u;
    [[nodiscard]] uint32_t next() noexcept {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    }
    [[nodiscard]] float uniform(float lo, float hi) noexcept {
        return lo + (hi - lo) * (static_cast<float>(next() % 8388608u) / 8388608.f);
    }
};

// Family accumulator (same shape as test_embed_kernel_robust.cpp).
struct Family {
    const char *name = "";
    uint32_t cases = 0u;
    uint32_t failures = 0u;
    float worst_abs_diff = 0.f;
    float worst_rel_diff = 0.f;
    luisa::string worst_desc;
};

struct Parity {
    luisa::vector<Family> families{};
    uint32_t total_failures() const noexcept {
        uint32_t t = 0u;
        for (const auto &f : families) {
            t += f.failures;
        }
        return t;
    }
};

Family &family(Parity &p, const char *name) {
    for (auto &f : p.families) {
        if (f.name == name) {
            return f;
        }
    }
    p.families.emplace_back(Family{name});
    return p.families.back();
}

// Absolute (bit-pattern) element compare for the bit-exact families.
void check_bit_exact(Family &fam, luisa::span<const float> got,
                     luisa::span<const float> ref, const char *desc) noexcept {
    uint32_t bad = 0u;
    const size_t n = got.size();
    for (size_t i = 0u; i < n; i++) {
        if (luisa::bit_cast<uint32_t>(got[i]) != luisa::bit_cast<uint32_t>(ref[i])) {
            if (bad < 4u) {
                LUISA_WARNING("{} idx {}: got {} ref {} (bit-pattern mismatch)", desc, i,
                              static_cast<double>(got[i]), static_cast<double>(ref[i]));
            }
            bad++;
        }
    }
    fam.cases++;
    fam.failures += bad;
    LUISA_INFO("{} bit-exact: bad={}/{}", desc, bad, n);
}

// Hybrid abs+rel element compare; tol = rel * max(1, |ref|).
void check_hybrid(Family &fam, luisa::span<const float> got, luisa::span<const float> ref,
                  float rel, const char *desc) noexcept {
    uint32_t bad = 0u;
    float worst_a = 0.f;
    float worst_r = 0.f;
    size_t worst_i = 0u;
    const size_t n = got.size();
    for (size_t i = 0u; i < n; i++) {
        const float d = std::abs(got[i] - ref[i]);
        if (d > worst_a) {
            worst_a = d;
            worst_i = i;
        }
        if (!(d <= rel * std::max(1.f, std::abs(ref[i])))) {
            if (bad < 4u) {
                LUISA_WARNING("{} idx {}: got {} ref {} diff {}", desc, i,
                              static_cast<double>(got[i]), static_cast<double>(ref[i]),
                              static_cast<double>(d));
            }
            bad++;
        }
    }
    worst_r = worst_a / std::max(1.f, std::abs(ref[worst_i]));
    fam.cases++;
    fam.failures += bad;
    if (worst_a > fam.worst_abs_diff) {
        fam.worst_abs_diff = worst_a;
        fam.worst_rel_diff = worst_r;
        fam.worst_desc = luisa::format("{} idx {}", desc, worst_i);
    }
    LUISA_INFO("{}: max_abs_diff={:.3e} rel={:.3e} bad={}/{}", desc, worst_a, worst_r, bad, n);
}

// ---------------------------------------------------------------------------
// (a) float4 view round-trip (read and write through BufferVar<float4>).
// ---------------------------------------------------------------------------
void check_float4_alias_read(Device &device, Stream &stream, Parity &parity) {
    constexpr uint32_t count = 1024u;// 256 float4s, exact multiple
    Lcg rng{0xC0FFEE01u ^ count};
    luisa::vector<float> src(count);
    for (auto &f : src) {
        // mixed magnitudes and signs, incl. exact 0 (bit-exact gate)
        f = rng.uniform(-8.f, 8.f);
        if ((rng.next() % 17u) == 0u) {
            f = 0.f;
        }
    }
    auto in_buf = device.create_buffer<float>(count);
    auto out_buf = device.create_buffer<float>(count);
    luisa::vector<float> got(count, 0.f);

    // alias: Buffer<float> -> BufferView<float4> (zero-copy reinterpretation)
    const auto in_v4 = in_buf.view().as<float4>();
    const auto out_v4 = out_buf.view().as<float4>();
    // Host-side semantics contract: element count truncates.
    expect(eq(in_v4.size(), count / 4u)) << "float4 alias must truncate to floor(count/4)";

    auto copy_kernel = Kernel1D([&](BufferVar<float4> s, BufferVar<float4> d) noexcept {
        const auto i = thread_x();
        d.write(i, s.read(i));
    });
    auto shader = device.compile(copy_kernel);
    stream << in_buf.copy_from(luisa::span{src})
           << shader(in_v4, out_v4).dispatch(in_v4.size())
           << out_buf.copy_to(luisa::span{got})
           << synchronize();
    check_bit_exact(family(parity, "float4_alias_read"), got, src, "float4_alias_read n=1024");
}

void check_float4_alias_write(Device &device, Stream &stream, Parity &parity) {
    constexpr uint32_t n4 = 64u;
    constexpr uint32_t count = n4 * 4u;
    Lcg rng{0xC0FFEE02u ^ n4};
    luisa::vector<float> src(count);
    for (auto &f : src) {
        f = rng.uniform(-16.f, 16.f);
    }
    auto in_buf = device.create_buffer<float>(count);
    auto out_buf = device.create_buffer<float>(count);
    luisa::vector<float> got(count, 0.f);

    auto write_kernel = Kernel1D([&](BufferVar<float> s, BufferVar<float4> d) noexcept {
        const auto i = thread_x();
        const auto base = i * 4u;
        d.write(i, make_float4(s.read(base), s.read(base + 1u), s.read(base + 2u),
                               s.read(base + 3u)));
    });
    auto shader = device.compile(write_kernel);
    stream << in_buf.copy_from(luisa::span{src})
           << shader(in_buf.view(), out_buf.view().as<float4>()).dispatch(n4)
           << out_buf.copy_to(luisa::span{got})
           << synchronize();
    check_bit_exact(family(parity, "float4_alias_write"), got, src,
                    "float4_alias_write n=256");
}

// ---------------------------------------------------------------------------
// (b) alignment / tail semantics: count=10 -> floor(10/4)=2 vec4s + 2-scalar tail.
// ---------------------------------------------------------------------------
void check_float4_tail(Device &device, Stream &stream, Parity &parity) {
    constexpr uint32_t count = 10u;
    Lcg rng{0xC0FFEE03u ^ count};
    luisa::vector<float> src(count);
    for (auto &f : src) {
        f = rng.uniform(-4.f, 4.f);
    }
    auto in_buf = device.create_buffer<float>(count);
    auto out_buf = device.create_buffer<float>(count);
    luisa::vector<float> got(count, 0.f);

    // Size truncation: 10 floats = 40 bytes -> floor(40/16) = 2 float4s.
    const auto in_v4 = in_buf.view().as<float4>();
    const auto out_v4 = out_buf.view().as<float4>();
    expect(eq(in_v4.size(), 2u)) << "count=10 float4 view must hold exactly floor(10/4)=2 vec4s";
    // The 2 vec4s cover elements [0, 8); the tail is elements [8, 10).
    constexpr uint32_t tail_begin = 8u;
    constexpr uint32_t tail_count = count - tail_begin;
    // Alignment contract (documented, verified via aligned offsets only):
    // BufferView<float4> requires offset_bytes % alignof(float4)==0. alignof(float4)
    // is 16, so vec4 views must start at element offsets that are multiples of 4;
    // element offset 0 (used everywhere here) is always aligned.
    static_assert(alignof(float4) == 16u, "float4 alignment assumption for tail split");

    auto vec_kernel = Kernel1D([&](BufferVar<float4> s, BufferVar<float4> d) noexcept {
        const auto i = thread_x();
        d.write(i, s.read(i));
    });
    auto tail_kernel = Kernel1D([&](BufferVar<float> s, BufferVar<float> d) noexcept {
        const auto i = thread_x();
        d.write(i, s.read(i));
    });
    auto vec_shader = device.compile(vec_kernel);
    auto tail_shader = device.compile(tail_kernel);
    // Aligned subview for the scalar tail (float views have 4-byte alignment).
    const auto s_tail = in_buf.view().subview(tail_begin, tail_count);
    const auto d_tail = out_buf.view().subview(tail_begin, tail_count);
    expect(eq(s_tail.offset_bytes() % alignof(float4), 0u))
        << "tail split point (element 8 = byte 32) is also float4-aligned";

    stream << in_buf.copy_from(luisa::span{src})
           << vec_shader(in_v4, out_v4).dispatch(in_v4.size())
           << tail_shader(s_tail, d_tail).dispatch(tail_count)
           << out_buf.copy_to(luisa::span{got})
           << synchronize();
    check_bit_exact(family(parity, "float4_tail"), got, src,
                    "float4_tail count=10 (2 vec4 + 2 scalar tail)");
}

// ---------------------------------------------------------------------------
// (c) DOT
// ---------------------------------------------------------------------------
void check_dot(Device &device, Stream &stream, Parity &parity) {
    constexpr uint32_t n = 256u;
    Lcg rng{0xC0FFEE04u};
    luisa::vector<float> a(4u * n), b(4u * n), ref(n, 0.f);
    for (auto &f : a) {
        f = rng.uniform(-4.f, 4.f);
    }
    for (auto &f : b) {
        f = rng.uniform(-4.f, 4.f);
    }
    for (uint32_t i = 0u; i < n; i++) {
        double s = 0.0;// fp64 reference; the gate compares fp32 output vs max(1,|ref|)
        for (uint32_t k = 0u; k < 4u; k++) {
            s += static_cast<double>(a[4u * i + k]) * static_cast<double>(b[4u * i + k]);
        }
        ref[i] = static_cast<float>(s);
    }
    auto a_buf = device.create_buffer<float4>(n);
    auto b_buf = device.create_buffer<float4>(n);
    auto o_buf = device.create_buffer<float>(n);
    luisa::vector<float> got(n, 0.f);

    auto dot_kernel = Kernel1D([&](BufferVar<float4> av, BufferVar<float4> bv,
                                   BufferVar<float> ov) noexcept {
        const auto i = thread_x();
        ov.write(i, dot(av.read(i), bv.read(i)));
    });
    // AST check (tensor_optimize.md §8.5): the kernel Function really calls DOT
    // and does NOT use cooperative operations.
    {
        const auto &fb = *dot_kernel.function();
        expect(fb.use_cooperative_operations() == false) << "dot kernel must not use coop ops";
        expect(fb.direct_builtin_callables().test(CallOp::DOT)) << "dot kernel must call DOT";
    }
    auto shader = device.compile(dot_kernel);
    stream << a_buf.copy_from(luisa::span{a})
           << b_buf.copy_from(luisa::span{b})
           << shader(a_buf.view(), b_buf.view(), o_buf.view()).dispatch(n)
           << o_buf.copy_to(luisa::span{got})
           << synchronize();
    check_hybrid(family(parity, "dot"), got, ref, 1e-6f, "dot float4 x256");
}

// ---------------------------------------------------------------------------
// (d) FMA
// ---------------------------------------------------------------------------
void check_fma(Device &device, Stream &stream, Parity &parity) {
    constexpr uint32_t n = 256u;
    Lcg rng{0xC0FFEE05u};
    luisa::vector<float> a(4u * n), b(4u * n), c(4u * n);
    luisa::vector<float> ref(4u * n, 0.f);
    for (auto &f : a) {
        f = rng.uniform(-4.f, 4.f);
    }
    for (auto &f : b) {
        f = rng.uniform(-4.f, 4.f);
    }
    for (auto &f : c) {
        f = rng.uniform(-1.f, 1.f);
    }
    for (size_t i = 0u; i < a.size(); i++) {
        ref[i] = static_cast<float>(static_cast<double>(a[i]) * static_cast<double>(b[i]) +
                                    static_cast<double>(c[i]));
    }
    auto a_buf = device.create_buffer<float4>(n);
    auto b_buf = device.create_buffer<float4>(n);
    auto c_buf = device.create_buffer<float4>(n);
    auto o_buf = device.create_buffer<float4>(n);
    luisa::vector<float> got(4u * n, 0.f);

    auto fma_kernel = Kernel1D([&](BufferVar<float4> av, BufferVar<float4> bv,
                                   BufferVar<float4> cv, BufferVar<float4> ov) noexcept {
        const auto i = thread_x();
        ov.write(i, fma(av.read(i), bv.read(i), cv.read(i)));
    });
    {
        const auto &fb = *fma_kernel.function();
        expect(fb.use_cooperative_operations() == false) << "fma kernel must not use coop ops";
        expect(fb.direct_builtin_callables().test(CallOp::FMA)) << "fma kernel must call FMA";
    }
    auto shader = device.compile(fma_kernel);
    stream << a_buf.copy_from(luisa::span{a})
           << b_buf.copy_from(luisa::span{b})
           << c_buf.copy_from(luisa::span{c})
           << shader(a_buf.view(), b_buf.view(), c_buf.view(), o_buf.view()).dispatch(n)
           << o_buf.copy_to(luisa::span{got})
           << synchronize();
    check_hybrid(family(parity, "fma"), got, ref, 1e-6f, "fma float4 x256");
}

// ---------------------------------------------------------------------------
// (e) REDUCE_SUM and length_squared
// ---------------------------------------------------------------------------
void check_reduce_sum(Device &device, Stream &stream, Parity &parity) {
    constexpr uint32_t n = 256u;
    Lcg rng{0xC0FFEE06u};
    luisa::vector<float> v(4u * n);
    luisa::vector<float> ref(n, 0.f);
    for (auto &f : v) {
        f = rng.uniform(-4.f, 4.f);
    }
    for (uint32_t i = 0u; i < n; i++) {
        double s = 0.0;
        for (uint32_t k = 0u; k < 4u; k++) {
            s += static_cast<double>(v[4u * i + k]);
        }
        ref[i] = static_cast<float>(s);
    }
    auto in_buf = device.create_buffer<float4>(n);
    auto o_buf = device.create_buffer<float>(n);
    luisa::vector<float> got(n, 0.f);

    auto reduce_kernel = Kernel1D([&](BufferVar<float4> iv, BufferVar<float> ov) noexcept {
        const auto i = thread_x();
        ov.write(i, reduce_sum(iv.read(i)));
    });
    {
        const auto &fb = *reduce_kernel.function();
        expect(fb.use_cooperative_operations() == false) << "reduce kernel must not use coop ops";
        expect(fb.direct_builtin_callables().test(CallOp::REDUCE_SUM))
            << "reduce kernel must call REDUCE_SUM";
    }
    auto shader = device.compile(reduce_kernel);
    stream << in_buf.copy_from(luisa::span{v})
           << shader(in_buf.view(), o_buf.view()).dispatch(n)
           << o_buf.copy_to(luisa::span{got})
           << synchronize();
    check_hybrid(family(parity, "reduce_sum"), got, ref, 1e-6f, "reduce_sum float4 x256");
}

void check_length_squared(Device &device, Stream &stream, Parity &parity) {
    constexpr uint32_t n = 256u;
    Lcg rng{0xC0FFEE07u};
    luisa::vector<float> v(4u * n);
    luisa::vector<float> ls_ref(n, 0.f);
    luisa::vector<float> dot_ref(n, 0.f);
    for (auto &f : v) {
        f = rng.uniform(-4.f, 4.f);
    }
    for (uint32_t i = 0u; i < n; i++) {
        double s = 0.0;
        for (uint32_t k = 0u; k < 4u; k++) {
            const double x = static_cast<double>(v[4u * i + k]);
            s += x * x;
        }
        ls_ref[i] = static_cast<float>(s);
        // fp32 dot(v,v) reference: the two DSL ops may legitimately differ in the
        // last ulp (hardware instruction choice); the 1e-6 hybrid gate absorbs it.
        float d = 0.f;
        for (uint32_t k = 0u; k < 4u; k++) {
            d += v[4u * i + k] * v[4u * i + k];
        }
        dot_ref[i] = d;
    }
    auto in_buf = device.create_buffer<float4>(n);
    auto ls_buf = device.create_buffer<float>(n);
    auto dt_buf = device.create_buffer<float>(n);
    luisa::vector<float> ls_got(n, 0.f);
    luisa::vector<float> dt_got(n, 0.f);

    auto ls_kernel = Kernel1D([&](BufferVar<float4> iv, BufferVar<float> lv,
                                  BufferVar<float> dv) noexcept {
        const auto i = thread_x();
        const auto v4 = iv.read(i);
        lv.write(i, length_squared(v4));
        dv.write(i, dot(v4, v4));
    });
    {
        const auto &fb = *ls_kernel.function();
        expect(fb.use_cooperative_operations() == false) << "len2 kernel must not use coop ops";
        expect(fb.direct_builtin_callables().test(CallOp::LENGTH_SQUARED))
            << "len2 kernel must call LENGTH_SQUARED";
        expect(fb.direct_builtin_callables().test(CallOp::DOT))
            << "len2 kernel must call DOT (for the v.v cross-check)";
    }
    auto shader = device.compile(ls_kernel);
    stream << in_buf.copy_from(luisa::span{v})
           << shader(in_buf.view(), ls_buf.view(), dt_buf.view()).dispatch(n)
           << ls_buf.copy_to(luisa::span{ls_got})
           << dt_buf.copy_to(luisa::span{dt_got})
           << synchronize();
    auto &fam = family(parity, "length_squared");
    check_hybrid(fam, ls_got, ls_ref, 1e-6f, "length_squared float4 x256");
    check_hybrid(fam, dt_got, dot_ref, 1e-6f, "length_squared==dot(v,v) float4 x256");
}

// ---------------------------------------------------------------------------
// (f) make_float2/3/4 construction + component access.
// ---------------------------------------------------------------------------
void check_vector_construct(Device &device, Stream &stream, Parity &parity) {
    constexpr uint32_t n = 64u;
    Lcg rng{0xC0FFEE08u};
    luisa::vector<float> s(12u * n);
    for (auto &f : s) {
        f = rng.uniform(-32.f, 32.f);
    }
    // device output layout per thread: [f2.xyzw-style 4? no] -> 9 lanes:
    // f2.x f2.y | f3.x f3.y f3.z | f4.x f4.y f4.z f4.w
    luisa::vector<float> ref(9u * n, 0.f);
    luisa::vector<float> got(9u * n, 0.f);
    for (uint32_t i = 0u; i < n; i++) {
        const float *p = s.data() + 12u * i;
        // f2 from lanes 0,1; f3 from lanes 2,3,4; f4 from lanes 5,6,7,8
        ref[9u * i + 0u] = p[0];
        ref[9u * i + 1u] = p[1];
        ref[9u * i + 2u] = p[2];
        ref[9u * i + 3u] = p[3];
        ref[9u * i + 4u] = p[4];
        ref[9u * i + 5u] = p[5];
        ref[9u * i + 6u] = p[6];
        ref[9u * i + 7u] = p[7];
        ref[9u * i + 8u] = p[8];
    }
    auto in_buf = device.create_buffer<float>(12u * n);
    auto out_buf = device.create_buffer<float>(9u * n);

    auto ctor_kernel = Kernel1D([&](BufferVar<float> iv, BufferVar<float> ov) noexcept {
        const auto i = thread_x();
        const auto b = i * 12u;
        Var<float2> f2 = make_float2(iv.read(b), iv.read(b + 1u));
        Var<float3> f3 = make_float3(iv.read(b + 2u), iv.read(b + 3u), iv.read(b + 4u));
        Var<float4> f4 =
            make_float4(iv.read(b + 5u), iv.read(b + 6u), iv.read(b + 7u), iv.read(b + 8u));
        const auto o = i * 9u;
        // component access
        ov.write(o + 0u, f2.x);
        ov.write(o + 1u, f2.y);
        ov.write(o + 2u, f3.x);
        ov.write(o + 3u, f3.y);
        ov.write(o + 4u, f3.z);
        ov.write(o + 5u, f4.x);
        ov.write(o + 6u, f4.y);
        ov.write(o + 7u, f4.z);
        ov.write(o + 8u, f4.w);
    });
    auto shader = device.compile(ctor_kernel);
    stream << in_buf.copy_from(luisa::span{s})
           << shader(in_buf.view(), out_buf.view()).dispatch(n)
           << out_buf.copy_to(luisa::span{got})
           << synchronize();
    check_bit_exact(family(parity, "vector_ctor"), got, ref, "make_float2/3/4 + .x/.y/.z/.w");
}

}// namespace

int main(int argc, char *argv[]) {
    luisa::log_level_verbose();
    auto [context, device] = luisa::test::create_device(argc, argv);
    auto stream = device.create_stream();

    Parity parity{};
    check_float4_alias_read(device, stream, parity);
    check_float4_alias_write(device, stream, parity);
    check_float4_tail(device, stream, parity);
    check_dot(device, stream, parity);
    check_fma(device, stream, parity);
    check_reduce_sum(device, stream, parity);
    check_length_squared(device, stream, parity);
    check_vector_construct(device, stream, parity);

    // Per-family worst-case table (Phase-0 evidence; NO perf claims here).
    LUISA_INFO("=== tensor microtest summary (backend '{}') ===", device.backend_name());
    for (const auto &f : parity.families) {
        LUISA_INFO("[{:16}] cases={} failures={} worst_abs={:.3e} worst_rel={:.3e} ({})", f.name,
                   f.cases, f.failures, f.worst_abs_diff, f.worst_rel_diff, f.worst_desc);
    }
    expect(parity.total_failures() == 0u)
        << luisa::format("tensor microtest produced {} failures over the families above",
                         parity.total_failures());
    return 0;
}
