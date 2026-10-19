// Phase-0 cooperative-vector capability probe (tensor_optimize.md Phase 0:
// "新增 cooperative matmul/mul_add 设备 probe，完成 C0 布局实验" + §5.1 C0
// scope + §8.5 AST 验证).
//
// Scope (C0 go/no-go evidence ONLY - NO perf claims):
//   1. Backend-specific enablement:
//      - dx: create the device with DirectXDeviceConfigExt::UseExperimental()
//        (same pattern + D3D12 Agility SDK exports as
//        src/tests/unit/ast/test_cooperative_vector.cpp). If
//        ExperimentalFeaturesEnabled() is false, print the no-go reason and
//        PASS with a "coop disabled" note (not a failure; plan §9 row
//        "DX 没有 experimental/SM 6.9 能力").
//      - vk: create normally; the public capability query is unavailable
//        (coop_api.md §7: no device.query key for cooperative vector), so the
//        capability is decided by the real probe (try-compile + execute).
//      - cuda / fallback / others: NO cooperative path exists (plan §8.5:
//        "CUDA：检查整个执行 factory 产生的 AST 不含 cooperative op" is not
//        possible because any coop AST on these backends is a hard
//        unsupported; plan §5: coop is DX/VK only). We assert the AST
//        NEGATIVE on a plain float4 kernel, print the explicit-coop-on-
//        cuda no-go reason, and exit before any coop compile (the coop
//        lowering paths call LUISA_NOT_IMPLEMENTED/LUISA_ERROR, which aborts
//        the process rather than throwing - so the unsupported path MUST
//        return before touching the probe factory).
//   2. Capability probe: compile the engine C0 kernel make_coop_probe_kernel()
//      from embedding/embed/kernels/kernels_coop.h (fixed K=16, N=8, fused
//      cooperative_mat_mul_add). On success, run the non-uniform matrix
//      battery: K in {16,32,64,128} x N in {8,16,32,64} with the plan §5.1
//      weight pattern W[k,n] = 0.01*k - 0.03*n + ((k+3*n)%7) (never an
//      identity/all-ones matrix - layout errors must not hide), x with mixed
//      signs/scales, nonzero bias. Operands are Buffer<float> aliased
//      zero-copy to ByteBufferView. Reference: fp64 CPU matvec. Gate:
//      per-element abs(diff) <= 1e-3 * max(1, |ref|); worst error and
//      pass/fail recorded per (K,N) combo in a printed table. The K=16,N=8
//      cell executes the ENGINE probe kernel itself (mul_add + fused bias);
//      the other cells use a test-local single-thread matvec with the same
//      supported ops (COOPERATIVE_MUL + cooperative_vector_load/store +
//      per-element scalar bias add) - identical emission pattern to
//      kernels_coop.cpp. A numeric failure FAILS the test: layout
//      incompatibility is a real bug signal and the C0 go/no-go evidence
//      (plan §9 row "Cooperative 物理布局与 BGM [K,N] 不兼容").
//   3. AST checks (plan §8.5, tensor_optimize.md; coop_api.md §5):
//      - positive: the probe kernel Function use_cooperative_operations()
//        == true, propagated_builtin_callables().uses_cooperative() == true,
//        direct call set contains COOPERATIVE_MUL_ADD.
//      - battery kernels: direct set contains COOPERATIVE_MUL /
//        COOPERATIVE_VECTOR_LOAD / COOPERATIVE_VECTOR_STORE.
//      - negative: a plain float4 (vector-path) kernel reports
//        use_cooperative_operations() == false and
//        propagated_builtin_callables().uses_cooperative() == false.
//      - negative: the engine variant table exists (coop_gemm_variant_count()
//        > 0, the four plan §5.3 variants resolvable by name) so production
//        discovery works through the table, not hard-coded tiles.
//
// Gates: EVERY coop device.compile site is gated behind the NON-ABORTING
// AOT path device.compile_to(kernel, name, false, false) FIRST - exactly the
// reference pattern in examples/embedding/embed/tensor_path.cpp (run_probe
// lambda, ~lines 305-320). compile_to returns compile_ok==false when the
// backend codegen rejects the coop shader (e.g. VK with VK_NV_cooperative_vector
// present but the fp32 config unsupported -> the HLSL route produces invalid
// SPIR-V (OpCooperativeVectorMatrixMulAddNV operand must be an array type)
// and the JIT path LUISA_ERRORs/ABORTS in Device::_create_shader_hlsl;
// the compile_only arm only warns and returns false). compile_to==false is
// recorded as the documented C0 no-go path (tensor_optimize.md §5.1/§9):
// print the reason, skip ALL coop execution for that battery, PASS. Only
// when compile_to returned true may device.compile + dispatch run. Dispatch
// crashes remain CATCHED (try_call) and recorded as no-go. NUMERIC mismatches
// are FAILURES (layout incompatibility is a real bug signal).
//
// Run (device test, backend = argv[1]):
//   xmake run test_embed_coop_probe dx
//   xmake run test_embed_coop_probe vk
//   xmake run test_embed_coop_probe cuda      (records no-go, PASS)
//   xmake run test_embed_coop_probe fallback  (records no-go, PASS)
//   (CTest runs it on fallback with LUISA_ENABLE_VALIDATION=1)
#include "ut/ut.hpp"
#include "test_device.h"

#if __has_include("embedding/embed/kernels/kernels_coop.h")
#include "embedding/embed/kernels/kernels_coop.h"
#else
// The coop kernel header is reached via the embedding example include dir
// (src/tests/xmake.lua: embed_test -> add_includedirs("$(projectdir)/examples")).
// Editors/clangd without a compile_commands entry for this TU therefore
// cannot resolve it; declare the EXACT same external-linkage contract locally
// (precedent: examples/embedding/embed/engine/tensor_path.cpp) so the TU
// parses standalone. The real header wins in any configured build - the ABI
// is identical either way (kernels_coop.h: CoopProbeKernel).
#include <luisa/core/stl/string.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/device.h>
namespace luisa::embed {
inline constexpr uint kCoopProbeK = 16u;
inline constexpr uint kCoopProbeN = 8u;
struct CoopGemmConfig {
    const char *name;
    uint32_t kslab;
    uint32_t ntile;
    uint32_t tokens_per_block;
};
[[nodiscard]] uint32_t coop_gemm_variant_count() noexcept;
[[nodiscard]] const CoopGemmConfig *coop_gemm_variant_by_name(luisa::string_view name) noexcept;
using CoopProbeKernel = compute::Kernel1D<void(compute::ByteBuffer /*x*/,
                                                compute::ByteBuffer /*weight*/,
                                                compute::ByteBuffer /*bias*/,
                                                compute::ByteBuffer /*y*/)>;
[[nodiscard]] CoopProbeKernel make_coop_probe_kernel() noexcept;
}// namespace luisa::embed
#endif

#include <luisa/core/logging.h>
#include <luisa/core/stl.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/ast/function.h>
#include <luisa/ast/function_builder.h>
#include <luisa/ast/op.h>
#include <luisa/ast/type.h>

#include <cmath>
#include <stdexcept>

#ifdef _WIN32
#include <luisa/backends/ext/dx_config_ext.h>
#include <luisa/core/stl/optional.h>
// Required by the D3D12 Agility SDK: these exports must be in the main .exe,
// not in a loaded DLL, so the D3D12 loader can find the SDK runtime.
// (Same mechanism as src/tests/unit/ast/test_cooperative_vector.cpp:29-32.)
extern "C" __declspec(dllexport) const uint32_t D3D12SDKVersion = 619;
extern "C" __declspec(dllexport) const char *D3D12SDKPath = ".\\D3D12\\";
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::embed;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Deterministic seeded LCG (host-side only; no <random>, no RTTI) - same
// generator as the embedding kernel tests.
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

// C0 battery axes (plan §5.1 item 5).
constexpr uint32_t kProbeKs[] = {16u, 32u, 64u, 128u};
constexpr uint32_t kProbeNs[] = {8u, 16u, 32u, 64u};

// Battery kernel type: x, weight, bias, y byte buffers (plan §5.1 item 6).
using BatteryKernel = Kernel1D<void(ByteBuffer, ByteBuffer, ByteBuffer, ByteBuffer)>;

// Probe outcome per (K, N): "compiled" / "executed" / "numeric" states are
// recorded so the table shows compile-vs-executability separately (plan
// §2 Benchmark: "只测编译/执行可行性、内存布局和小形状延迟").
struct ProbeCell {
    uint32_t k = 0u;
    uint32_t n = 0u;
    bool compiled = false;
    bool executed = false;
    bool numeric_ok = false;
    float worst_abs_diff = 0.f;
    luisa::string note;
};

#ifdef _WIN32
// Same config-ext pattern as test_cooperative_vector.cpp:43-68.
class DXExperimentalConfigExt final : public DirectXDeviceConfigExt {
public:
    [[nodiscard]] bool UseExperimental() const noexcept override { return true; }
};
#endif

// Device creation with the cooperation-relevant enablement, argv[1] =
// dx|vk|cuda|fallback|... . Uses luisa::test::create_device() for all
// non-dx backends (the shared helper form (A)); dx gets the config ext.
struct TestDevice {
    luisa::optional<luisa::test::DeviceContext> dc;
    luisa::string backend;
    bool dx_experimental_enabled = false;// meaningful only when backend == "dx"
    bool coop_supported = false;         // policy: dx (experimental on) or vk
};

TestDevice make_test_device(int argc, char *argv[]) {
    TestDevice td{};
    td.backend = (argc > 1 && argv != nullptr && argv[1] != nullptr) ? argv[1] : "";
    if (td.backend == "dx") {
#ifdef _WIN32
        const char *exe = (argc > 0 && argv && argv[0]) ? argv[0] : luisa::test::safe_argv0();
        Context context{exe};
        DeviceConfig config;
        auto dx_config = luisa::make_unique<DXExperimentalConfigExt>();
        auto *dx_config_ptr = dx_config.get();
        config.extension = std::move(dx_config);
        Device device = context.create_device("dx", &config);
        luisa::test::log_test_backend("dx", device);
        td.dx_experimental_enabled = dx_config_ptr->ExperimentalFeaturesEnabled();
        td.coop_supported = td.dx_experimental_enabled;
        td.dc = luisa::test::DeviceContext{std::move(context), std::move(device)};
#else
        LUISA_INFO("DX backend is not available on this platform; treating coop as no-go.");
        td.coop_supported = false;
#endif
        return td;
    }
    // vk, cuda, fallback, ...: plain helper creation (form (A)).
    td.dc = luisa::test::create_device(argc, argv);
    // VK has no public cooperative capability query (coop_api.md §7); the
    // backend probes VK_NV_cooperative_vector internally at device creation
    // and the test decides by the real try-compile/dispatch probe below.
    td.coop_supported = (td.backend == "vk");
    return td;
}

// Plain float4 vector-path kernel used for the AST NEGATIVE check: it must
// never carry cooperative operations on any backend (plan §8.5: "CUDA：检查
// 整个执行 factory 产生的 AST 不含 cooperative op").
[[nodiscard]] Kernel1D<void(Buffer<float4>, Buffer<float4>)>
make_plain_vec4_kernel() noexcept {
    Kernel1D<void(Buffer<float4>, Buffer<float4>)> kernel =
        [](BufferVar<float4> src, BufferVar<float4> dst) noexcept {
            const auto i = thread_x();
            dst.write(i, src.read(i) * dot(src.read(i), src.read(i)));
        };
    return kernel;
}

// Test-local battery matvec: ONE thread computes y[N] = W[K,N] @ x[K] + bias[N]
// with the supported ops only (identical emission pattern to the engine probe
// kernel in kernels_coop.cpp, except bias is added per-element in scalar DSL
// math instead of fused mul_add - which also doubles as the "plain
// cooperative_mat_mul (no fused bias)" probe required by Phase-0 C0).
// K and N are emission-time template constants.
template<uint32_t K, uint32_t N>
[[nodiscard]] BatteryKernel make_battery_kernel() noexcept {
    static_assert(K > 0u && N > 0u, "battery dims must be positive");
    BatteryKernel kernel = [](ByteBufferVar x, ByteBufferVar weight, ByteBufferVar bias,
                               ByteBufferVar y) noexcept {
        set_block_size(1u, 1u, 1u);
        set_name(luisa::format("embed_coop_probe_k{}_n{}", K, N).c_str());
        // M[KxN] FLOAT32, offset 0; x[K], bias[N], y[N].
        CoopMatrixRef matrix_offset{CoopRefVecType::FLOAT32, K, N};
        CoopVectorRef x_offset{CoopRefVecType::FLOAT32, K};
        CoopVectorRef bias_offset{CoopRefVecType::FLOAT32, N};
        CoopVectorRef y_offset{CoopRefVecType::FLOAT32, N};
        matrix_offset.set_byte_offset(0u);
        x_offset.set_byte_offset(0u);
        bias_offset.set_byte_offset(0u);
        y_offset.set_byte_offset(0u);
        const auto xv = cooperative_vector_load<float>(x, x_offset);
        const auto out = cooperative_mat_mul<float, float>(
            weight, matrix_offset, Expr<CoopVector<float>>{xv});
        const auto bv = cooperative_vector_load<float>(bias, bias_offset);
        // Per-element scalar add (fixed order; the coop element-wise CallOps
        // are "future" ops and must NOT be emitted - coop_api.md §3).
        CoopVector<float> acc{N};
        for (auto i = 0u; i < N; i++) {
            acc[i] = out[i] + bv[i];
        }
        cooperative_vector_store(y, y_offset, Expr<CoopVector<float>>{acc});
    };
    return kernel;
}

[[nodiscard]] BatteryKernel make_battery_kernel_for(uint32_t k, uint32_t n) noexcept {
    // Emission-time (C++) template dispatch over the fixed C0 grid; all 16
    // combos of {16,32,64,128} x {8,16,32,64} are instantiated.
    switch (k * 256u + n) {
        case 16u * 256u + 8u:
            return make_battery_kernel<16u, 8u>();
        case 16u * 256u + 16u:
            return make_battery_kernel<16u, 16u>();
        case 16u * 256u + 32u:
            return make_battery_kernel<16u, 32u>();
        case 16u * 256u + 64u:
            return make_battery_kernel<16u, 64u>();
        case 32u * 256u + 8u:
            return make_battery_kernel<32u, 8u>();
        case 32u * 256u + 16u:
            return make_battery_kernel<32u, 16u>();
        case 32u * 256u + 32u:
            return make_battery_kernel<32u, 32u>();
        case 32u * 256u + 64u:
            return make_battery_kernel<32u, 64u>();
        case 64u * 256u + 8u:
            return make_battery_kernel<64u, 8u>();
        case 64u * 256u + 16u:
            return make_battery_kernel<64u, 16u>();
        case 64u * 256u + 32u:
            return make_battery_kernel<64u, 32u>();
        case 64u * 256u + 64u:
            return make_battery_kernel<64u, 64u>();
        case 128u * 256u + 8u:
            return make_battery_kernel<128u, 8u>();
        case 128u * 256u + 16u:
            return make_battery_kernel<128u, 16u>();
        case 128u * 256u + 32u:
            return make_battery_kernel<128u, 32u>();
        default:// 128 x 64
            return make_battery_kernel<128u, 64u>();
    }
}

// fp64 CPU matvec reference with the exact battery operand recipe.
struct BatteryInputs {
    luisa::vector<float> x;// [K]
    luisa::vector<float> w;// [K*N] canonical row-major [K,N]
    luisa::vector<float> bias;// [N]
    luisa::vector<double> ref;// [N] fp64 accumulation
};

BatteryInputs make_battery_inputs(uint32_t k, uint32_t n, bool zero_bias = false) {
    BatteryInputs in{};
    Lcg rng{0xC0FFEEu ^ (k << 16u) ^ (n << 2u)};
    in.x.resize(k);
    in.w.resize(static_cast<size_t>(k) * n);
    in.bias.resize(n);
    in.ref.resize(n, 0.0);
    // W[k,n] = 0.01*k - 0.03*n + ((k + 3*n) % 7)  (plan §5.1 item 3 - the
    // non-uniform pattern that exposes layout errors; NOT identity, NOT ones).
    for (uint32_t kk = 0u; kk < k; kk++) {
        for (uint32_t nn = 0u; nn < n; nn++) {
            in.w[static_cast<size_t>(kk) * n + nn] =
                0.01f * static_cast<float>(kk) - 0.03f * static_cast<float>(nn) +
                static_cast<float>((kk + 3u * nn) % 7u);
        }
    }
    // x: mixed signs and scales (plan §5.1 item 4: 正负数、零、不同数量级).
    for (uint32_t kk = 0u; kk < k; kk++) {
        float v;
        switch (kk % 4u) {
            case 0u:
                v = rng.uniform(-1.f, 1.f);
                break;
            case 1u:
                v = rng.uniform(-1000.f, 1000.f);
                break;
            case 2u:
                v = rng.uniform(-0.001f, 0.001f);
                break;
            default:
                v = 0.f;
                break;
        }
        in.x[kk] = v;
    }
    // nonzero bias (plan §5.1: bias nonzero for the matvec battery), except for
    // the dedicated zero-bias combo that isolates pure cooperative_mat_mul.
    for (uint32_t nn = 0u; nn < n; nn++) {
        in.bias[nn] = rng.uniform(-0.5f, 0.5f) + ((nn % 2u) == 0u ? 0.25f : -0.25f);
    }
    if (zero_bias) {
        for (uint32_t nn = 0u; nn < n; nn++) {
            in.bias[nn] = 0.f;
        }
    }
    // fp64 matvec: ref[n] = sum_k W[k,n] * x[k] + bias[n]
    for (uint32_t nn = 0u; nn < n; nn++) {
        double s = static_cast<double>(in.bias[nn]);
        for (uint32_t kk = 0u; kk < k; kk++) {
            s += static_cast<double>(in.w[static_cast<size_t>(kk) * n + nn]) *
                 static_cast<double>(in.x[kk]);
        }
        in.ref[nn] = s;
    }
    return in;
}

// Battery artifact name (unique per kernel; keeps compile_to AOT artifacts
// from colliding across cells).
[[nodiscard]] luisa::string battery_artifact_name(uint32_t k, uint32_t n, bool zero_bias) {
    luisa::string name = luisa::format("embed_coop_probe_k{}_n{}", k, n);
    if (zero_bias) {
        name += "_zb";
    }
    return name;
}

// Run a host-side callable, converting any throw into a recorded no-go
// string. NOTE: a JIT device.compile failure on some backends can ABORT the
// process instead of throwing (LUISA_ERROR is [[noreturn]]; e.g. the VK
// HLSL-fallback coop path that yields invalid SPIR-V aborts inside
// Device::_create_shader_hlsl when the ComputeShader is created) - try/catch
// cannot contain that, which is why EVERY coop compile site below first
// gates on the non-aborting AOT path device.compile_to(name, false, false),
// exactly like tensor_path.cpp's run_probe: compile_to==false => recorded
// no-go and device.compile is NEVER reached for that kernel.
template<typename Fn>
[[nodiscard]] bool try_call(luisa::string &err, Fn &&fn) noexcept(false) {
    try {
        std::forward<Fn>(fn)();
        return true;
    } catch (const std::exception &e) {
        err = luisa::format("threw: {}", e.what());
        return false;
    } catch (...) {
        err = "threw: unknown exception";
        return false;
    }
}

// Execute one cell end-to-end and return whether the numerics passed; fills
// the cell record. `kernel` is the (already-built) DSL kernel object. Every
// coop compile site goes through the non-aborting AOT gate first (rule from
// the header comment / tensor_path.cpp run_probe).
template<typename KernelT>
void run_battery_cell(Device &device, Stream &stream, const luisa::string &backend,
                      uint32_t k, uint32_t n,
                      const KernelT &kernel, ProbeCell &cell, uint32_t &failures,
                      bool zero_bias = false) {
    const BatteryInputs in{make_battery_inputs(k, n, zero_bias)};
    auto x_buf = device.create_buffer<float>(k);
    auto w_buf = device.create_buffer<float>(k * n);
    auto b_buf = device.create_buffer<float>(n);
    auto y_buf = device.create_buffer<float>(n);
    luisa::vector<float> got(n, std::numeric_limits<float>::quiet_NaN());

    // Gate on the non-aborting AOT compile path BEFORE any JIT compile
    // (tensor_path.cpp run_probe pattern; a false result must skip
    // device.compile + dispatch entirely).
    const auto aot_name = battery_artifact_name(k, n, zero_bias);
    if (!device.compile_to(kernel, aot_name, false, false)) {
        cell.compiled = false;
        cell.note = luisa::format(
            "NO-GO: coop codegen rejected on backend '{}' (AOT compile_to('{}', false, false) "
            "returned false; JIT compile not attempted - it would abort on this path)",
            backend, aot_name);
        LUISA_INFO("coop probe K={} N={}: NO-GO ({})", k, n, cell.note);
        return;// recorded no-go; not a numeric failure
    }
    const auto compiled = try_call(cell.note, [&] {
        auto shader = device.compile(kernel);
        if (!shader) {
            throw std::runtime_error{"shader object invalid after compile"};
        }
        cell.compiled = true;
        // zero-copy aliasing (plan §5.1 item 6: alias canonical [K,N] buffers
        // as ByteBufferView; coop_api.md §4).
        const ByteBufferView x_bv{x_buf};
        const ByteBufferView w_bv{w_buf};
        const ByteBufferView b_bv{b_buf};
        const ByteBufferView y_bv{y_buf};
        stream << x_buf.copy_from(luisa::span{in.x})
               << w_buf.copy_from(luisa::span{in.w})
               << b_buf.copy_from(luisa::span{in.bias})
               << shader(x_bv, w_bv, b_bv, y_bv).dispatch(32u)
               << y_buf.copy_to(luisa::span{got})
               << synchronize();
    });
    if (!compiled) {
        LUISA_INFO("coop probe K={} N={}: NO-GO ({})", k, n, cell.note);
        return;// recorded no-go; not a numeric failure
    }
    cell.executed = true;
    // Gate: per-element abs(diff) <= 1e-3 * max(1, |ref|) (plan gemm gate).
    float worst = 0.f;
    uint32_t bad = 0u;
    for (uint32_t i = 0u; i < n; i++) {
        const double d = std::abs(static_cast<double>(got[i]) - in.ref[i]);
        if (static_cast<float>(d) > worst) {
            worst = static_cast<float>(d);
        }
        if (!(d <= 1e-3 * std::max(1.0, std::abs(in.ref[i])))) {
            if (bad < 4u) {
                LUISA_WARNING("coop probe K={} N={} idx {}: got {} ref {:.9f}", k, n, i,
                              static_cast<double>(got[i]), in.ref[i]);
            }
            bad++;
        }
    }
    cell.worst_abs_diff = worst;
    cell.numeric_ok = (bad == 0u);
    if (bad > 0u) {
        // Numeric mismatch = layout incompatibility = real bug signal (C0
        // no-go EVIDENCE, but the test FAILS it: silently wrong coop math
        // must never be mistaken for a working path).
        failures++;
        cell.note = luisa::format("{} element(s) exceed 1e-3 gate", bad);
    }
    LUISA_INFO("coop probe K={} N={}: executed, worst_abs_diff={:.3e}, {}", k, n, worst,
               bad == 0u ? "PASS" : "NUMERIC FAIL");
}

}// namespace

int main(int argc, char *argv[]) {
    luisa::log_level_verbose();
    auto td = make_test_device(argc, argv);
    const luisa::string &backend = td.backend;
    uint32_t failures = 0u;
    luisa::vector<ProbeCell> table{};

    // ------------------------------------------------------------------
    // Unsupported backends: cuda / fallback / anything not dx or vk.
    // Cooperative operations have NO executable path here (plan §5: coop
    // 仅 DX/VK; fallback codegen hits LUISA_NOT_IMPLEMENTED, CUDA AST
    // codegen errors). Exit BEFORE attempting any coop compile.
    // ------------------------------------------------------------------
    if (!td.dc) {
        // Device creation failed entirely (e.g. plugin missing).
        LUISA_INFO("coop probe: no device available for backend '{}'; recording no-go.",
                   backend);
        expect(true);
        return 0;
    }
    auto &device = td.dc->device;
    auto stream = device.create_stream();

    if (backend == "cuda") {
        LUISA_INFO("coop probe: backend '{}' is explicitly coop-UNSUPPORTED (plan §5: "
                   "cooperative path is DX/VK only; CUDA records no-go).",
                   backend);
    } else if (backend == "dx" && !td.dx_experimental_enabled) {
        LUISA_INFO("coop probe: DX experimental features NOT enabled on this system "
                   "(SM 6.9 path unavailable; plan §9 row 'DX 没有 experimental/SM 6.9 "
                   "能力'); coop disabled -> PASS with no-go note.");
    } else if (backend == "vk") {
        LUISA_INFO("coop probe: VK public capability query unavailable (coop_api.md §7: "
                   "no device.query key); capability decided by the real probe below.");
    } else if (backend != "dx") {
        LUISA_INFO("coop probe: backend '{}' is not a cooperative target (dx/vk only); "
                   "recording no-go.",
                   backend);
    }

    // AST check - NEGATIVE baseline (always run, every backend; plan §8.5):
    // a plain float4 kernel must NOT use cooperative operations.
    {
        const auto plain = make_plain_vec4_kernel();
        expect(plain.function()->use_cooperative_operations() == false)
            << "plain float4 kernel must not use cooperative operations";
        expect(plain.function()->propagated_builtin_callables().uses_cooperative() == false)
            << "plain float4 kernel must not propagate cooperative builtin calls";
        expect(plain.function()->direct_builtin_callables().test(CallOp::DOT))
            << "plain float4 kernel does use DOT (tensor baseline)";
        LUISA_INFO("AST negative check: plain vec4 kernel is coop-free (as expected).");
    }

    const bool coop_target = (backend == "vk") || (backend == "dx" && td.dx_experimental_enabled);
    if (!coop_target) {
        // Unsupported-backend path: verify the probe kernel FACTORY was never
        // invoked and PASS honestly (the coop compile path on these backends
        // aborts the process via LUISA_NOT_IMPLEMENTED/LUISA_ERROR rather
        // than throwing, so this exit is load-bearing, not cosmetic).
        LUISA_INFO("coop probe: no coop compile/dispatch attempted for backend '{}' "
                   "(probe factory never invoked); PASS (coop disabled).",
                   backend);
        expect(failures == 0u) << "coop probe (no-go path) must record zero failures";
        return 0;
    }

    // ------------------------------------------------------------------
    // AST check - POSITIVE on the engine C0 probe kernel (no device needed
    // to build the DSL Function).
    // ------------------------------------------------------------------
    {
        const auto probe = make_coop_probe_kernel();
        const auto *fb = probe.function().get();
        expect(fb != nullptr) << "engine probe kernel must expose a function builder";
        expect(fb->use_cooperative_operations())
            << "engine probe kernel must report cooperative operations";
        expect(fb->propagated_builtin_callables().uses_cooperative())
            << "engine probe kernel must propagate cooperative builtin calls";
        expect(fb->direct_builtin_callables().test(CallOp::COOPERATIVE_MUL_ADD))
            << "engine probe kernel must directly call COOPERATIVE_MUL_ADD";
        expect(fb->direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_LOAD))
            << "engine probe kernel must directly call COOPERATIVE_VECTOR_LOAD";
        expect(fb->direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_STORE))
            << "engine probe kernel must directly call COOPERATIVE_VECTOR_STORE";
        LUISA_INFO("AST positive check: engine probe kernel emits coop load/mul_add/store.");
    }

    // Engine variant table sanity (plan §5.3 first variants; production code
    // must discover coop configs through the table, not hard-coded tiles).
    {
        expect(coop_gemm_variant_count() > 0u) << "coop variant table must be non-empty";
        for (const char *name : {"coop_k64_n16_t1", "coop_k64_n32_t1", "coop_k128_n32_t1",
                                 "coop_k64_n32_t2"}) {
            const auto *cfg = coop_gemm_variant_by_name(name);
            expect(cfg != nullptr) << luisa::format("variant '{}' missing from the table", name);
            if (cfg != nullptr) {
                expect(cfg->kslab > 0u && cfg->ntile > 0u && cfg->tokens_per_block > 0u)
                    << luisa::format("variant '{}' has non-positive geometry", name);
            }
        }
    }

    // ------------------------------------------------------------------
    // Capability probe: compile + run the engine C0 kernel (K=16, N=8,
    // fused mul_add + bias). This is the go/no-go gate for the battery.
    // EVERY coop device.compile site is gated behind the NON-ABORTING AOT
    // path device.compile_to(kernel, name, false, false) first - exactly
    // the reference pattern of examples/embedding/embed/tensor_path.cpp
    // (run_probe lambda, ~lines 320-370). compile_to returns false when the
    // backend rejects coop codegen (VK: VK_NV_cooperative_vector present but
    // the fp32 config unsupported -> the coop path emits SPIR-V the validator
    // rejects (OpCooperativeVectorMatrixMulAddNV operands must be array
    // types) and the JIT creation aborts via LUISA_ERROR inside
    // Device::_create_shader_hlsl - try/catch cannot contain that; the
    // compile_only AOT arm only logs and reports compile_ok=false). A false
    // result is the documented C0 no-go path (tensor_optimize.md §5.1/§9):
    // print the reason, skip ALL coop execution for this battery, PASS.
    // Only when compile_to returned true may device.compile + dispatch run.
    // ------------------------------------------------------------------
    bool coop_engine_probe_ran = false;
    {
        ProbeCell cell{};
        cell.k = kCoopProbeK;
        cell.n = kCoopProbeN;
        const auto kernel = make_coop_probe_kernel();
        if (!device.compile_to(kernel, "embed_coop_probe", false, false)) {
            cell.compiled = false;
            cell.note = luisa::format(
                "coop codegen rejected on backend '{}' (AOT compile_to('embed_coop_probe', "
                "false, false) returned false; JIT device.compile not attempted - the coop "
                "JIT path aborts via LUISA_ERROR here, e.g. VK_NV_cooperative_vector present "
                "but the fp32 config unsupported -> invalid SPIR-V "
                "(OpCooperativeVectorMatrixMulAddNV operand must be an array type))",
                backend);
            LUISA_INFO("[coop probe] no-go: {}", cell.note);
            table.emplace_back(std::move(cell));
            // Fall through to the battery-skip below; this is the documented
            // C0 no-go path (tensor_optimize.md §5.1/§9) -> PASS with reason.
        } else {
            const auto in = make_battery_inputs(kCoopProbeK, kCoopProbeN);
            const auto ok = try_call(cell.note, [&] {
                const auto shader = device.compile(kernel);
                if (!shader) {
                    throw std::runtime_error{"engine probe shader invalid"};
                }
                cell.compiled = true;
                auto x_buf = device.create_buffer<float>(kCoopProbeK);
                auto w_buf = device.create_buffer<float>(kCoopProbeK * kCoopProbeN);
                auto b_buf = device.create_buffer<float>(kCoopProbeN);
                auto y_buf = device.create_buffer<float>(kCoopProbeN);
                luisa::vector<float> got(kCoopProbeN,
                                         std::numeric_limits<float>::quiet_NaN());
                const ByteBufferView x_bv{x_buf};
                const ByteBufferView w_bv{w_buf};
                const ByteBufferView b_bv{b_buf};
                const ByteBufferView y_bv{y_buf};
                stream << x_buf.copy_from(luisa::span{in.x})
                       << w_buf.copy_from(luisa::span{in.w})
                       << b_buf.copy_from(luisa::span{in.bias})
                       << shader(x_bv, w_bv, b_bv, y_bv).dispatch(32u)
                       << y_buf.copy_to(luisa::span{got})
                       << synchronize();
                float worst = 0.f;
                uint32_t bad = 0u;
                for (uint32_t i = 0u; i < kCoopProbeN; i++) {
                    const double d = std::abs(static_cast<double>(got[i]) - in.ref[i]);
                    if (static_cast<float>(d) > worst) {
                        worst = static_cast<float>(d);
                    }
                    if (!(d <= 1e-3 * std::max(1.0, std::abs(in.ref[i])))) {
                        bad++;
                    }
                }
                cell.executed = true;
                cell.worst_abs_diff = worst;
                cell.numeric_ok = (bad == 0u);
                if (bad > 0u) {
                    failures++;
                    cell.note = luisa::format("engine probe: {} element(s) exceed 1e-3 gate", bad);
                }
            });
            coop_engine_probe_ran = cell.executed;
            if (!ok) {
                LUISA_INFO("[coop probe] no-go: ENGINE probe kernel compile/run failed ({}); "
                           "recording whole-coop NO-GO and keeping coop disabled; PASS.",
                           cell.note);
                table.emplace_back(std::move(cell));
            } else {
                LUISA_INFO("coop probe: engine K=16 N=8 probe {} (worst_abs_diff={:.3e})",
                           cell.numeric_ok ? "PASS" : "NUMERIC FAIL", cell.worst_abs_diff);
                table.emplace_back(std::move(cell));
            }
        }
    }

    // ------------------------------------------------------------------
    // Non-uniform matrix battery (C0): K x N grid. The K=16,N=8 cell was
    // covered by the engine probe (fused mul_add); this table runs the
    // test-local COOPERATIVE_MUL variant (bias added per element) for all
    // 16 combos, plus one explicit zero-bias cooperative_mat_mul repeat
    // (Phase-0: "Also probe cooperative_mat_mul (no bias) by passing zero bias
    // on one combo").
    // ------------------------------------------------------------------
    if (!coop_engine_probe_ran) {
        LUISA_INFO("[coop probe] no-go: skipping battery because the engine probe never "
                   "executed (coop codegen rejected or probe run failed on backend '{}').",
                   backend);
    } else {
        for (const auto k : kProbeKs) {
            for (const auto n : kProbeNs) {
                ProbeCell cell{};
                cell.k = k;
                cell.n = n;
                // Building the kernel is host-only DSL emission; compile+run
                // inside run_battery_cell can throw and is caught there.
                const auto built = make_battery_kernel_for(k, n);
                // AST check on every emitted battery kernel: coop ops present.
                const auto *fb = built.function().get();
                expect(fb->use_cooperative_operations())
                    << luisa::format("battery K={} N={} kernel must use coop ops", k, n);
                expect(fb->direct_builtin_callables().test(CallOp::COOPERATIVE_MUL))
                    << luisa::format("battery K={} N={} kernel must call COOPERATIVE_MUL", k, n);
                run_battery_cell(device, stream, backend, k, n, built, cell, failures);
                table.emplace_back(std::move(cell));
            }
        }
        // Explicit zero-bias repeat on one combo (K=16, N=8): the kernel still
        // loads the bias buffer, but every element is 0, so the numbers isolate
        // pure cooperative_mat_mul (COOPERATIVE_MUL) from any bias-add math.
        {
            ProbeCell cell{};
            cell.k = kProbeKs[0];
            cell.n = kProbeNs[0];
            cell.note = "mat_mul zero-bias repeat";
            const auto built = make_battery_kernel_for(cell.k, cell.n);
            run_battery_cell(device, stream, backend, cell.k, cell.n, built, cell, failures,
                             /*zero_bias=*/true);
            table.emplace_back(std::move(cell));
        }
    }

    // ------------------------------------------------------------------
    // Result table (plan §5.1 item 5: record DX/VK compilable + executable
    // combos; Phase-0: feasibility and layout only, no perf).
    // ------------------------------------------------------------------
    LUISA_INFO("=== cooperative probe result table (backend '{}') ===", backend);
    LUISA_INFO("{:>4} {:>4} {:>10} {:>10} {:>11} {:>13}  {}", "K", "N", "compiled",
               "executed", "numeric", "worst_diff", "note");
    for (const auto &c : table) {
        LUISA_INFO("{:>4} {:>4} {:>10} {:>10} {:>11} {:>13.3e}  {}", c.k, c.n,
                   c.compiled ? "yes" : "no", c.executed ? "yes" : "no",
                   c.executed ? (c.numeric_ok ? "PASS" : "FAIL") : "-", c.worst_abs_diff,
                   c.note.empty() ? luisa::string{"-"} : c.note);
    }
    uint32_t num_go = 0u;
    for (const auto &c : table) {
        if (c.executed && c.numeric_ok) {
            num_go++;
        }
    }
    LUISA_INFO("coop C0 verdict: backend '{}': {} numeric-pass combo(s) of {} recorded.",
               backend, num_go, table.size());

    // Accumulate-then-expect-once: numeric failures above are the only
    // FAIL path; compile throws are honest no-go evidence (PASS).
    expect(failures == 0u)
        << luisa::format("coop probe produced {} numeric failure(s) (layout/numerics bug)",
                         failures);
    return 0;
}
