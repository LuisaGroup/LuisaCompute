// bge-m3 embedding engine -- tensor-path selection implementation
// (tensor_optimize.md sections 2.1/2.2/9; coop_api.md sections 6/7/9/10).
//
// See tensor_path.h for the API contract. Two independent resolvers live
// here:
//  - tensor_path_requested(): LUISA_EMBED_TENSOR_PATH env parse (magic-
//    static, log-once; plan 2.1 "解析并缓存"; idiom copied from
//    embed_p1_mode()/executor.cpp:142-155 and the lowercase string parse of
//    LUISA_EMBED_WEIGHTS at executor.cpp:230-261 -- here deliberately *not*
//    fail-closed like that precedent: the engine must never fail to start
//    over a config typo, plan 9's kill-switch spirit).
//  - query_cooperative_support(): the plan-2.2 capability gate. No backend
//    offers a public cooperative-vector query key (coop_api.md 6/7/10), so
//    the decisive check is the shipped try-compile-and-run pattern of
//    src/tests/unit/ast/test_cooperative_vector.cpp, with the C0
//    non-uniform matrix of plan 5.1 doubling as the layout validation.
#include "tensor_path.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/stream.h>

// The probe kernel factory lives in the coop kernel module
// (embedding/embed/kernels/kernels_coop.h: CoopProbeKernel =
// compute::Kernel1D over four ByteBuffers -- x, weight, bias, y; fixed
// K=16, N=8, one cooperative thread; kCoopProbeK/kCoopProbeN exported).
#include "embedding/embed/kernels/kernels_coop.h"

// Exception-safety availability. The probe guards its compile/dispatch with
// try/catch only where C++ exceptions actually exist: xmake builds compile
// the engine targets with exceptions = "no-cxx" plus _HAS_EXCEPTIONS=0 on
// Windows (scripts/xmake_func.lua:345-356) -- there try/catch is a compile
// error; the CMake build keeps exceptions enabled (and adds /EHsc for
// clang-cl, src/CMakeLists.txt:34 -- "catch at integration boundaries still
// compiles", cpp-style skill). The feature-test macro below makes the guard
// vanish exactly when the language does not provide it. Catching cannot
// help against the Luisa fatal path itself (LUISA_ERROR -> std::abort; see
// the caveat in query_cooperative_support) -- it only contains genuinely
// throwing (third-party) calls.
#if defined(_CPPUNWIND) || defined(__cpp_exceptions) || defined(__EXCEPTIONS)
#define LUISA_EMBED_PROBE_HAS_EXCEPTIONS 1
#else
#define LUISA_EMBED_PROBE_HAS_EXCEPTIONS 0
#endif

namespace luisa::embed {

using compute::ByteBufferView;
using compute::Device;
using compute::Stream;

namespace {

// trim both ends of the raw env value (spaces, tabs, CR/LF -- getenv can
// carry a trailing CR from a Windows setx).
[[nodiscard]] luisa::string_view trim(luisa::string_view s) noexcept {
    constexpr luisa::string_view kws{" \t\r\n\f\v"};
    const auto b = s.find_first_not_of(kws);
    if (b == luisa::string_view::npos) {
        return {};
    }
    const auto e = s.find_last_not_of(kws);
    return s.substr(b, e - b + 1u);
}

// exact value comparison; when case_insensitive, ASCII A-Z is folded to
// lower case (same fold as the LUISA_EMBED_WEIGHTS parser at
// executor.cpp:233-235).
[[nodiscard]] bool value_is(luisa::string_view v, luisa::string_view name) noexcept {
    luisa::string lower;
    lower.reserve(v.size());
    for (const auto c : v) {
        lower.push_back((c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c);
    }
    return lower == name;
}

}// namespace

TensorPath tensor_path_requested() noexcept {
    // Resolved exactly once per process (thread-safe magic static; the same
    // idiom as embed_p1_mode()/executor.cpp:142-155 -- "the resolvers are
    // magic-static, so a second read is free and consistent", executor.md
    // 6.3.3). The log fires inside the same one-shot init -> log-once.
    static const TensorPath path = [] {
        const auto *raw = std::getenv("LUISA_EMBED_TENSOR_PATH");
        const auto v = trim(raw != nullptr ? luisa::string_view{raw} : luisa::string_view{});
        if (v.empty()) {
            // Unset -> legacy: the new paths are opt-in (plan 0.2).
            LUISA_INFO("embedding tensor path: LUISA_EMBED_TENSOR_PATH unset -> '{}' (source=default; new paths are opt-in).",
                       tensor_path_name(TensorPath::Legacy));
            return TensorPath::Legacy;
        }
        // exact lowercase first (preferred spelling), then case-insensitive.
        if (v == "legacy" || value_is(v, "legacy")) {
            LUISA_INFO("embedding tensor path: requested '{}' (source=env{}).",
                       tensor_path_name(TensorPath::Legacy),
                       v == "legacy" ? "" : ", case-folded");
            return TensorPath::Legacy;
        }
        if (v == "vector" || value_is(v, "vector")) {
            LUISA_INFO("embedding tensor path: requested '{}' (source=env{}).",
                       tensor_path_name(TensorPath::Vector),
                       v == "vector" ? "" : ", case-folded");
            return TensorPath::Vector;
        }
        if (v == "coop" || value_is(v, "coop")) {
            LUISA_INFO("embedding tensor path: requested '{}' (source=env{}).",
                       tensor_path_name(TensorPath::Coop),
                       v == "coop" ? "" : ", case-folded");
            return TensorPath::Coop;
        }
        if (v == "auto" || value_is(v, "auto")) {
            LUISA_INFO("embedding tensor path: requested '{}' (source=env{}).",
                       tensor_path_name(TensorPath::Auto),
                       v == "auto" ? "" : ", case-folded");
            return TensorPath::Auto;
        }
        // Unknown value: never fail to start -- warn loudly and fall back
        // to the kill switch (plan 9). Deliberately NOT LUISA_ERROR even
        // though the LUISA_EMBED_WEIGHTS precedent (executor.cpp:244) is
        // fail-closed: this switch guards startup of the whole engine.
        LUISA_WARNING("embedding tensor path: LUISA_EMBED_TENSOR_PATH=\"{}\" is not legacy|vector|coop|auto; falling back to 'legacy' (the kill switch).",
                      v);
        return TensorPath::Legacy;
    }();
    return path;
}

const char *tensor_path_name(TensorPath path) noexcept {
    switch (path) {
        case TensorPath::Legacy: return "legacy";
        case TensorPath::Vector: return "vector";
        case TensorPath::Coop: return "coop";
        case TensorPath::Auto: return "auto";
    }
    return "unknown";
}

namespace {

// --- the C0 cooperative probe (tensor_optimize.md section 5.1) -----------
//
// Shape contract with make_coop_probe_kernel(): fixed K=16 (input/inner
// dim), N=8 (output dim), single thread -- kernels_coop.h declares
// "x, weight, bias, y; fixed K=16,N=8" (one cooperative_mat_mul_add:
// y[N] = M[K,N] @ x[K] + bias[N], CoopMatrixRef(K=16, N=8), fp32). These
// mirror the sibling header's kCoopProbeK/kCoopProbeN; when the shared
// header is active the static_assert below keeps the two in lockstep.
constexpr uint32_t kProbeK = 16u;
constexpr uint32_t kProbeN = 8u;
static_assert(kProbeK == kCoopProbeK && kProbeN == kCoopProbeN,
              "tensor_path probe geometry disagrees with kernels_coop.h");

// NON-UNIFORM matrix / mixed-sign-and-scale vector / off-center bias, per
// plan 5.1 C0 and 2.2: "不能用单位矩阵做测试后就宣布兼容，因为许多错误布局
// 也能通过单位矩阵" (an identity or all-ones test matrix hides layout
// errors). W[k,n] = 0.01*k - 0.03*n + ((k+3n)%7) makes every (k,n) entry
// distinct with both signs and several magnitudes; x mixes 1e-3 and ~1e0
// scales and negatives; the bias spans -1.0..-0.3 so a dropped/misplaced
// bias term (the mul-vs-mul_add question) changes the result.
void fill_probe_data(luisa::vector<float> &w, luisa::vector<float> &x, luisa::vector<float> &b,
                     luisa::vector<float> &ref_y) noexcept {
    w.resize(static_cast<size_t>(kProbeK) * kProbeN);
    x.resize(kProbeK);
    b.resize(kProbeN);
    ref_y.resize(kProbeN);
    for (uint32_t k = 0u; k < kProbeK; k++) {
        for (uint32_t n = 0u; n < kProbeN; n++) {
            w[static_cast<size_t>(k) * kProbeN + n] =
                0.01f * static_cast<float>(k) - 0.03f * static_cast<float>(n) +
                static_cast<float>((k + 3u * n) % 7u);
        }
    }
    for (uint32_t k = 0u; k < kProbeK; k++) {
        x[k] = 0.001f * static_cast<float>(k % 5u) - 0.5f * static_cast<float>((k + 1u) % 3u);
    }
    for (uint32_t n = 0u; n < kProbeN; n++) {
        b[n] = 0.1f * static_cast<float>(n) - 1.0f;
    }
    // fp64 CPU reference, naive accumulation in ascending-k order (plan 8.1
    // "Cooperative 微测试: 非均匀矩阵, fp64 CPU accumulation"; same math as
    // the probe kernel's single cooperative_mat_mul_add call).
    for (uint32_t n = 0u; n < kProbeN; n++) {
        double acc = static_cast<double>(b[n]);
        for (uint32_t k = 0u; k < kProbeK; k++) {
            acc += static_cast<double>(w[static_cast<size_t>(k) * kProbeN + n]) *
                   static_cast<double>(x[k]);
        }
        ref_y[n] = static_cast<float>(acc);
    }
}

// One gate element: abs(diff) <= 1e-3 * max(1, abs(ref)) (plan 8.1 -- the
// GEMM acceptance gate reused for the coop probe). Updates worst_abs_err.
[[nodiscard]] bool probe_element_ok(float got, float ref, float &worst_abs_err) noexcept {
    const auto diff = std::fabs(got - ref);
    worst_abs_err = worst_abs_err < diff ? diff : worst_abs_err;
    const auto scale = 1.0f > std::fabs(ref) ? 1.0f : std::fabs(ref);
    return diff <= 1e-3f * scale;
}

}// namespace

CooperativeSupport query_cooperative_support(Device &device) noexcept {
    // Cached per-process including the failure reason (the probe is a one-
    // time cold-start cost: a small AOT compile + a single-block dispatch).
    static const CooperativeSupport support = [&device] {
        const auto backend = device.backend_name();

        // Gate 1: backend allowlist (plan section 2: this engine enables
        // cooperative vectors on DX and VK ONLY. CUDA is refused even
        // though the CUDA codegen carries some cooperative implementations
        // -- plan section 0.3/2 forbid selecting the coop path on CUDA at
        // all ("CUDA 永不选择该路径"; the plan's CUDA target has its own
        // cooperative codegen route that is outside this plan's support
        // matrix, and plan 2.2 asks the engine to report cooperative=false
        // for CUDA internally -- which is exactly this allowlist).
        // Anything else -- fallback, metal, remote, simd, unknown -- is
        // refused without touching the compiler.
        const bool allowed = backend == "dx" || backend == "vk";
        if (!allowed) {
            return CooperativeSupport{false, luisa::format("backend '{}' is not a cooperative-vector target", backend)};
        }

        // Gate 2 (feature enabled): on both allowed backends the feature
        // state is probed INTERNALLY at device creation (DX: coop kernels
        // compile to SM 6.9, gated by DirectXDeviceConfigExt::
        // UseExperimental() -> D3D12EnableExperimentalFeatures, with the
        // ExperimentalFeaturesEnabled() feedback only visible to the device
        // CREATOR; VK: the VK_NV_cooperative_vector + replicated-composites
        // + Vulkan-memory-model probe in the backend). Neither exposes a
        // public Device::query key for it (coop_api.md sections 6 and 7 --
        // there is no public device.query("cooperative_vector") string on
        // any backend). So this gate cannot be read back from here and is
        // folded into gate 3, the try-compile pattern coop_api.md 7.1 calls
        // the shipped approach. NOTE: deliberately NOT calling
        // device.query("cooperative_vector") -- on DX an unknown key logs a
        // LUISA_WARNING every call (LCDevice.cpp:953-960) and both backends
        // return {}, i.e. noise with zero information. When the engine
        // layer adds the key plan 2.2 asks for, check it here FIRST and
        // fail before compiling. The caller that can observe the DX
        // round-trip (the experimental device factory, executor.md 5(b))
        // should reach coop/auto only when it saw the feature enabled.

        // Gate 3 (decisive) + gate 4 (the same run validates layout): the
        // real probe -- compile AND run AND compare against the fp64 CPU
        // reference with the C0 non-uniform matrix.
        //
        // ---- ABORT-vs-THROW CAVEAT (read before "fixing" this) ----------
        // LuisaCompute backends report fatal failures with LUISA_ERROR,
        // which is [[noreturn]] and calls std::abort()
        // (include/luisa/core/logging.h:112-133) -- NOT a C++ exception.
        // The device-test pattern in
        // src/tests/unit/ast/test_cooperative_vector.cpp confirms this is
        // the intended fatal surface (it checks ExperimentalFeaturesEnabled
        // and SKIPS device execution rather than catching anything). The VK
        // JIT path aborts on codegen/compile errors (vk/compute_shader.cpp
        // :272 "Compile Error"; the SPIR-V target-feature LUISA_ERRORs and
        // XIR-to-SPIRV handoff validators in
        // common/spirv/spirv_codegen/entry.cpp; vk/device.cpp:3658
        // "Vulkan SPIR-V cache load failed without recompilation"), and the
        // DX JIT path aborts on DXC errors (dx/Shader/ComputeShader.cpp
        // :165 error arm of compile_compute). try/catch CANNOT survive any
        // of that. The one non-aborting compile surface is the AOT
        // Device::compile_to (device.h:269-280): it returns compile_ok
        // = false instead of aborting when DXC rejects the HLSL (DX:
        // LCDevice.cpp save_compute arm + LUISA_WARNING only; VK HLSL
        // route: device.cpp "DXC compile error ... compile_ok = false").
        // On the VK NATIVE XIR-to-SPIRV route a coop op rejected by the
        // dialect/target layer can still abort mid-compile -- that residual
        // risk cannot be contained from host code and is why the CALLER
        // (executor; plan 2.2's four conditions) decides whether to reach
        // this probe at all (e.g. only after observing the DX
        // ExperimentalFeaturesEnabled() round-trip true, or a VK device
        // whose backend log shows VK_NV_cooperative_vector enabled).
        // Therefore:
        //  - the probe compiles with compile_to (AOT) FIRST and only
        //    creates the dispatchable JIT shader when that passed;
        //  - where the build really has C++ exceptions (see
        //    LUISA_EMBED_PROBE_HAS_EXCEPTIONS), everything is additionally
        //    wrapped in try/catch so throwing third-party layers cannot
        //    take the engine down;
        //  - in the xmake default build (exceptions = no-cxx) the guard
        //    compiles away -- a throwing callee would terminate, per the
        //    limitation above.
        // ------------------------------------------------------------------

        // The probe body. Split out so the exception guard below wraps
        // exactly the risky calls without duplicating them; everything
        // owns its resources through RAII (shader/buffers/command list/
        // stream destruct at scope exit). NOT marked noexcept: in builds
        // with C++ exceptions the guard below must be able to catch a
        // throwing third-party callee; the enclosing function's own
        // noexcept is what bounds the public API.
        const auto run_probe = [&]() -> CooperativeSupport {
            auto kernel = make_coop_probe_kernel();
            if (kernel.function() == nullptr) {
                return CooperativeSupport{false, "make_coop_probe_kernel() returned an invalid kernel"};
            }
            if (!device.compile_to(kernel, "embed_coop_probe", false, false)) {
                return CooperativeSupport{false, luisa::format(
                                                     "cooperative probe kernel failed to compile (SM 6.9 / NV cooperative-vector codegen rejected on backend '{}')",
                                                     backend)};
            }
            // Dispatchable shader (JIT creation; typically served from the
            // same codegen/cache path after the AOT gate passed).
            auto probe_shader = device.compile(kernel);
            if (!probe_shader || !probe_shader.compile_ok()) {
                return CooperativeSupport{false, "cooperative probe shader was not usable after compile"};
            }

            // Self-contained stream: create -> record -> commit ->
            // synchronize -> destroy (the function signature only receives
            // the Device; the local Stream destructs on scope exit).
            Stream stream = device.create_stream();
            if (!stream) {
                return CooperativeSupport{false, "device.create_stream() failed on the cooperative probe"};
            }

            // Canonical row-major fp32 buffers; weight [K,N] is what the
            // CoopMatrixRef addresses (the plan-2.2 layout question made
            // empirical).
            auto w_buf = device.create_buffer<float>(static_cast<size_t>(kProbeK) * kProbeN);
            auto x_buf = device.create_buffer<float>(kProbeK);
            auto b_buf = device.create_buffer<float>(kProbeN);
            auto y_buf = device.create_buffer<float>(kProbeN);
            if (!w_buf || !x_buf || !b_buf || !y_buf) {
                return CooperativeSupport{false, "failed to create the cooperative probe buffers"};
            }

            luisa::vector<float> w, x, b;
            luisa::vector<float> ref_y;
            fill_probe_data(w, x, b, ref_y);
            luisa::vector<float> got_y(kProbeN, 0.0f);

            auto cl = compute::CommandList::create();
            // Bind the typed buffers as ByteBuffers through zero-copy
            // ByteBufferViews (coop_api.md section 4; the worked matvec of
            // section 9 binds exactly this way:
            // shader(mat_buf_view, bias_buf_view, vec_buf_view,
            // out_buf.view()).dispatch(1u)). All offsets are 0 and every
            // buffer is 4-byte aligned, so the CoopVectorRef/CoopMatrixRef
            // alignment contract holds.
            cl << w_buf.copy_from(luisa::span<const float>{w})
               << x_buf.copy_from(luisa::span<const float>{x})
               << b_buf.copy_from(luisa::span<const float>{b})
               << probe_shader(ByteBufferView{x_buf},
                               ByteBufferView{w_buf},
                               ByteBufferView{b_buf},
                               ByteBufferView{y_buf})
                      .dispatch(32u)// one warp; only thread 0 performs the coop matvec
               << y_buf.copy_to(luisa::span<float>{got_y});
            stream << cl.commit() << compute::synchronize();

            float worst = 0.0f;
            for (uint32_t n = 0u; n < kProbeN; n++) {
                if (!probe_element_ok(got_y[n], ref_y[n], worst)) {
                    const auto msg = luisa::format(
                        "cooperative probe numeric failure at y[{}]: got {:.9g}, fp64 reference {:.9g} (gate abs(diff) <= 1e-3 * max(1,|ref|), worst abs err {:.3g}) -- layout/precision not validated on backend '{}'",
                        n, got_y[n], ref_y[n], worst, backend);
                    return CooperativeSupport{false, msg};
                }
            }
            LUISA_INFO("embedding coop probe: passed on backend '{}' (K={}, N={}, fp32 matvec+bias vs fp64 reference, gate 1e-3*max(1,|ref|), worst abs err {:.3g}).",
                       backend, kProbeK, kProbeN, worst);
            return CooperativeSupport{true, "probe passed"};
        };

#if LUISA_EMBED_PROBE_HAS_EXCEPTIONS
        // Exceptions really available in this build (CMake, or xmake with
        // enable_exception): guard the risky calls so a throwing
        // third-party boundary (DXC/validator DLL entry points) is reported
        // as "unsupported" instead of taking the engine down. A Luisa-side
        // fatal path still aborts -- see the caveat above.
        try {
            return run_probe();
        }
        catch (...) {
            return CooperativeSupport{false, luisa::format(
                                                 "cooperative probe threw an unexpected error on backend '{}' (treated as unsupported)",
                                                 backend)};
        }
#else
        // xmake default build: exceptions = "no-cxx", _HAS_EXCEPTIONS=0 --
        // try/catch would not compile here. The probe runs ungarded: the
        // AOT compile_to gate above is still non-aborting for DXC rejections,
        // and the caller decides whether to reach this probe at all (the
        // documented plan-2.2 division of the four capability conditions).
        return run_probe();
#endif
    }();
    return support;
}

TensorPath resolve_tensor_path(Device &device, TensorPath requested) noexcept {
    switch (requested) {
        case TensorPath::Legacy: {
            return TensorPath::Legacy;// kill switch, no capability query
        }
        case TensorPath::Vector: {
            return TensorPath::Vector;// vector ops: DX/VK/CUDA all eligible
        }
        case TensorPath::Coop:
        case TensorPath::Auto: {
            const auto support = query_cooperative_support(device);
            if (support.supported) {
                return TensorPath::Coop;
            }
            if (requested == TensorPath::Coop) {
                // Plan 2.1: an explicit coop request must NOT silently fall
                // back -- it errors clearly (never silently produce wrong
                // results). LUISA_ERROR logs and aborts; the executor calls
                // this only after tensor_path_requested() returned Coop, so
                // the operator sees the message at startup and fixes the
                // env/backend.
                LUISA_ERROR_WITH_LOCATION(
                    "LUISA_EMBED_TENSOR_PATH=coop requested but {}; falling back is NOT allowed for explicit coop",
                    support.reason);
                return TensorPath::Legacy;// unreachable: LUISA_ERROR aborts
            }
            // auto: cooperative unavailable -> vector (plan 2.2: the four
            // conditions must ALL hold for auto to pick coop; otherwise
            // record the reason and use vector). CUDA never gets coop from
            // auto -- the allowlist inside the query refuses it, so auto
            // resolves to vector there, exactly as plan 2.1/section 2
            // require ("CUDA 的 auto 永远只在 vector/legacy 中选").
            LUISA_INFO("embedding tensor path: auto -> 'vector' (cooperative unavailable: {}).",
                       support.reason);
            return TensorPath::Vector;
        }
    }
    // out-of-range enum value (config corruption): the kill switch.
    LUISA_WARNING("embedding tensor path: requested path value {} is not a known TensorPath; falling back to 'legacy'.",
                  static_cast<uint32_t>(requested));
    return TensorPath::Legacy;
}

void log_tensor_path_selection(luisa::string_view backend,
                               TensorPath requested,
                               TensorPath resolved,
                               luisa::string_view reason) noexcept {
    // Plan 2.1 logging requirement (single line): backend, requested,
    // actual path, reason. The GEMM/attention variants, weight format and
    // the remaining env fields are carried by the caller's own session/
    // executor summary line (executor.md 6.3).
    LUISA_INFO("embedding tensor path: backend='{}', requested='{}', resolved='{}', reason='{}'.",
               backend, tensor_path_name(requested), tensor_path_name(resolved), reason);
}

}// namespace luisa::embed
