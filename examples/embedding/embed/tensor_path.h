// bge-m3 embedding engine -- tensor-path selection infrastructure
// (tensor_optimize.md section 2.1 path enum + semantics, 2.2 capability
// probing, 9 kill switch).
//
// This is the single place that decides which compute path the embedding
// engine takes for projections:
//   - legacy: verbatim current kernel selection (the kill switch of plan
//     section 9 -- always available, never wrong on any backend).
//   - vector: plain float4/DOT/FMA vector ops; DX/VK/CUDA all eligible.
//   - coop:   cooperative-vector matvec; DX and VK ONLY (plan section 2:
//     "CUDA never selects this path"). An explicit coop request that the
//     device cannot satisfy is a hard error -- no silent fallback (plan
//     2.1: explicit coop must fail clearly, never silently produce wrong
//     results).
//   - auto:   coop when query_cooperative_support() says yes, vector
//     otherwise (research builds may later shape-select; for now auto only
//     adds the coop upgrade on top of the vector baseline).
//
// Path selection is resolved from the process environment once per process
// (thread-safe magic-static, same idiom as embed_p1_mode() in executor.cpp
// and the session.cpp env resolvers; plan 2.1: "配置在进程首次使用时解析并
// 缓存"). The resolved choice is always logged (plan 2.1 logging
// requirement: backend, requested path, actual path, reason).
//
// Cooperative capability probing follows plan 2.2's four conditions --
// backend allowed + feature enabled + real probe passed + layout validated.
// There is NO public device.query() capability string for cooperative
// vectors on any backend (DX's query only understands "device_name"; VK
// probes the NV extensions internally and exposes the status only through
// the backend-internal SPIR-V target-feature mask). The shipped, supported
// way to find out -- used by src/tests/unit/ast/test_cooperative_vector.cpp
// itself -- is the try-compile-and-run pattern; that is what
// query_cooperative_support() does here (see the caveat on its
// abort-vs-throw limitation in tensor_path.cpp).
#pragma once
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/runtime/device.h>

namespace luisa::embed {

// Tensor/operator path of the embedding engine (tensor_optimize.md 2.1).
// The numeric values double as the stable env-config encoding; keep them.
enum class TensorPath : uint32_t {
    Legacy = 0u,// current kernels, bit-for-bit (kill switch, plan 9)
    Vector = 1u,// plain vector/tensor ops (float4/DOT/FMA); DX/VK/CUDA
    Coop = 2u,  // cooperative-vector matvec; DX/VK only, explicit request errors when unsupported
    Auto = 3u,  // coop when the probe passes, else vector
};

// LUISA_EMBED_TENSOR_PATH = legacy|vector|coop|auto (case-insensitive;
// surrounding whitespace trimmed; exact lowercase preferred). Resolved once
// per process (magic-static). Unset -> Legacy: the new paths are opt-in
// (plan 0.2: "默认保留 legacy 路径"). An unknown value logs LUISA_WARNING
// and falls back to Legacy -- the engine must never fail to start over a
// config typo. The resolved value and its source (env/default) are logged
// once.
[[nodiscard]] TensorPath tensor_path_requested() noexcept;

// Result of the cooperative-vector capability query (plan 2.2).
struct CooperativeSupport {
    bool supported;    // true only when every gate passed, including the real device probe
    luisa::string reason;// human-readable; "probe passed" on success, else what failed
};

// Cooperative-vector support query for `device`. Gates, in order:
//  1. backend allowlist: only "dx" and "vk" can be supported -- anything
//     else (cuda, fallback, metal, ...) is never a cooperative-vector
//     target for this engine (plan section 2 table).
//  2. feature enabled: on both allowed backends the feature state is
//     probed INTERNALLY at device creation (DX: the experimental SM 6.9
//     path gated by DirectXDeviceConfigExt::UseExperimental(); VK: the
//     VK_NV_cooperative_vector probe in the backend). There is NO public
//     Device::query() capability key for cooperative vectors on either
//     backend (coop_api.md sections 6/7), so this gate cannot be read back
//     from here -- it is folded into gate 3, which is the shipped way to
//     find out ("try-compile" pattern, coop_api.md section 7.1).
//  3. real probe: compile and run make_coop_probe_kernel()
//     (embedding/embed/kernels/kernels_coop.h, K=16/N=8 fp32 matvec + bias
//     + load/store) with the NON-UNIFORM layout-sensitive test matrix of
//     plan 5.1 C0 -- W[k,n] = 0.01*k - 0.03*n + float((k+3*n)%7), mixed-
//     sign/scale x, bias[n] = 0.1*n - 1.0 -- against an fp64 CPU reference
//     under the gate abs(diff) <= 1e-3f * max(1.0f, abs(ref)) per element.
//     A uniform matrix would let a wrong layout pass, hence the C0 recipe.
//     This single probe is the decisive check AND doubles as the layout
//     validation required by plan 2.2 ("能力查询之后仍需一个小矩阵的实际
//     编译/执行 probe").
//  4. layout validated: implied by (3) -- the probe aliases a canonical
//     row-major [K,N] fp32 buffer through the CoopMatrixRef byte offset, so
//     a layout mismatch fails numerically (plan 5.1: never declare success
//     on an identity matrix).
//
// The result (including its failure reason) is cached per-process: the
// probe pays one small compile + dispatch on first use. Keep it self-
// contained: the probe stream is created on `device`, synchronized, and
// destroyed inside the call.
//
// ABORT-VS-THROW CAVEAT (see the note in tensor_path.cpp): LuisaCompute
// backends report fatal failures through LUISA_ERROR, which ABORTS the
// process instead of throwing, so no try/catch can contain them. Where the
// build actually has C++ exceptions (CMake; xmake with enable_exception)
// the probe additionally guards its calls with a catch-all so throwing
// third-party boundaries cannot take the engine down; the xmake default
// (exceptions = no-cxx) compiles the guard away. A backend that aborts
// mid-compile cannot be survived from here, and on the VK native
// XIR-to-SPIRV route a coop rejection may still abort, so the caller (the
// executor) decides whether to call this function at all for a given
// device configuration (plan 2.2: the caller owns the "backend allowed +
// feature enabled" conditions; this query supplies "real probe passed +
// layout validated").
[[nodiscard]] CooperativeSupport query_cooperative_support(compute::Device &device) noexcept;

// Resolve the ACTUAL path for `device` from the requested one (plan 2.1):
//   Legacy -> Legacy; Vector -> Vector (no coop query, CUDA-safe).
//   Coop   -> Coop when the support query passed; otherwise
//             LUISA_ERROR_WITH_LOCATION with an explicit message -- an
//             explicit coop request must NOT silently fall back.
//   Auto   -> Coop when the support query passed, Vector otherwise (the
//             fallback reason is logged at INFO).
// Never returns Coop when query_cooperative_support().supported == false.
[[nodiscard]] TensorPath resolve_tensor_path(compute::Device &device, TensorPath requested) noexcept;

// "legacy" / "vector" / "coop" / "auto" -- the exact env-value spellings.
[[nodiscard]] const char *tensor_path_name(TensorPath path) noexcept;

// Single LUISA_INFO line with the plan-2.1 logging fields (backend,
// requested path, resolved path, reason). Call after resolve_tensor_path().
void log_tensor_path_selection(luisa::string_view backend,
                               TensorPath requested,
                               TensorPath resolved,
                               luisa::string_view reason) noexcept;

}// namespace luisa::embed
