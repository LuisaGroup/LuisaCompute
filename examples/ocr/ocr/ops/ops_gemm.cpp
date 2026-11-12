// OCR engine -- GEMM / GEMV operator implementations (Phase 2).
// Device-free factories; correctness-first naive FP32 accumulation.
#include "ocr/ops/ops_gemm.h"

#include <luisa/dsl/sugar.h>

#include "ocr/ops/ops_common.h"
#include "ocr/quantization.h"
#include "ocr/runtime/shader_cache.h"

namespace luisa::ocr {

[[nodiscard]] luisa::string_view gemm_epilogue_name(GemmEpilogue epilogue) noexcept {
    switch (epilogue) {
        case GemmEpilogue::none: return "none";
        case GemmEpilogue::bias: return "bias";
        case GemmEpilogue::bias_gelu_erf: return "bias_gelu_erf";
        case GemmEpilogue::bias_gelu_tanh: return "bias_gelu_tanh";
        case GemmEpilogue::bias_relu: return "bias_relu";
        case GemmEpilogue::bias_relu6: return "bias_relu6";
        case GemmEpilogue::bias_silu: return "bias_silu";
        case GemmEpilogue::bias_hard_swish: return "bias_hard_swish";
        case GemmEpilogue::residual: return "residual";
        case GemmEpilogue::bias_residual: return "bias_residual";
        default: return "unknown";
    }
}

[[nodiscard]] luisa::string_view gemm_weight_format_name(GemmWeightFormat format) noexcept {
    switch (format) {
        case GemmWeightFormat::fp32: return "fp32";
        case GemmWeightFormat::fp16: return "fp16";
        case GemmWeightFormat::bf16: return "bf16";
        case GemmWeightFormat::int4_sym: return "int4_sym";
        case GemmWeightFormat::int4_asym: return "int4_asym";
        case GemmWeightFormat::int8_sym: return "int8_sym";
        default: return "unknown";
    }
}

[[nodiscard]] bool gemm_format_is_quantized(GemmWeightFormat format) noexcept {
    return format == GemmWeightFormat::int4_sym || format == GemmWeightFormat::int4_asym ||
           format == GemmWeightFormat::int8_sym;
}

namespace {

using namespace luisa::compute;

// Compile-time epilogue application. `acc` is the FP32 dot product; aux is
// bias[N] or residual[M*N] depending on the (compile-time) epilogue.
template<GemmEpilogue Ep>
[[nodiscard]] Var<float> apply_epilogue(Var<float> acc, const BufferFloat &aux, Var<uint> row, Var<uint> col,
                                        Var<uint> n) noexcept {
    if constexpr (Ep == GemmEpilogue::none) {
        return acc;
    } else if constexpr (Ep == GemmEpilogue::residual) {
        return acc + aux.read(row * n + col);
    } else {
        Var<float> v = acc + aux.read(col);
        if constexpr (Ep == GemmEpilogue::bias) { return v; }
        else if constexpr (Ep == GemmEpilogue::bias_gelu_erf) { return ocr_gelu_erf(v); }
        else if constexpr (Ep == GemmEpilogue::bias_gelu_tanh) { return ocr_gelu_tanh(v); }
        else if constexpr (Ep == GemmEpilogue::bias_relu) { return ocr_relu(v); }
        else if constexpr (Ep == GemmEpilogue::bias_relu6) { return ocr_relu6(v); }
        else if constexpr (Ep == GemmEpilogue::bias_silu) { return ocr_silu(v); }
        else if constexpr (Ep == GemmEpilogue::bias_hard_swish) { return ocr_hard_swish(v); }
        else if constexpr (Ep == GemmEpilogue::bias_residual) { return v + aux.read(n + row * n + col); }
        else { return v; }
    }
}

// One B element decoded to FP32 (storage-only). Format is compile-time.
// The dense path reads b_f32; every packed path reads b_packed (+scales/zp).
[[nodiscard]] Var<float> load_b_f32(const BufferFloat &b_f32, Var<uint> index) noexcept {
    return b_f32.read(index);
}
template<GemmWeightFormat Fmt>
[[nodiscard]] Var<float> load_b_packed(const BufferUInt &b_packed, const BufferFloat &scales, const BufferUInt &zero_points,
                                       Var<uint> index, Var<uint> group_size) noexcept {
    if constexpr (Fmt == GemmWeightFormat::fp16) {
        return ocr_read_half(b_packed, index, false);
    } else if constexpr (Fmt == GemmWeightFormat::bf16) {
        return ocr_read_half(b_packed, index, true);
    } else if constexpr (Fmt == GemmWeightFormat::int4_sym) {
        return ocr_read_int4(b_packed, scales, zero_points, index, group_size, true);
    } else if constexpr (Fmt == GemmWeightFormat::int4_asym) {
        return ocr_read_int4(b_packed, scales, zero_points, index, group_size, false);
    } else if constexpr (Fmt == GemmWeightFormat::int8_sym) {
        return ocr_read_int8(b_packed, scales, index, group_size);
    } else {
        return 0.0f;// fp32 never reaches the packed path
    }
}

// Naive prefill GEMM body, dense FP32 B: one thread per output element.
template<GemmEpilogue Ep>
void gemm_body_f32(const BufferFloat &A, const BufferFloat &B, const BufferFloat &C, const BufferFloat &aux,
                   Var<uint> M, Var<uint> N, Var<uint> K, Var<uint> padded_cols) noexcept {
    Var<uint> row = def(0u);
    Var<uint> col = def(0u);
    ocr_flattened_row_col(padded_cols, row, col);
    $if (row < M & col < N) {
        Var<float> acc = def(0.0f);
        $for (k, K) {
            acc += A.read(row * K + k) * load_b_f32(B, k * N + col);
        };
        C.write(row * N + col, apply_epilogue<Ep>(acc, aux, row, col, N));
    };
}

// Naive prefill GEMM body, packed B: one thread per output element.
template<GemmEpilogue Ep, GemmWeightFormat Fmt>
void gemm_body_packed(const BufferFloat &A, const BufferUInt &B, const BufferFloat &C, const BufferFloat &aux,
                      const BufferFloat &scales, const BufferUInt &zero_points,
                      Var<uint> M, Var<uint> N, Var<uint> K, Var<uint> group_size,
                      Var<uint> padded_cols) noexcept {
    Var<uint> row = def(0u);
    Var<uint> col = def(0u);
    ocr_flattened_row_col(padded_cols, row, col);
    $if (row < M & col < N) {
        Var<float> acc = def(0.0f);
        $for (k, K) {
            acc += A.read(row * K + k) * load_b_packed<Fmt>(B, scales, zero_points, k * N + col, group_size);
        };
        C.write(row * N + col, apply_epilogue<Ep>(acc, aux, row, col, N));
    };
}

// Decode GEMV/small-M body, dense FP32 B: one thread per output column.
template<GemmEpilogue Ep>
void gemv_body_f32(const BufferFloat &A, const BufferFloat &B, const BufferFloat &C, const BufferFloat &aux,
                   Var<uint> M, Var<uint> N, Var<uint> K) noexcept {
    const auto col = ocr_linear_id();
    $if (col < N) {
        $for (row, M) {
            Var<float> acc = def(0.0f);
            $for (k, K) {
                acc += A.read(row * K + k) * load_b_f32(B, k * N + col);
            };
            C.write(row * N + col, apply_epilogue<Ep>(acc, aux, row, col, N));
        };
    };
}

// Decode GEMV/small-M body, packed B: one thread per output column.
template<GemmEpilogue Ep, GemmWeightFormat Fmt>
void gemv_body_packed(const BufferFloat &A, const BufferUInt &B, const BufferFloat &C, const BufferFloat &aux,
                      const BufferFloat &scales, const BufferUInt &zero_points,
                      Var<uint> M, Var<uint> N, Var<uint> K, Var<uint> group_size) noexcept {
    const auto col = ocr_linear_id();
    $if (col < N) {
        $for (row, M) {
            Var<float> acc = def(0.0f);
            $for (k, K) {
                acc += A.read(row * K + k) * load_b_packed<Fmt>(B, scales, zero_points, k * N + col, group_size);
            };
            C.write(row * N + col, apply_epilogue<Ep>(acc, aux, row, col, N));
        };
    };
}

// Host-side switch over epilogue instantiations.
template<typename Make>
auto for_each_epilogue(GemmEpilogue ep, Make &&make) noexcept {
    switch (ep) {
        case GemmEpilogue::none: return make.template operator()<GemmEpilogue::none>();
        case GemmEpilogue::bias: return make.template operator()<GemmEpilogue::bias>();
        case GemmEpilogue::bias_gelu_erf: return make.template operator()<GemmEpilogue::bias_gelu_erf>();
        case GemmEpilogue::bias_gelu_tanh: return make.template operator()<GemmEpilogue::bias_gelu_tanh>();
        case GemmEpilogue::bias_relu: return make.template operator()<GemmEpilogue::bias_relu>();
        case GemmEpilogue::bias_relu6: return make.template operator()<GemmEpilogue::bias_relu6>();
        case GemmEpilogue::bias_silu: return make.template operator()<GemmEpilogue::bias_silu>();
        case GemmEpilogue::bias_hard_swish: return make.template operator()<GemmEpilogue::bias_hard_swish>();
        case GemmEpilogue::residual: return make.template operator()<GemmEpilogue::residual>();
        case GemmEpilogue::bias_residual: return make.template operator()<GemmEpilogue::bias_residual>();
        default: return make.template operator()<GemmEpilogue::none>();
    }
}

}// namespace

[[nodiscard]] GemmKernel make_gemm_kernel(GemmEpilogue epilogue) noexcept {
    return for_each_epilogue(epilogue, []<GemmEpilogue Ep>() noexcept {
        GemmKernel kernel = [](BufferFloat A, BufferFloat B, BufferFloat C, BufferFloat aux,
                               Var<uint> M, Var<uint> N, Var<uint> K, Var<uint> padded_cols) noexcept {
            set_block_size(16u, 16u, 1u);
            set_name("ocr_gemm_fp32");
            gemm_body_f32<Ep>(A, B, C, aux, M, N, K, padded_cols);
        };
        return kernel;
    });
}

[[nodiscard]] GemmPackedKernel make_gemm_packed_kernel(GemmEpilogue epilogue, GemmWeightFormat format) noexcept {
    const auto build = [&]<GemmWeightFormat Fmt>() noexcept {
        return for_each_epilogue(epilogue, []<GemmEpilogue Ep>() noexcept {
            GemmPackedKernel kernel = [](BufferFloat A, BufferUInt B, BufferFloat C, BufferFloat aux,
                                         BufferFloat scales, BufferUInt zero_points,
                                         Var<uint> M, Var<uint> N, Var<uint> K, Var<uint> group_size,
                                         Var<uint> padded_cols) noexcept {
                set_block_size(16u, 16u, 1u);
                set_name("ocr_gemm_packed");
                gemm_body_packed<Ep, Fmt>(A, B, C, aux, scales, zero_points, M, N, K, group_size, padded_cols);
            };
            return kernel;
        });
    };
    switch (format) {
        case GemmWeightFormat::fp16: return build.template operator()<GemmWeightFormat::fp16>();
        case GemmWeightFormat::bf16: return build.template operator()<GemmWeightFormat::bf16>();
        case GemmWeightFormat::int4_sym: return build.template operator()<GemmWeightFormat::int4_sym>();
        case GemmWeightFormat::int4_asym: return build.template operator()<GemmWeightFormat::int4_asym>();
        case GemmWeightFormat::int8_sym: return build.template operator()<GemmWeightFormat::int8_sym>();
        default: return build.template operator()<GemmWeightFormat::fp16>();// fp32 uses make_gemm_kernel
    }
}

[[nodiscard]] Result<GemmPackedKernel> make_gemm_w4a16_kernel(GemmEpilogue epilogue) noexcept {
    // INDEPENDENT W4A16 variant. Phase 2 implements FP32 activation arithmetic
    // only; FP16 activation arithmetic is a later, separately gated arm. Fail
    // closed rather than silently substituting the W4A32 path.
    (void)epilogue;
    return Result<GemmPackedKernel>::failure(
        StatusCode::unsupported,
        "W4A16 requires FP16 activation arithmetic, which is not implemented in Phase 2; use the W4A32 (FP32 activation) variant",
        "w4a16");
}

[[nodiscard]] GemvKernel make_gemv_kernel(GemmEpilogue epilogue) noexcept {
    return for_each_epilogue(epilogue, []<GemmEpilogue Ep>() noexcept {
        GemvKernel kernel = [](BufferFloat A, BufferFloat B, BufferFloat C, BufferFloat aux,
                               Var<uint> M, Var<uint> N, Var<uint> K) noexcept {
            set_block_size(256u, 1u, 1u);
            set_name("ocr_gemv_fp32");
            gemv_body_f32<Ep>(A, B, C, aux, M, N, K);
        };
        return kernel;
    });
}

[[nodiscard]] GemvPackedKernel make_gemv_packed_kernel(GemmEpilogue epilogue, GemmWeightFormat format) noexcept {
    const auto build = [&]<GemmWeightFormat Fmt>() noexcept {
        return for_each_epilogue(epilogue, []<GemmEpilogue Ep>() noexcept {
            GemvPackedKernel kernel = [](BufferFloat A, BufferUInt B, BufferFloat C, BufferFloat aux,
                                         BufferFloat scales, BufferUInt zero_points,
                                         Var<uint> M, Var<uint> N, Var<uint> K, Var<uint> group_size) noexcept {
                set_block_size(256u, 1u, 1u);
                set_name("ocr_gemv_packed");
                gemv_body_packed<Ep, Fmt>(A, B, C, aux, scales, zero_points, M, N, K, group_size);
            };
            return kernel;
        });
    };
    switch (format) {
        case GemmWeightFormat::fp16: return build.template operator()<GemmWeightFormat::fp16>();
        case GemmWeightFormat::bf16: return build.template operator()<GemmWeightFormat::bf16>();
        case GemmWeightFormat::int4_sym: return build.template operator()<GemmWeightFormat::int4_sym>();
        case GemmWeightFormat::int4_asym: return build.template operator()<GemmWeightFormat::int4_asym>();
        case GemmWeightFormat::int8_sym: return build.template operator()<GemmWeightFormat::int8_sym>();
        default: return build.template operator()<GemmWeightFormat::fp16>();
    }
}

[[nodiscard]] GatedActKernel make_gated_act_kernel(uint32_t act) noexcept {
    // act is a compile-time host value captured by the factory lambda.
    GatedActKernel kernel = [act](BufferFloat gate, BufferFloat up, BufferFloat C, Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_gated_act");
        const auto i = ocr_linear_id();
        $if (i < count) {
            const auto g = gate.read(i);
            const auto u = up.read(i);
            Var<float> activated = def(0.0f);
            switch (act) {
                case 0u: activated = ocr_silu(g); break;
                case 1u: activated = ocr_gelu_erf(g); break;
                case 2u: activated = ocr_gelu_tanh(g); break;
                case 3u: activated = ocr_relu(g); break;
                default: activated = g; break;
            }
            C.write(i, activated * u);
        };
    };
    return kernel;
}

namespace {

// S1: the kernels index A/B/C with uint32; any logical buffer element count
// beyond UINT32_MAX would wrap. Fail closed before any allocation/launch.
[[nodiscard]] Status check_gemm_envelope(uint64_t m, uint64_t n, uint64_t k) {
    uint64_t ak = 0, kn = 0, mn = 0;
    if (!checked_mul(m, k, ak) || ak > UINT32_MAX) {
        return Status::error(StatusCode::invalid_argument,
                             "M*K exceeds the uint32 index envelope; chunk rows", "m*k");
    }
    if (!checked_mul(k, n, kn) || kn > UINT32_MAX) {
        return Status::error(StatusCode::invalid_argument,
                             "K*N exceeds the uint32 index envelope; chunk columns", "k*n");
    }
    if (!checked_mul(m, n, mn) || mn > UINT32_MAX) {
        return Status::error(StatusCode::invalid_argument,
                             "M*N exceeds the uint32 index envelope; chunk rows", "m*n");
    }
    return Status::success();
}

}// namespace

[[nodiscard]] Result<LaunchPlan2D> plan_gemm_dispatch(uint64_t m, uint64_t n, uint64_t k) {
    using R = Result<LaunchPlan2D>;
    auto env = check_gemm_envelope(m, n, k);
    if (!env.ok()) { return R::failure(env); }
    return plan_threads_2d(m, n, 16u, 16u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_gemv_dispatch(uint64_t m, uint64_t n, uint64_t k) {
    using R = Result<LaunchPlan1D>;
    auto env = check_gemm_envelope(m, n, k);
    if (!env.ok()) { return R::failure(env); }
    return plan_threads(n, 256u);
}

// Epilogue-aware plan: additionally proves the AUX buffer element count fits
// the uint32 index envelope (bias_residual aux = N + M*N can exceed M*N).
// gemm_aux_elements itself stays a pure size reporter.
[[nodiscard]] Result<LaunchPlan2D> plan_gemm_dispatch_epilogue(uint64_t m, uint64_t n, uint64_t k,
                                                               GemmEpilogue epilogue) {
    using R = Result<LaunchPlan2D>;
    auto aux = gemm_aux_elements(epilogue, m, n);
    if (!aux.ok()) { return R::failure(aux.status); }
    auto aux32 = checked_u32(*aux.value, "gemm aux element count");
    if (!aux32.ok()) { return R::failure(aux32.status); }
    return plan_gemm_dispatch(m, n, k);
}

[[nodiscard]] Result<uint64_t> gemm_aux_elements(GemmEpilogue epilogue, uint64_t m, uint64_t n) {
    using R = Result<uint64_t>;
    uint64_t mn = 0;
    if (!checked_mul(m, n, mn)) {
        return R::failure(StatusCode::invalid_argument, "M*N overflows uint64", "m*n");
    }
    switch (epilogue) {
        case GemmEpilogue::none: return R::success(0u);
        case GemmEpilogue::residual: return R::success(mn);
        case GemmEpilogue::bias_residual: {
            uint64_t total = 0;
            if (!checked_add(n, mn, total)) {
                return R::failure(StatusCode::invalid_argument, "aux size overflows uint64", "aux");
            }
            return R::success(total);
        }
        default: return R::success(n);// bias family
    }
}

[[nodiscard]] Result<LaunchPlan1D> plan_gated_act_dispatch(uint64_t count) {
    return plan_threads(count, 256u);
}

[[nodiscard]] luisa::string gemm_variant_key(GemmEpilogue epilogue, GemmWeightFormat format, bool gemv) {
    luisa::string variant{gemv ? "gemv_" : "gemm_"};
    variant.append(gemm_weight_format_name(format));
    variant.append("_");
    variant.append(gemm_epilogue_name(epilogue));
    return make_shader_key("gemm", variant);
}

}// namespace luisa::ocr