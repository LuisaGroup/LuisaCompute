// bge-m3 embedding engine -- cooperative-vector (coop) matvec projection
// kernels (optimization plan tensor_optimize.md §5: C0 probe + C2 K-slab x
// N-tile matvec). See kernels_coop.h for the host contract and the algorithm
// sketch (plan §5.3 pseudocode, reproduced inside the kernel emitter below).
//
// Emission rules honored here (plan §1.2/§1.3, §5.3; coop_api.md §3/§10):
// - ONLY supported cooperative CallOps are emitted: COOPERATIVE_MUL,
//   COOPERATIVE_MUL_ADD, COOPERATIVE_VECTOR_LOAD, COOPERATIVE_VECTOR_STORE
//   (plus the per-element scalar ACCESS CoopVector::operator[] lowers to).
//   The element-wise coop CallOps (COOPERATIVE_VECTOR_ADD/_SUB/_MUL/...) are
//   "future" ops: no DSL wrapper exists and the backends reject them
//   (spirv_codegen/dialect.cpp "cooperative-vector element-wise operations
//   are not implemented"). K-slab joining and the epilogue are therefore
//   pure per-element scalar DSL math in a fixed, deterministic order.
// - NO COOPERATIVE_VECTOR_ACCUMULATE to join K slabs (forbidden -- plan
//   §1.3/§5.3; it is a global-memory RMW whose per-lane order is not fixed).
// - fp32 only (CoopRefVecType::FLOAT32 everywhere; the CoopRefVecType enum
//   has no BF16 member -- coop_api.md §1; packed half/bf16 must never reach
//   these entry points, asserted at the boundary the host can see).
// - K % KSLAB == 0 and N % NTILE == 0 are caller guarantees (plan §5.3): the
//   emitter derives the slab/tile trip counts as plain host-side constants
//   when they are static and guards the runtime trip counts with the exact
//   divisibility device_assert below; ragged K/N must fall back to the
//   regular GEMM on the host, never here. Ragged M IS handled (per-token
//   guard only -- M is the only unbounded runtime dimension).
//
// Layout note (THE EXPERIMENT, plan §5.1/C0): the DSL never exposes a layout
// choice; codegen pins every matrix operand to MUL_OPTIMAL (DX) /
// InferencingOptimalNV (VK) with stride 0 (coop_api.md §8). We address the
// canonical row-major [K,N] fp32 weights with CoopMatrixRef(KSLAB, NTILE) at
// byte offset (s*KSLAB*N + n0)*4. Whether the optimal layouts honor that
// (stride-N row addressing for NTILE < N) is exactly what the C0 probe
// validates -- nothing in this file decides it.
#include "embedding/embed/kernels/kernels_coop.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl.h>
#include <luisa/dsl/sugar.h>

namespace luisa::embed {

namespace {

// The DSL builtins (set_block_size, thread_x, ite, ...) are found by ADL on
// Var<>, but the bare calls below follow the engine convention of a
// using-directive in the kernel-building scope (same as kernels_gemm.cpp).
using namespace luisa::compute;// NOLINT(misc-header-include-cycle)

// ERF-GELU, exactly the bge-m3/ONNX form 0.5 * x * (1 + erf(x / sqrt(2))).
// Copied verbatim from kernels_gemm.cpp (the DSL exposes no native erf
// instruction, so this is the Abramowitz & Stegun 7.1.26 rational
// approximation -- max abs error 1.5e-7 -- composed only from DSL
// abs/exp/ite so every backend computes the same value).
[[nodiscard]] Var<float> dsl_gelu_erf(Var<float> x) noexcept {
    const Var u = x * 0.70710678118654752440f;// x / sqrt(2)
    const Var z = abs(u);
    const Var one = def(1.0f);
    const Var t = one / (one + 0.3275911f * z);
    // Horner form of a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5.
    const Var poly = t * (0.254829592f +
                          t * (-0.284496736f +
                               t * (1.421413741f +
                                    t * (-1.453152027f + t * 1.061405429f))));
    const Var erf_u = one - poly * compute::exp(-(z * z));
    return 0.5f * x * (one + compute::ite(u >= 0.0f, erf_u, -erf_u));
}

// Flattened block index (identical idiom to kernels_gemm/kernels_embed/
// kernels_norm -- dispatch counts TOTAL THREADS, lc_optimize trap #1).
[[nodiscard]] Var<uint> coop_flattened_block_id() noexcept {
    const Var bs = compute::block_size().x;
    return compute::dispatch_id().y * (compute::dispatch_size().x / bs) +
           compute::dispatch_id().x / bs;
}

}// namespace

//----------------------------------------------------------------------------------------------------------------------
// C0 probe kernel (plan §5.1).
//
// One block, ONE cooperative thread computes y[N] = M[K,N] @ x[K] + bias[N]
// for the fixed probe shape K=16, N=8 with a single cooperative_mat_mul_add
// (the fused-bias form -- the probe doubles as the mul_add + bias-position
  // check of plan §8.1's coop micro-test list). All byte offsets are the
  // literal 0 (host side; coop_api.md §9 worked matvec binds the same way).
  // The probe shape (single block, one thread, K=16, N=8, fp32 weights) is
  // the fp32-weights contract of the header carried by construction: every
  // CoopMatrixRef/CoopVectorRef here is CoopRefVecType::FLOAT32. The host
  // dispatches one block (block (1,1,1) per the set_block_size below) and the
  // probe test driver validates the result against an fp64 CPU reference --
  // that comparison is what validates the canonical-row-major layout.
  //----------------------------------------------------------------------------------------------------
CoopProbeKernel make_coop_probe_kernel() noexcept {
    using ByteBufVar = Var<ByteBuffer>;

    constexpr auto kK = kCoopProbeK;// input dim  (CoopMatrixRef n)
    constexpr auto kN = kCoopProbeN;// output dim (CoopMatrixRef m)

    CoopProbeKernel kernel =
        [](ByteBufVar x, ByteBufVar weight, ByteBufVar bias, ByteBufVar y) noexcept {
            // One warp per block (the DSL requires block % 32 == 0,
            // src/dsl/builtin.cpp); thread 0 alone performs the coop matvec.
            set_block_size(32u, 1u, 1u);
            set_name("embed_coop_probe_k16_n8");
            $if (thread_x() == 0u) {

            // M[16x8] (K x N) and the bias vector [8] (N), FLOAT32, offset 0.
            CoopMatrixRef matrix_offset{CoopRefVecType::FLOAT32, kK, kN};
            CoopVectorRef bias_offset{CoopRefVecType::FLOAT32, kN};
            // x[16] (K), FLOAT32, offset 0.
            CoopVectorRef x_offset{CoopRefVecType::FLOAT32, kK};
            // y[8] (N), FLOAT32, offset 0.
            CoopVectorRef y_offset{CoopRefVecType::FLOAT32, kN};

            matrix_offset.set_byte_offset(0u);
            bias_offset.set_byte_offset(0u);
            x_offset.set_byte_offset(0u);
            y_offset.set_byte_offset(0u);

            const auto xv = cooperative_vector_load<float>(x, x_offset);
            // y = M @ x + bias, fused (COOPERATIVE_MUL_ADD).
            const auto out = cooperative_mat_mul_add<float, float>(
                weight, matrix_offset, bias, bias_offset,
                Expr<CoopVector<float>>{xv});
            cooperative_vector_store(y, y_offset, Expr<CoopVector<float>>{out});
            };
        };
    return kernel;
}

namespace {

//----------------------------------------------------------------------------------------------------------------------
// C2 matvec emitter (plan §5.3).
//
// KSLAB / NTILE / TPB are emission-time template constants (plan §5.3:
// "Kslab/Ntile 是 emission-time 常量"); the loop trip counts over K/KSLAB
// and N/NTILE are RUNTIME expressions derived from the kernel scalars, with
// the divisibility asserted on device so an unsupported ragged shape trips
// loudly instead of silently reading past the matrix (plan §5.3: ragged K/N
// -> host must fall back to the regular GEMM; the assert here is the
// backstop, not the contract).
//
// Epilogue template parameter selects the post-bias math via `if constexpr`
// (only Bias / BiasGeluErf are instantiated -- the factory asserts the rest).
//----------------------------------------------------------------------------------------------------------------------
template<uint32_t KSLAB, uint32_t NTILE, uint32_t TPB, GemmEpilogue EP>
[[nodiscard]] CoopGemmKernel make_coop_gemm_kernel_impl(const char *vname) noexcept {
    static_assert(KSLAB > 0u && NTILE > 0u && TPB > 0u, "coop tile dims must be positive");
    static_assert(NTILE <= 64u, "coop accumulator NTILE stays within the swept small set");
    static_assert(KSLAB <= 256u, "coop KSLAB stays within the swept small set");
    static_assert(EP == GemmEpilogue::Bias || EP == GemmEpilogue::BiasGeluErf,
                  "coop matvec v1 emits Bias or BiasGeluErf only");

    using ByteBufVar = Var<ByteBuffer>;
    using UIntVar = Var<uint32_t>;

    CoopGemmKernel kernel =
        [vname](ByteBufVar x, ByteBufVar weight, ByteBufVar bias, ByteBufVar y,
                UIntVar M, UIntVar N, UIntVar K) noexcept {
            set_block_size(kCoopGemmBlock, 1u, 1u);
            set_name(vname);

            // Backstop for the caller contract (plan §5.3): K % KSLAB == 0 and
            // N % NTILE == 0. Ragged shapes must never dispatch here.
            device_assert((K % KSLAB) == 0u,
                          "coop matvec: K must be a multiple of KSLAB (ragged K falls back to GEMM)");
            device_assert((N % NTILE) == 0u,
                          "coop matvec: N must be a multiple of NTILE (ragged N falls back to GEMM)");

            const Var tid = thread_x();
            // Flattened block index (1D launch and host-flattened 2D both OK).
            const Var blk = coop_flattened_block_id();
            constexpr auto block = static_cast<uint32_t>(kCoopGemmBlock);
            // First token of this thread's TPB-token group.
            const Var token0 = blk * (block * TPB) + tid * TPB;

            // Runtime trip counts: K/KSLAB and N/NTILE. The divisibility is
            // asserted on device above, so integer division is exact.
            const Var slab_count = K / KSLAB;
            const Var tile_count = N / NTILE;

            // 64-bit-safe byte-offset helper: ((uint64)a * (uint64)b + c) * 4,
            // truncated to the uint32 the CoopVectorRef/CoopMatrixRef API
            // takes. The bge-m3 envelope (hidden 1024, FFN intermediate 4096,
            // <= 16384 padded rows per job, weights <= ~64 MiB) keeps every
            // product below 2^32 -- the truncation is exact (plan §5.3
            // "offset computed 64-bit, then cast to the API's 32-bit byte
            // offset"). `cast<uint64_t>` / `cast<uint32_t>` are the DSL static
            // casts (uint32 -> uint64 widening is lossless; the uint64 ->
            // uint32 narrowing is exact within the envelope above).
            auto byte_off = [](const Var<uint> &a, const Var<uint> &b,
                               const Var<uint> &c) noexcept -> Var<uint> {
                const Var<uint64_t> bytes =
                    (a.cast<uint64_t>() * b.cast<uint64_t>() + c.cast<uint64_t>()) * 4ull;
                return bytes.cast<uint32_t>();
            };

            // Per-token loop: TPB tokens per thread, C++-unrolled (plan §5.3's
            // 1/2/4-token experiment). Ragged M is handled here by the guard;
            // ragged K/N never reaches this kernel (assert above + host guard).
            for (auto j = 0u; j < TPB; j++) {
                const Var token = token0 + static_cast<uint32_t>(j);
                $if (token < M) {
                    // N-tile loop: one [KSLAB x NTILE] accumulator tile per
                    // iteration of the output row. x is re-loaded per n-tile
                    // (L1/L2 cached -- accepted v1 simplification; plan §5.3
                    // flags the weight/x reuse trade as a probe axis).
                    $for (n0, tile_count) {
                        // acc[NTILE] = 0 (C++-unrolled fixed-order init).
                        // CoopVector locals are function-scoped DSL registers:
                        // this re-zero is the per-iteration reset, exactly like
                        // the gemm register family's Var<float> acc (storage is
                        // fixed at emission time -- lc_optimize §10.5).
                        CoopVector<float> acc{NTILE};
                        for (auto i = 0u; i < NTILE; i++) {
                            acc[i] = def(0.0f);
                        }
                        // K-slab loop (runtime): the partial result register is
                        // created once (outside $for) and re-assigned per slab;
                        // inside the loop we emit ONE COOPERATIVE_MUL or, on the
                        // LAST slab, ONE COOPERATIVE_MUL_ADD that fuses +bias
                        // (bias read from the bias buffer at the n-tile base --
                        // plan §5.3 "bias via mul_add on last slab"). Branching
                        // the op itself is required: the bias must join the
                        // accumulation for free on the last slab, and an
                        // unconditional mul_add with a dummy bias buffer would
                        // need a $for-free way to say "no bias" on the earlier
                        // slabs that doesn't add bytes. The DSL $if records both
                        // arms once; at runtime the guard is slab-uniform.
                        CoopVector<float> partial{NTILE};
                        $for (s, slab_count) {
                            // x slab: token-major row token*K .. (token+1)*K,
                            // chunk starting at k = s*KSLAB.
                            CoopVectorRef x_ref{CoopRefVecType::FLOAT32, KSLAB};
                            x_ref.set_byte_offset(byte_off(token, K, s * KSLAB));
                            const auto xv = cooperative_vector_load<float>(x, x_ref);
                            // Weight slab tile: canonical row-major [K,N] with
                            // row stride N -- rows s*KSLAB..+KSLAB, columns
                            // n0*NTILE..+NTILE -> byte offset
                            // ((s*KSLAB)*N + n0*NTILE) * 4.
                            // (Whether the MUL_OPTIMAL/InferencingOptimal
                            // layout honors this stride-N window is THE probe;
                            // see the file header. n0 is the TILE INDEX, the
                            // column base is n0*NTILE.)
                            CoopMatrixRef m_ref{CoopRefVecType::FLOAT32, KSLAB, NTILE};
                            m_ref.set_byte_offset(byte_off(s * KSLAB, N, n0 * NTILE));

                            $if (s < slab_count - 1u) {
                                // Plain slab: partial = W_slab @ x_slab.
                                const auto r = cooperative_mat_mul<float, float>(
                                    weight, m_ref, Expr<CoopVector<float>>{xv});
                                for (auto i = 0u; i < NTILE; i++) {
                                    partial[i] = r[i];
                                }
                            }
                            $else {
                                // Last slab: fused +bias via mul_add. Bias
                                // vector ref: FLOAT32, NTILE, byte offset
                                // (n0*NTILE)*4 of bias[N].
                                CoopVectorRef b_ref{CoopRefVecType::FLOAT32, NTILE};
                                b_ref.set_byte_offset(n0 * NTILE * 4u);
                                const auto r = cooperative_mat_mul_add<float, float>(
                                    weight, m_ref, bias, b_ref,
                                    Expr<CoopVector<float>>{xv});
                                for (auto i = 0u; i < NTILE; i++) {
                                    partial[i] = r[i];
                                }
                            };

                            // Per-element LOCAL, fixed-order accumulate
                            // (NO cooperative_vector_accumulate -- plan §5.3).
                            for (auto i = 0u; i < NTILE; i++) {
                                acc[i] = acc[i] + partial[i];
                            }
                        };

                        // acc carries the fused +bias from the last slab's
                        // mul_add. Epilogue, then store the N-tile.
                        CoopVectorRef y_ref{CoopRefVecType::FLOAT32, NTILE};
                        y_ref.set_byte_offset(byte_off(token, N, n0 * NTILE));
                        if constexpr (EP == GemmEpilogue::Bias) {
                            cooperative_vector_store(y, y_ref, Expr<CoopVector<float>>{acc});
                        } else {
                            static_assert(EP == GemmEpilogue::BiasGeluErf, "guarded above");
                            // erf-GELU AFTER the bias (the bias is already in
                            // acc via the last-slab mul_add).
                            CoopVector<float> out{NTILE};
                            for (auto i = 0u; i < NTILE; i++) {
                                out[i] = dsl_gelu_erf(acc[i]);
                            }
                            cooperative_vector_store(y, y_ref, Expr<CoopVector<float>>{out});
                        }
                    };
                };
            }
        };
    return kernel;
}

// Build the templated kernel for a (runtime) epilogue flag. Bias and
// BiasGeluErf are the only coop-emitted epilogues (plan §1.3/§5.3).
template<uint32_t KSLAB, uint32_t NTILE, uint32_t TPB>
[[nodiscard]] CoopGemmKernel build_coop_gemm_ep(GemmEpilogue ep, const char *vname) noexcept {
    switch (ep) {
        case GemmEpilogue::Bias:
            return make_coop_gemm_kernel_impl<KSLAB, NTILE, TPB, GemmEpilogue::Bias>(vname);
        case GemmEpilogue::BiasGeluErf:
            return make_coop_gemm_kernel_impl<KSLAB, NTILE, TPB, GemmEpilogue::BiasGeluErf>(vname);
        default:
            break;
    }
    // Unreachable: the public factory asserts ep is Bias/BiasGeluErf before
    // calling here; Bias is the safe default for any stray value.
    return make_coop_gemm_kernel_impl<KSLAB, NTILE, TPB, GemmEpilogue::Bias>(vname);
}

using CoopGemmBuilder = CoopGemmKernel (*)(GemmEpilogue, const char *) noexcept;

struct CoopEntry {
    CoopGemmConfig cfg;
    CoopGemmBuilder builder;
};

// (name, kslab, ntile, tokens_per_block). The first variant is the shipped
// default (coop_gemm_variant(0)); plan §5.3's recommended first batch.
constexpr CoopEntry kCoopEntries[] = {
    {{"coop_k64_n16_t1", 64u, 16u, 1u}, &build_coop_gemm_ep<64u, 16u, 1u>},
    {{"coop_k64_n32_t1", 64u, 32u, 1u}, &build_coop_gemm_ep<64u, 32u, 1u>},
    {{"coop_k128_n32_t1", 128u, 32u, 1u}, &build_coop_gemm_ep<128u, 32u, 1u>},
    {{"coop_k64_n32_t2", 64u, 32u, 2u}, &build_coop_gemm_ep<64u, 32u, 2u>},
};

constexpr uint32_t kCoopEntryCount =
    static_cast<uint32_t>(sizeof(kCoopEntries) / sizeof(kCoopEntries[0]));

static_assert(kCoopEntryCount == 4u, "coop variant table must carry the 4 planned entries");

[[nodiscard]] int32_t coop_entry_by_name(luisa::string_view name) noexcept {
    for (auto i = 0u; i < kCoopEntryCount; i++) {
        if (name == luisa::string_view{kCoopEntries[i].cfg.name}) {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

[[nodiscard]] int32_t coop_entry_by_cfg(const CoopGemmConfig &cfg) noexcept {
    // Identity by (name, kslab, ntile, tpb): a table entry is exactly one of
    // the four compiled variants. -1 when `cfg` is not a known variant.
    if (const auto i = coop_entry_by_name(cfg.name); i >= 0) {
        const auto &e = kCoopEntries[static_cast<uint32_t>(i)];
        if (e.cfg.kslab == cfg.kslab && e.cfg.ntile == cfg.ntile &&
            e.cfg.tokens_per_block == cfg.tokens_per_block) {
            return i;
        }
    }
    return -1;
}

}// namespace

//----------------------------------------------------------------------------------------------------------------------
// Variant table accessors (mirror the gemm table style).
//----------------------------------------------------------------------------------------------------------------------
uint32_t coop_gemm_variant_count() noexcept {
    return kCoopEntryCount;
}

const CoopGemmConfig *coop_gemm_variant(uint32_t i) noexcept {
    return i < kCoopEntryCount ? &kCoopEntries[i].cfg : nullptr;
}

const CoopGemmConfig *coop_gemm_variant_by_name(luisa::string_view name) noexcept {
    const auto i = coop_entry_by_name(name);
    return i >= 0 ? &kCoopEntries[static_cast<uint32_t>(i)].cfg : nullptr;
}

//----------------------------------------------------------------------------------------------------------------------
// Factory: pick the table entry by `cfg`, validate the epilogue, build.
// KSLAB/NTILE/TPB are emission-time constants (one template instantiation per
// table row); `vname` is baked into the DSL kernel name for profiling and
// must encode variant+epilogue+format (plan §5.4), e.g.
// "embed_coop_gemm_bias_k64_n16_t1".
//----------------------------------------------------------------------------------------------------------------------
CoopGemmKernel make_coop_gemm_kernel(GemmEpilogue ep, const CoopGemmConfig &cfg,
                                     const char *vname) noexcept {
    return make_coop_gemm_kernel(ep, cfg, vname, GemmWeightFormat::Fp32);
}

CoopGemmKernel make_coop_gemm_kernel(GemmEpilogue ep, const CoopGemmConfig &cfg,
                                     const char *vname, GemmWeightFormat wf) noexcept {
    // Host contract enforced at the boundary (plan §1.3: packed half/bf16 must
    // NOT enter the coop path; only fp32 weights are valid here).
    LUISA_ASSERT(wf == GemmWeightFormat::Fp32,
                 "coop matvec requires fp32 weights; packed half/bf16 must not enter the coop path");
    // The coop v1 epilogue set is Bias / BiasGeluErf only (plan §1.3: erf-GELU
    // stays for FFN up; the header documents the supported epilogues).
    LUISA_ASSERT(ep == GemmEpilogue::Bias || ep == GemmEpilogue::BiasGeluErf,
                 "coop matvec v1 supports only the Bias / BiasGeluErf epilogues");
    const auto i = coop_entry_by_cfg(cfg);
    LUISA_ASSERT(i >= 0,
                 "coop matvec: config '{}' (kslab {}, ntile {}, tpb {}) is not in the variant table",
                 cfg.name, cfg.kslab, cfg.ntile, cfg.tokens_per_block);
    const auto &e = kCoopEntries[static_cast<uint32_t>(i)];
    return e.builder(ep, vname);
}

uint2 coop_gemm_dispatch_size_config(const CoopGemmConfig &cfg, uint32_t m) noexcept {
    // TOTAL THREADS (lc_optimize trap #1): ceil(tokens-per-block-thread-group)
    // blocks * 256 threads. One block covers 256*tpb tokens (tpb per thread).
    const auto per_block = kCoopGemmBlock * cfg.tokens_per_block;
    const auto blocks = (m + per_block - 1u) / per_block;
    return {blocks * kCoopGemmBlock, 1u};
}

}// namespace luisa::embed
