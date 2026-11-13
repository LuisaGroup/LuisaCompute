// OCR engine -- Gated Delta Net operator implementations (Phase 9).
// Device-free DSL factories + independent double-accumulation host oracles.
// The exact equations/layouts are declared in ops_gdn.h.
#include "ocr/ops/ops_gdn.h"

#include <cmath>

#include <luisa/dsl/sugar.h>

#include "ocr/ops/ops_common.h"
#include "ocr/runtime/shader_cache.h"

namespace luisa::ocr {

using namespace luisa::compute;

// GdnLayerConfig::validate() lives in runtime/hybrid_gdn_config.cpp (the shared
// host-safe geometry contract); this TU only implements the device kernels and
// the independent host oracles.

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

[[nodiscard]] GdnConvSiluKernel make_gdn_conv_silu_kernel() noexcept {
    GdnConvSiluKernel kernel = [](BufferFloat src, BufferFloat weight, BufferFloat bias,
                                  BufferFloat ring, BufferFloat out,
                                  Var<uint> tokens, Var<uint> channels, Var<uint> kernel_size,
                                  Var<uint> pos0) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_gdn_conv_silu");
        const auto c = ocr_linear_id();
        $if (c < channels & kernel_size >= 2u & tokens > 0u) {
            const auto rw = kernel_size - 1u;
            // Process tokens in causal order inside the thread. A ring slot
            // s = p % rw is overwritten with x_p only AFTER every output that
            // reads the older tap at slot s (positions p-rw .. p-1) has been
            // computed, so in-place ring updates are race-free per channel.
            $for (t, tokens) {
                const auto p = pos0 + t;
                Var<float> acc = bias.read(c);
                $for (j, kernel_size) {
                    // Signed absolute tap position; the host validator caps
                    // pos0+tokens at INT32_MAX so the signed cast is exact.
                    const auto signed_pos = cast<int>(p) - cast<int>(rw) + cast<int>(j);
                    Var<float> xv = def(0.0f);
                    // Left zero padding: negative absolute positions read 0.
                    // The branch guards the READ itself (never read-then-mask).
                    $if (signed_pos >= 0) {
                        const auto abs_pos = cast<uint>(signed_pos);
                        $if (abs_pos < pos0) {
                            xv = ring.read(c * rw + (abs_pos % rw));
                        } $else {
                            xv = src.read((abs_pos - pos0) * channels + c);
                        };
                    };
                    acc += weight.read(c * kernel_size + j) * xv;
                };
                out.write(t * channels + c, ocr_silu(acc));
                // Push the pre-activation value into the ring AFTER the
                // output was computed (see the causality note above).
                ring.write(c * rw + (p % rw), src.read(t * channels + c));
            };
        };
    };
    return kernel;
}

[[nodiscard]] GdnGatesKernel make_gdn_gates_kernel() noexcept {
    GdnGatesKernel kernel = [](BufferFloat a, BufferFloat b, BufferFloat a_log, BufferFloat dt_bias,
                               BufferFloat beta_out, BufferFloat g_out,
                               Var<uint> tokens, Var<uint> heads, Var<float> softplus_threshold) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_gdn_gates");
        const auto id = ocr_linear_id();
        $if (id < tokens * heads) {
            const auto h = id % heads;
            const auto av = a.read(id) + dt_bias.read(h);
            Var<float> sp = def(0.0f);
            $if (av > softplus_threshold) {
                sp = av;
            } $else {
                sp = log(1.0f + exp(av));
            };
            beta_out.write(id, ocr_sigmoid(b.read(id)));
            g_out.write(id, -exp(a_log.read(h)) * sp);
        };
    };
    return kernel;
}

[[nodiscard]] GdnRecurrentKernel make_gdn_recurrent_kernel() noexcept {
    GdnRecurrentKernel kernel = [](BufferFloat q, BufferFloat k, BufferFloat v,
                                   BufferFloat beta, BufferFloat g, BufferFloat state, BufferFloat out,
                                   Var<uint> tokens, Var<uint> heads, Var<uint> head_dim) noexcept {
        set_block_size(64u, 1u, 1u);
        set_name("ocr_gdn_recurrent");
        const auto h = ocr_linear_id();
        $if (h < heads & head_dim > 0u & tokens > 0u) {
            const auto d = head_dim;
            const auto sbase = h * d * d;
            $for (t, tokens) {
                const auto th = t * heads + h;
                const auto row = th * d;
                const auto alpha = exp(g.read(th));
                const auto bt = beta.read(th);
                // 1. Decay the whole state.
                $for (i, d * d) { state.write(sbase + i, state.read(sbase + i) * alpha); };
                // 2. err[v] = v[v] - (S^T k)[v]; S[k,v] layout.
                $for (vi, d) {
                    Var<float> sk = def(0.0f);
                    $for (ki, d) { sk += state.read(sbase + ki * d + vi) * k.read(row + ki); };
                    const auto err = v.read(row + vi) - sk;
                    // 3. Rank-1 delta update: S += beta * k err^T.
                    $for (ki, d) {
                        const auto idx = sbase + ki * d + vi;
                        state.write(idx, state.read(idx) + bt * k.read(row + ki) * err);
                    };
                };
                // 4. o[v] = (S^T q)[v].
                $for (vi, d) {
                    Var<float> o = def(0.0f);
                    $for (ki, d) { o += state.read(sbase + ki * d + vi) * q.read(row + ki); };
                    out.write(row + vi, o);
                };
            };
        };
    };
    return kernel;
}

[[nodiscard]] GdnChunkKernel make_gdn_chunk_kernel() noexcept {
    GdnChunkKernel kernel = [](BufferFloat q, BufferFloat k, BufferFloat v,
                               BufferFloat beta, BufferFloat g, BufferFloat state, BufferFloat out,
                               BufferFloat w_scratch, BufferFloat u_scratch, BufferFloat c_scratch,
                               Var<uint> tokens, Var<uint> heads, Var<uint> head_dim) noexcept {
        set_block_size(64u, 1u, 1u);
        set_name("ocr_gdn_chunk");
        const auto h = ocr_linear_id();
        $if (h < heads & head_dim > 0u & tokens > 0u) {
            const auto d = head_dim;
            const auto sbase = h * d * d;
            const auto wbase = h * tokens * d;
            const auto cbase = h * tokens;
            // 1. Cumulative log decay c_i = sum_{j<=i} g_j.
            Var<float> cum = def(0.0f);
            $for (i, tokens) {
                cum += g.read(i * heads + h);
                c_scratch.write(cbase + i, cum);
            };
            const auto c_total = cum;
            // 2. WY/UT forward substitution (serial inside the chunk --
            //    correctness-first; NOT a parallel scan):
            //    L[i,j] = beta_i <k_i,k_j> exp(c_i - c_j), j < i
            //    w_i = beta_i exp(c_i) k_i - sum_{j<i} L[i,j] w_j
            //    u_i = beta_i v_i           - sum_{j<i} L[i,j] u_j
            $for (i, tokens) {
                const auto ci = c_scratch.read(cbase + i);
                const auto bi = beta.read(i * heads + h);
                const auto irow = i * heads * d + h * d;
                $for (vi, d) {
                    Var<float> wv = bi * exp(ci) * k.read(irow + vi);
                    Var<float> uv = bi * v.read(irow + vi);
                    $for (j, i) {
                        const auto cj = c_scratch.read(cbase + j);
                        const auto jrow = j * heads * d + h * d;
                        Var<float> kk = def(0.0f);
                        $for (ki, d) { kk += k.read(irow + ki) * k.read(jrow + ki); };
                        const auto lij = bi * kk * exp(ci - cj);
                        wv -= lij * w_scratch.read(wbase + j * d + vi);
                        uv -= lij * u_scratch.read(wbase + j * d + vi);
                    };
                    w_scratch.write(wbase + i * d + vi, wv);
                    u_scratch.write(wbase + i * d + vi, uv);
                };
            };
            // 3. Pseudo values ue_i = u_i - S_prev^T w_i, AND the inter-chunk
            //    output part out_i = exp(c_i) * (S_prev^T q_i) -- computed
            //    HERE while the state still holds S_prev (before step 4).
            $for (i, tokens) {
                const auto ci = c_scratch.read(cbase + i);
                const auto irow = i * heads * d + h * d;
                const auto inter = exp(ci);
                $for (vi, d) {
                    Var<float> sw = def(0.0f);
                    Var<float> sq = def(0.0f);
                    $for (ki, d) {
                        const auto sv = state.read(sbase + ki * d + vi);
                        sw += sv * w_scratch.read(wbase + i * d + ki);
                        sq += sv * q.read(irow + ki);
                    };
                    u_scratch.write(wbase + i * d + vi, u_scratch.read(wbase + i * d + vi) - sw);
                    out.write(irow + vi, inter * sq);
                };
            };
            // 4. State transition: S = exp(c_T) S + sum_i exp(c_T-c_i) k_i ue_i^T.
            const auto decay = exp(c_total);
            $for (i, d * d) { state.write(sbase + i, state.read(sbase + i) * decay); };
            $for (i, tokens) {
                const auto ci = c_scratch.read(cbase + i);
                const auto coef = exp(c_total - ci);
                const auto irow = i * heads * d + h * d;
                $for (ki, d) {
                    const auto kk = coef * k.read(irow + ki);
                    $for (vi, d) {
                        const auto idx = sbase + ki * d + vi;
                        state.write(idx, state.read(idx) + kk * u_scratch.read(wbase + i * d + vi));
                    };
                };
            };
            // 5. Intra-chunk output accumulation:
            //    out_i += sum_{j<=i} <q_i,k_j> exp(c_i-c_j) ue_j
            //    (the inter-chunk part exp(c_i) S_prev^T q_i was written in
            //    step 3 while the state still held S_prev).
            $for (i, tokens) {
                const auto ci = c_scratch.read(cbase + i);
                const auto irow = i * heads * d + h * d;
                $for (vi, d) {
                    Var<float> o = out.read(irow + vi);// inter-chunk part (step 3b)
                    $for (j, i + 1u) {
                        const auto cj = c_scratch.read(cbase + j);
                        const auto jrow = j * heads * d + h * d;
                        Var<float> qk = def(0.0f);
                        $for (ki, d) { qk += q.read(irow + ki) * k.read(jrow + ki); };
                        o += qk * exp(ci - cj) * u_scratch.read(wbase + j * d + vi);
                    };
                    out.write(irow + vi, o);
                };
            };
        };
    };
    return kernel;
}

[[nodiscard]] GdnGatedNormKernel make_gdn_gated_norm_kernel() noexcept {
    GdnGatedNormKernel kernel = [](BufferFloat o, BufferFloat z, BufferFloat norm_w, BufferFloat out,
                                   Var<uint> tokens, Var<uint> heads, Var<uint> head_dim, Var<float> eps) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_gdn_gated_norm");
        const auto id = ocr_linear_id();
        $if (id < tokens * heads & head_dim > 0u) {
            const auto base = id * head_dim;
            Var<float> sumsq = def(0.0f);
            $for (i, head_dim) {
                const auto v = o.read(base + i);
                sumsq += v * v;
            };
            const auto inv = rsqrt(sumsq / cast<float>(head_dim) + eps);
            $for (i, head_dim) {
                out.write(base + i, o.read(base + i) * inv * norm_w.read(i) * ocr_silu(z.read(base + i)));
            };
        };
    };
    return kernel;
}

[[nodiscard]] ZeroFillKernel make_zero_fill_kernel() noexcept {
    ZeroFillKernel kernel = [](BufferFloat out, Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_zero_fill");
        const auto i = ocr_linear_id();
        $if (i < count) { out.write(i, 0.0f); };
    };
    return kernel;
}

[[nodiscard]] ElementwiseSigmoidKernel make_elementwise_sigmoid_kernel() noexcept {
    ElementwiseSigmoidKernel kernel = [](BufferFloat x, BufferFloat out, Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_elementwise_sigmoid");
        const auto i = ocr_linear_id();
        $if (i < count) { out.write(i, ocr_sigmoid(x.read(i))); };
    };
    return kernel;
}

// ---------------------------------------------------------------------------
// Dispatch helpers / validators
// ---------------------------------------------------------------------------

[[nodiscard]] Result<LaunchPlan1D> plan_gdn_conv_dispatch(uint64_t channels) {
    return plan_threads(channels, 256u);
}
[[nodiscard]] Result<LaunchPlan1D> plan_gdn_gates_dispatch(uint64_t tokens, uint64_t heads) {
    return plan_threads(tokens * heads, 256u);
}
[[nodiscard]] Result<LaunchPlan1D> plan_gdn_heads_dispatch(uint64_t heads) {
    return plan_threads(heads, 64u);
}
[[nodiscard]] Result<LaunchPlan1D> plan_gdn_gated_norm_dispatch(uint64_t tokens, uint64_t heads) {
    return plan_threads(tokens * heads, 256u);
}
[[nodiscard]] Result<LaunchPlan1D> plan_gdn_elementwise_dispatch(uint64_t count) {
    return plan_threads(count, 256u);
}

[[nodiscard]] Status validate_gdn_conv_inputs(uint64_t tokens, uint64_t channels,
                                              uint64_t kernel_size, uint64_t pos0) {
    if (channels == 0u || kernel_size < 2u) {
        return Status::error(StatusCode::invalid_argument, "GDN conv geometry must be nonzero (K >= 2)", "geometry");
    }
    if (tokens * channels > 0xFFFFFFFFull || channels * kernel_size > 0xFFFFFFFFull ||
        channels * (kernel_size - 1u) > 0xFFFFFFFFull) {
        return Status::error(StatusCode::invalid_argument, "GDN conv element count exceeds the uint32 envelope", "size");
    }
    if (pos0 + tokens > 0x7FFFFFFFull) {
        return Status::error(StatusCode::invalid_argument,
                             "GDN conv absolute position exceeds the int32 envelope (signed tap math)", "pos0");
    }
    return Status::success();
}

[[nodiscard]] Status validate_gdn_state_inputs(uint64_t tokens, uint64_t heads, uint64_t head_dim) {
    if (heads == 0u || head_dim == 0u) {
        return Status::error(StatusCode::invalid_argument, "GDN heads/head_dim must be nonzero", "geometry");
    }
    if (tokens * heads * head_dim > 0xFFFFFFFFull || heads * head_dim * head_dim > 0xFFFFFFFFull) {
        return Status::error(StatusCode::invalid_argument, "GDN state element count exceeds the uint32 envelope", "size");
    }
    return Status::success();
}

[[nodiscard]] luisa::string gdn_variant_key(luisa::string_view op) {
    return make_shader_key("gdn", op);
}

// ---------------------------------------------------------------------------
// Independent host CPU oracles (double accumulation).
// ---------------------------------------------------------------------------

namespace {
[[nodiscard]] double gdn_softplus(double x, double threshold) {
    return x > threshold ? x : std::log1p(std::exp(x));
}
}// namespace

[[nodiscard]] Status ref_gdn_conv_silu(luisa::span<const float> src, luisa::span<const float> weight,
                                       luisa::span<const float> bias, luisa::span<float> ring,
                                       uint32_t tokens, uint32_t channels, uint32_t kernel_size,
                                       uint32_t pos0, luisa::vector<float> &out) {
    if (channels == 0u || kernel_size < 2u) {
        return Status::error(StatusCode::invalid_argument, "GDN conv geometry must be nonzero (K >= 2)", "geometry");
    }
    const uint64_t rw = kernel_size - 1u;
    if (src.size() != static_cast<uint64_t>(tokens) * channels ||
        weight.size() != static_cast<uint64_t>(channels) * kernel_size ||
        bias.size() != channels || ring.size() != static_cast<uint64_t>(channels) * rw) {
        return Status::error(StatusCode::invalid_argument, "GDN conv oracle input size mismatch", "size");
    }
    out.resize(static_cast<size_t>(tokens) * channels);
    for (uint32_t t = 0; t < tokens; t++) {
        const uint64_t p = static_cast<uint64_t>(pos0) + t;
        for (uint32_t c = 0; c < channels; c++) {
            double acc = bias[c];
            for (uint32_t j = 0; j < kernel_size; j++) {
                const int64_t abs_pos = static_cast<int64_t>(p) - static_cast<int64_t>(rw) + j;
                double xv = 0.0;
                if (abs_pos >= 0) {
                    if (abs_pos < pos0) {
                        xv = ring[static_cast<size_t>(c) * rw + static_cast<size_t>(abs_pos % static_cast<int64_t>(rw))];
                    } else {
                        xv = src[static_cast<size_t>(abs_pos - pos0) * channels + c];
                    }
                }
                acc += static_cast<double>(weight[static_cast<size_t>(c) * kernel_size + j]) * xv;
            }
            out[static_cast<size_t>(t) * channels + c] = static_cast<float>(acc / (1.0 + std::exp(-acc)));
        }
        // Ring push AFTER the token's outputs (causal order).
        for (uint32_t c = 0; c < channels; c++) {
            ring[static_cast<size_t>(c) * rw + static_cast<size_t>(p % rw)] = src[static_cast<size_t>(t) * channels + c];
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_gdn_gates(luisa::span<const float> a, luisa::span<const float> b,
                                   luisa::span<const float> a_log, luisa::span<const float> dt_bias,
                                   uint32_t tokens, uint32_t heads, float softplus_threshold,
                                   luisa::vector<float> &beta, luisa::vector<float> &g) {
    if (heads == 0u) {
        return Status::error(StatusCode::invalid_argument, "GDN heads must be nonzero", "heads");
    }
    const uint64_t n = static_cast<uint64_t>(tokens) * heads;
    if (a.size() != n || b.size() != n || a_log.size() != heads || dt_bias.size() != heads) {
        return Status::error(StatusCode::invalid_argument, "GDN gates oracle input size mismatch", "size");
    }
    beta.resize(static_cast<size_t>(n));
    g.resize(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; i++) {
        const uint32_t h = static_cast<uint32_t>(i % heads);
        const double sp = gdn_softplus(static_cast<double>(a[i]) + dt_bias[h], softplus_threshold);
        beta[static_cast<size_t>(i)] = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(b[i]))));
        g[static_cast<size_t>(i)] = static_cast<float>(-std::exp(static_cast<double>(a_log[h])) * sp);
    }
    return Status::success();
}

[[nodiscard]] Status ref_gdn_recurrent(luisa::span<const float> q, luisa::span<const float> k,
                                       luisa::span<const float> v, luisa::span<const float> beta,
                                       luisa::span<const float> g, luisa::span<float> state,
                                       uint32_t tokens, uint32_t heads, uint32_t head_dim,
                                       luisa::vector<float> &out,
                                       luisa::vector<float> *state_snapshots) {
    if (heads == 0u || head_dim == 0u) {
        return Status::error(StatusCode::invalid_argument, "GDN heads/head_dim must be nonzero", "geometry");
    }
    const uint64_t n = static_cast<uint64_t>(tokens) * heads * head_dim;
    const uint64_t sn = static_cast<uint64_t>(heads) * head_dim * head_dim;
    if (q.size() != n || k.size() != n || v.size() != n || state.size() != sn ||
        beta.size() != static_cast<uint64_t>(tokens) * heads || g.size() != static_cast<uint64_t>(tokens) * heads) {
        return Status::error(StatusCode::invalid_argument, "GDN recurrent oracle input size mismatch", "size");
    }
    out.resize(static_cast<size_t>(n));
    if (state_snapshots != nullptr) {
        state_snapshots->resize(static_cast<size_t>(sn) * (tokens + 1u));
        std::copy(state.begin(), state.end(), state_snapshots->begin());
    }
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < heads; h++) {
            const uint64_t th = static_cast<uint64_t>(t) * heads + h;
            const uint64_t row = th * head_dim;
            const uint64_t sb = static_cast<uint64_t>(h) * head_dim * head_dim;
            const double alpha = std::exp(static_cast<double>(g[th]));
            const double bt = beta[th];
            for (uint64_t i = 0; i < static_cast<uint64_t>(head_dim) * head_dim; i++) {
                state[sb + i] = static_cast<float>(static_cast<double>(state[sb + i]) * alpha);
            }
            for (uint32_t vi = 0; vi < head_dim; vi++) {
                double sk = 0.0;
                for (uint32_t ki = 0; ki < head_dim; ki++) {
                    sk += static_cast<double>(state[sb + static_cast<uint64_t>(ki) * head_dim + vi]) * k[row + ki];
                }
                const double err = static_cast<double>(v[row + vi]) - sk;
                for (uint32_t ki = 0; ki < head_dim; ki++) {
                    const uint64_t idx = sb + static_cast<uint64_t>(ki) * head_dim + vi;
                    state[idx] = static_cast<float>(static_cast<double>(state[idx]) + bt * k[row + ki] * err);
                }
            }
            for (uint32_t vi = 0; vi < head_dim; vi++) {
                double o = 0.0;
                for (uint32_t ki = 0; ki < head_dim; ki++) {
                    o += static_cast<double>(state[sb + static_cast<uint64_t>(ki) * head_dim + vi]) * q[row + ki];
                }
                out[row + vi] = static_cast<float>(o);
            }
        }
        if (state_snapshots != nullptr) {
            std::copy(state.begin(), state.end(),
                      state_snapshots->begin() + static_cast<size_t>(sn) * (t + 1u));
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_gdn_chunk(luisa::span<const float> q, luisa::span<const float> k,
                                   luisa::span<const float> v, luisa::span<const float> beta,
                                   luisa::span<const float> g, luisa::span<float> state,
                                   uint32_t tokens, uint32_t heads, uint32_t head_dim,
                                   luisa::vector<float> &out) {
    if (heads == 0u || head_dim == 0u) {
        return Status::error(StatusCode::invalid_argument, "GDN heads/head_dim must be nonzero", "geometry");
    }
    const uint64_t n = static_cast<uint64_t>(tokens) * heads * head_dim;
    const uint64_t sn = static_cast<uint64_t>(heads) * head_dim * head_dim;
    if (q.size() != n || k.size() != n || v.size() != n || state.size() != sn ||
        beta.size() != static_cast<uint64_t>(tokens) * heads || g.size() != static_cast<uint64_t>(tokens) * heads) {
        return Status::error(StatusCode::invalid_argument, "GDN chunk oracle input size mismatch", "size");
    }
    out.resize(static_cast<size_t>(n));
    const uint32_t T = tokens, d = head_dim;
    // Full per-head L matrix + generic forward substitution (deliberately a
    // DIFFERENT code path from the device's fused loops).
    luisa::vector<double> L(static_cast<size_t>(T) * T);
    luisa::vector<double> cum(T);
    luisa::vector<double> w(static_cast<size_t>(T) * d), u(static_cast<size_t>(T) * d), ue(static_cast<size_t>(T) * d);
    for (uint32_t h = 0; h < heads; h++) {
        const uint64_t sb = static_cast<uint64_t>(h) * d * d;
        double c = 0.0;
        for (uint32_t i = 0; i < T; i++) {
            c += g[static_cast<uint64_t>(i) * heads + h];
            cum[i] = c;
        }
        const double c_total = c;
        for (uint32_t i = 0; i < T; i++) {
            for (uint32_t j = 0; j < T; j++) { L[static_cast<size_t>(i) * T + j] = 0.0; }
        }
        for (uint32_t i = 0; i < T; i++) {
            const uint64_t irow = (static_cast<uint64_t>(i) * heads + h) * d;
            const double bi = beta[static_cast<uint64_t>(i) * heads + h];
            for (uint32_t j = 0; j < i; j++) {
                const uint64_t jrow = (static_cast<uint64_t>(j) * heads + h) * d;
                double kk = 0.0;
                for (uint32_t ki = 0; ki < d; ki++) { kk += static_cast<double>(k[irow + ki]) * k[jrow + ki]; }
                L[static_cast<size_t>(i) * T + j] = bi * kk * std::exp(cum[i] - cum[j]);
            }
        }
        for (uint32_t i = 0; i < T; i++) {
            const uint64_t irow = (static_cast<uint64_t>(i) * heads + h) * d;
            const double bi = beta[static_cast<uint64_t>(i) * heads + h];
            for (uint32_t vi = 0; vi < d; vi++) {
                double wv = bi * std::exp(cum[i]) * k[irow + vi];
                double uv = bi * v[irow + vi];
                for (uint32_t j = 0; j < i; j++) {
                    const double lij = L[static_cast<size_t>(i) * T + j];
                    wv -= lij * w[static_cast<size_t>(j) * d + vi];
                    uv -= lij * u[static_cast<size_t>(j) * d + vi];
                }
                w[static_cast<size_t>(i) * d + vi] = wv;
                u[static_cast<size_t>(i) * d + vi] = uv;
            }
        }
        for (uint32_t i = 0; i < T; i++) {
            for (uint32_t vi = 0; vi < d; vi++) {
                double sw = 0.0;
                for (uint32_t ki = 0; ki < d; ki++) {
                    sw += static_cast<double>(state[sb + static_cast<uint64_t>(ki) * d + vi]) * w[static_cast<size_t>(i) * d + ki];
                }
                ue[static_cast<size_t>(i) * d + vi] = u[static_cast<size_t>(i) * d + vi] - sw;
            }
        }
        // Outputs use the PRE-transition state.
        for (uint32_t i = 0; i < T; i++) {
            const uint64_t irow = (static_cast<uint64_t>(i) * heads + h) * d;
            for (uint32_t vi = 0; vi < d; vi++) {
                double o = 0.0;
                for (uint32_t ki = 0; ki < d; ki++) {
                    o += static_cast<double>(state[sb + static_cast<uint64_t>(ki) * d + vi]) * q[irow + ki];
                }
                o *= std::exp(cum[i]);
                for (uint32_t j = 0; j <= i; j++) {
                    const uint64_t jrow = (static_cast<uint64_t>(j) * heads + h) * d;
                    double qk = 0.0;
                    for (uint32_t ki = 0; ki < d; ki++) { qk += static_cast<double>(q[irow + ki]) * k[jrow + ki]; }
                    o += qk * std::exp(cum[i] - cum[j]) * ue[static_cast<size_t>(j) * d + vi];
                }
                out[irow + vi] = static_cast<float>(o);
            }
        }
        // State transition.
        const double decay = std::exp(c_total);
        for (uint64_t i = 0; i < static_cast<uint64_t>(d) * d; i++) {
            state[sb + i] = static_cast<float>(static_cast<double>(state[sb + i]) * decay);
        }
        for (uint32_t i = 0; i < T; i++) {
            const uint64_t irow = (static_cast<uint64_t>(i) * heads + h) * d;
            const double coef = std::exp(c_total - cum[i]);
            for (uint32_t ki = 0; ki < d; ki++) {
                const double kk = coef * k[irow + ki];
                for (uint32_t vi = 0; vi < d; vi++) {
                    const uint64_t idx = sb + static_cast<uint64_t>(ki) * d + vi;
                    state[idx] = static_cast<float>(static_cast<double>(state[idx]) + kk * ue[static_cast<size_t>(i) * d + vi]);
                }
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_gdn_gated_norm(luisa::span<const float> o, luisa::span<const float> z,
                                        luisa::span<const float> norm_w, uint32_t tokens,
                                        uint32_t heads, uint32_t head_dim, float eps,
                                        luisa::vector<float> &out) {
    if (heads == 0u || head_dim == 0u) {
        return Status::error(StatusCode::invalid_argument, "GDN heads/head_dim must be nonzero", "geometry");
    }
    const uint64_t n = static_cast<uint64_t>(tokens) * heads * head_dim;
    if (o.size() != n || z.size() != n || norm_w.size() != head_dim) {
        return Status::error(StatusCode::invalid_argument, "GDN gated-norm oracle input size mismatch", "size");
    }
    out.resize(static_cast<size_t>(n));
    const uint64_t rows = static_cast<uint64_t>(tokens) * heads;
    for (uint64_t r = 0; r < rows; r++) {
        const uint64_t base = r * head_dim;
        double sumsq = 0.0;
        for (uint32_t i = 0; i < head_dim; i++) {
            sumsq += static_cast<double>(o[base + i]) * o[base + i];
        }
        const double inv = 1.0 / std::sqrt(sumsq / head_dim + eps);
        for (uint32_t i = 0; i < head_dim; i++) {
            const double zv = z[base + i];
            const double silu = zv / (1.0 + std::exp(-zv));
            out[base + i] = static_cast<float>(o[base + i] * inv * norm_w[i] * silu);
        }
    }
    return Status::success();
}

}// namespace luisa::ocr
