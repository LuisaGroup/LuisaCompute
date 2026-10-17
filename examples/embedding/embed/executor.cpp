// bge-m3 embedding engine -- graph executor implementation (plan §3.4/§4/§5).
// See executor.h for the ownership model, trace-hook semantics, and the
// B2 dispatch-contract notes.
//
// ONNX-op mapping of the recorded sequence (plan §4 table):
//   embed_lookup  -> Gather x2 + Add + embedding LayerNormalization (#1)
//   qkv GEMM      -> MatMul + Add(bias) with fused [1024,3072] weight (#3 Bias)
//   attention     -> #4 MatMul(QK^T)+Mul(scale)+mask Add, #5 Softmax, #6
//                    MatMul(PV)+Transpose -- or the M4 fused flash kernel
//   attn.out GEMM -> MatMul + Add (#3 Bias) ; residual Add (#7) ; LN1 (#2)
//   ffn.up GEMM   -> MatMul + Add + Erf-chain (BiasGeluErf epilogue, #3)
//   ffn.down GEMM -> MatMul + Add (#3 Bias) ; residual Add (#7) ; LN2 (#2)
//   slice_cls (#8) + l2_normalize (#9) -> dense pooling path
//   sparse_head / colbert_proj (#10)   -> optional retrieval heads
#include "executor.h"

#include <algorithm>
#include <cmath>

#include <luisa/core/logging.h>

#include "kernels/kernels_attention.h"
#include "kernels/kernels_embed.h"
#include "kernels/kernels_gemm.h"
#include "kernels/kernels_heads.h"
#include "kernels/kernels_norm.h"

namespace luisa::embed {

using compute::Buffer;
using compute::BufferView;
using compute::CommandList;
using compute::Device;
using compute::Shader;
using compute::ShaderOption;

uint32_t embed_bucket_for(uint32_t seq) noexcept {
    static constexpr uint32_t kBuckets[] = {128u, 512u, 2048u, 8192u};// plan §3.4
    for(const auto b : kBuckets) {
        if(seq <= b) {
            return b;
        }
    }
    return kBuckets[3u];
}

namespace {

const Tensor &m_require(const Package &p, const luisa::string &name) {
    const auto *t = p.find(name);
    if(t == nullptr) [[unlikely]] {
        LUISA_ERROR("embedding WeightStore: package is missing tensor \"{}\".", name);
    }
    return *t;
}

}// namespace

// B2's launch helpers are DEFINED in kernels_attention.cpp but not declared in
// the parent-owned kernels_attention.h; re-declared verbatim here (namespace
// luisa::embed, EXTERNAL linkage) so the executor dispatch matches the kernel geometry exactly
// (see executor.h DISPATCH CONTRACT).
uint2 attention_scores_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads) noexcept;
uint2 attention_pv_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim) noexcept;
uint32_t softmax_row_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads) noexcept;
uint2 flash_attention_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads) noexcept;
uint32_t flash_attention_scratch_floats(uint32_t batch, uint32_t seq_len, uint32_t num_heads) noexcept;


WeightStore::WeightStore(Device &device, const Package &package)
    : m_device{device} {

    const auto L = package.num_layers();
    m_hidden = package.hidden_size();
    m_heads = package.num_heads();
    m_head_dim = m_hidden / m_heads;
    m_inter = package.intermediate_size();
    if(m_hidden == 0u || m_heads == 0u || m_hidden % m_heads != 0u || L == 0u) [[unlikely]] {
        LUISA_ERROR("embedding WeightStore: invalid package hyper-parameters (layers={}, H={}, heads={}).",
                    L, m_hidden, m_heads);
    }

    const auto m_take = [&](const luisa::string &name, Buffer<float> &dst) {
        const auto &t = m_require(package, name);
        const auto n = static_cast<size_t>(t.element_count());
        dst = m_device.create_buffer<float>(n);
        m_uploads.push_back(UploadItem{&dst, t.data, n});
    };

    m_take("emb.word", m_word);
    m_take("emb.pos", m_pos);
    m_take("emb.ln.gamma", m_emb_ln_gamma);
    m_take("emb.ln.beta", m_emb_ln_beta);

    m_layers.reserve(L);
    for(uint32_t i = 0; i < L; i++) {
        m_layers.emplace_back();
        auto &lw = m_layers.back();// fill IN PLACE: UploadItem stores pointers to
        // these member buffers, so they must never move (m_layers is reserved
        // above and append-only afterwards).
        m_take(luisa::format("layer{}.qkv", i), lw.qkv);
        m_take(luisa::format("layer{}.qkv.bias", i), lw.qkv_bias);
        m_take(luisa::format("layer{}.attn.out", i), lw.attn_out);
        m_take(luisa::format("layer{}.attn.out.bias", i), lw.attn_out_bias);
        m_take(luisa::format("layer{}.ln1.gamma", i), lw.ln1_gamma);
        m_take(luisa::format("layer{}.ln1.beta", i), lw.ln1_beta);
        m_take(luisa::format("layer{}.ffn.up", i), lw.ffn_up);
        m_take(luisa::format("layer{}.ffn.up.bias", i), lw.ffn_up_bias);
        m_take(luisa::format("layer{}.ffn.down", i), lw.ffn_down);
        m_take(luisa::format("layer{}.ffn.down.bias", i), lw.ffn_down_bias);
        m_take(luisa::format("layer{}.ln2.gamma", i), lw.ln2_gamma);
        m_take(luisa::format("layer{}.ln2.beta", i), lw.ln2_beta);
    }

    m_take("head.sparse.w", m_sparse_w);// [H,1] == plain [H] dot vector
    m_sparse_b = m_require(package, "head.sparse.b").data[0];
    m_take("head.colbert.w", m_colbert_w);
    m_take("head.colbert.b", m_colbert_b);
}

void WeightStore::record_upload(CommandList &cl) const {
    for(const auto &u : m_uploads) {
        cl << u.buffer->view().copy_from(luisa::span{u.src, u.count});
    }
}

Executor::Executor(Device &device, const WeightStore &weights, compute::Stream &owner_stream) noexcept
    : m_device{device}, m_owner_stream{owner_stream}, m_weights{weights} {

    const auto compile = [&](auto &&kernel, const char *name) {
        ShaderOption opt;
        opt.name = name;
        return m_device.compile(std::forward<decltype(kernel)>(kernel), opt);
    };

    m_embed = compile(make_embed_lookup_kernel(), "emb_lookup");
    m_gemm_bias = compile(make_gemm_kernel(GemmEpilogue::Bias), "gemm_bias");
    m_gemm_gelu = compile(make_gemm_kernel(GemmEpilogue::BiasGeluErf), "gemm_bias_gelu_erf");
    m_residual = compile(make_residual_add_kernel(), "residual_add");
    m_layernorm = compile(make_layernorm_kernel(), "layernorm");
    m_l2norm = compile(make_l2_normalize_kernel(), "l2_normalize");
    m_slice_cls = compile(make_slice_cls_kernel(), "slice_cls");
    m_sparse_head = compile(make_sparse_head_kernel(), "sparse_head");
    m_colbert_proj = compile(make_colbert_proj_kernel(), "colbert_proj");
    m_attn_scores = compile(make_attention_scores_kernel(), "attn_scores");
    m_attn_softmax = compile(make_softmax_row_kernel(), "softmax_row");
    m_attn_pv = compile(make_attention_pv_kernel(), "attn_pv");
    // m_flash stays unset: lazily compiled on the first S > 1024 encode (M4).
}

void Executor::m_ensure_pool(uint32_t rows, uint32_t batch, uint32_t seq) {

    const uint32_t H = m_weights.hidden_size();
    const uint32_t I = m_weights.intermediate_size();

    if(rows > m_pool_rows) {
        // Grow-only reallocation (rare, amortized): flush the owning worker
        // stream before the replaced buffers are destroyed, so the vk
        // validation layer retires its resource-usage bookkeeping
        // ("destroyed while stream is still using it" abort under
        // LUISA_ENABLE_VALIDATION). The scheduler only dispatches to an IDLE
        // worker, so this sync is a zero-wait bookkeeping flush, never a
        // completion stall; the non-growth fast path never synchronizes.
        m_pool_rows = rows;
        const auto rf = static_cast<size_t>(rows);
        m_ids_dev = m_device.create_buffer<uint32_t>(rf);
        m_mask_dev = m_device.create_buffer<float>(rf);
        m_h0 = m_device.create_buffer<float>(rf * H);
        m_h1 = m_device.create_buffer<float>(rf * H);
        m_qkv = m_device.create_buffer<float>(rf * 3u * H);
        m_att = m_device.create_buffer<float>(rf * H);
        m_tmp1 = m_device.create_buffer<float>(rf * H);
        m_tmp2 = m_device.create_buffer<float>(rf * H);
        m_ffn = m_device.create_buffer<float>(rf * I);
        m_cls = m_device.create_buffer<float>(H);
        m_dense_dev = m_device.create_buffer<float>(rf * H);// B*H <= B*seq*H (seq >= 1)
        m_sparse_dev = m_device.create_buffer<float>(rf);
        m_colbert_dev = m_device.create_buffer<float>(rf * H);
    }

    if(seq <= kNaiveAttentionMaxSeq) {
        // logits workspace [B*heads, S, S] fp32 (plan §5: <= 64 MiB at B=1,S=1024)
        const auto need = static_cast<size_t>(batch) * m_weights.num_heads() * seq * seq;
        if(need > m_logits_floats) {
            m_owner_stream.synchronize();// retire-usage flush (see above)
            m_logits_floats = static_cast<uint32_t>(need);
            m_logits = m_device.create_buffer<float>(need);
        }
    } else {
        // flash scratch: DUMMY per the B2 contract (1 float, never touched)
        if(!m_flash_scratch) {
            const auto need = static_cast<size_t>(flash_attention_scratch_floats(batch, seq, m_weights.num_heads()));
            m_flash_scratch = m_device.create_buffer<float>(need);
        }
    }
}

void Executor::m_ensure_trace_slots(uint32_t floats_per_slot) {
    if(floats_per_slot <= m_trace_floats) {
        return;
    }
    m_owner_stream.synchronize();// same retire-usage note as m_ensure_pool
    m_trace_floats = floats_per_slot;
    m_owner_stream.synchronize();// retire-usage flush before slot replacement
    for(uint32_t k = 0; k < kTraceStates; k++) {
        m_trace_slots[k] = m_device.create_buffer<float>(floats_per_slot);
        m_trace_slots[k].set_name(luisa::format("embed_trace_slot_{}", k));
    }
}

void Executor::record_encode(CommandList &cl, const EncodeJob &job) {

    const auto B = job.batch;
    const auto S = job.seq;
    const auto H = m_weights.hidden_size();
    const auto heads = m_weights.num_heads();
    const auto hd = m_weights.head_dim();
    const auto I = m_weights.intermediate_size();
    const auto L = m_weights.num_layers();
    const uint32_t rows = B * S;
    const auto rf = static_cast<size_t>(rows);

    if(S == 0u || B == 0u || S + kEmbedPosOffset > 8194u) [[unlikely]] {
        LUISA_ERROR("executor: invalid job shape B={} S={} (position table ceiling is 8194).", B, S);
    }
    if(job.ids.size() != rf || job.mask.size() != rf || job.dense_out.size() != static_cast<size_t>(B) * H) [[unlikely]] {
        LUISA_ERROR("executor: job staging sizes do not match B*S (ids={}, mask={}, dense={}).",
                    job.ids.size(), job.mask.size(), job.dense_out.size());
    }

    m_ensure_pool(rows, B, S);

    const bool tracing = static_cast<bool>(m_trace_cb);
    if(tracing) {
        m_ensure_trace_slots(static_cast<uint32_t>(rf * H));
    }
    m_traced_states = 0u;
    m_trace_rows = rows;
    m_trace_cols = H;

    const auto trace = [&](uint32_t layer, BufferView<float> src) {
        if(tracing) {
            const auto n = rf * H;
            cl << m_trace_slots[layer].view(0, n).copy_from(src.subview(0, n));
            m_traced_states = layer + 1u;
        }
    };

    //--- staging upload -------------------------------------------------------
    cl << m_ids_dev.view(0, rf).copy_from(job.ids)
       << m_mask_dev.view(0, rf).copy_from(job.mask);

    //--- embedding block (kernel #1; emb.word already carries the folded
    // token-type row 0; XLM-R position offset +2; padded rows keep the uniform
    // offset and are masked/dropped downstream) ------------------------------
    // ADAPTATION (documented deviation, B1): the shipped embed_lookup kernel
    // reads pos_emb at row (s) and IGNORES its pos_offset argument (proved by
    // direct-call diff vs. the host reference in test_embed_executor's BGEM3_DIAG
    // path; Phase-A only ever passed 0). The executor compensates by binding a
    // view of emb.pos shifted down by kEmbedPosOffset rows, and passes
    // pos_offset = 0. This is correct both pre- and post-fix of kernel #1: a
    // fixed kernel adding a 0 offset is a no-op. The +2 shift reproduces
    // XLM-R's position convention (first real token = row 2; pack_model.cpp
    // header) so traced layer 0 matches golden_layers.bin to 4.8e-7.
    const auto pos_rows = m_weights.pos_emb().size();// full table capacity
    cl << m_embed(m_weights.word_emb().view(),
                  m_weights.pos_emb().view(static_cast<size_t>(kEmbedPosOffset) * H,
                                           (pos_rows / H - kEmbedPosOffset) * H),
                  m_ids_dev.view(0, rf),
                  m_weights.emb_ln_gamma().view(), m_weights.emb_ln_beta().view(),
                  m_h0.view(0, rf * H),
                  S, H, kEmbedLayerNormEps, 0u)
             .dispatch(static_cast<uint>(rf * kEmbedBlock));
    trace(0u, m_h0.view(0, rf * H));

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const auto scores_disp = attention_scores_dispatch(B, S, heads);
    const auto pv_disp = attention_pv_dispatch(B, S, heads, hd);

    //--- 24 encoder layers ----------------------------------------------------
    for(uint32_t i = 0; i < L; i++) {
        const auto &lw = m_weights.layer(i);

        // qkv projection: [rows,H] @ [H,3H] + bias -> qkv
        cl << m_gemm_bias(m_h0.view(0, rf * H), lw.qkv.view(),
                          m_qkv.view(0, rf * 3u * H), lw.qkv_bias.view(),
                          rows, 3u * H, H)
                 .dispatch(gemm_dispatch_size(rows, 3u * H));

        // attention (naive trio for S <= 1024, fused flash above)
        if(S <= kNaiveAttentionMaxSeq) {
            const auto lf = static_cast<size_t>(B) * heads * S * S;
            cl << m_attn_scores(m_qkv.view(0, rf * 3u * H), m_mask_dev.view(0, rf),
                                m_logits.view(0, lf),
                                S, heads, hd, scale, kAttentionNegLarge)
                     .dispatch(scores_disp);
            const uint32_t sm_rows = B * heads * S;
            const uint32_t sm_chunk = std::max(4u, (65535u / heads) & ~3u);// mult of 4
            for(uint32_t r0 = 0u; r0 < sm_rows; r0 += sm_chunk) {
                const uint32_t n = std::min(sm_chunk, sm_rows - r0);
                cl << m_attn_softmax(m_logits.view(static_cast<size_t>(r0) * S,
                                                   static_cast<size_t>(n) * S),
                                     S)
                         .dispatch(n * kAttentionBlock);
            }
            cl << m_attn_pv(m_qkv.view(0, rf * 3u * H), m_logits.view(0, lf),
                            m_att.view(0, rf * H), S, heads, hd)
                     .dispatch(pv_disp);
        } else {
            if(!m_flash.has_value()) [[unlikely]] {
                ShaderOption opt;
                opt.name = "flash_attention";
                m_flash.emplace(m_device.compile(make_flash_attention_kernel(), opt));
            }
            const auto sf = static_cast<size_t>(flash_attention_scratch_floats(B, S, heads));
            cl << (*m_flash)(m_qkv.view(0, rf * 3u * H), m_mask_dev.view(0, rf),
                             m_att.view(0, rf * H), m_flash_scratch.view(0, sf),
                             B, S, heads, hd, scale, kAttentionNegLarge)
                     .dispatch(flash_attention_dispatch(B, S, heads));
        }

        // out projection + residual + LN1
        cl << m_gemm_bias(m_att.view(0, rf * H), lw.attn_out.view(),
                          m_tmp1.view(0, rf * H), lw.attn_out_bias.view(),
                          rows, H, H)
                 .dispatch(gemm_dispatch_size(rows, H))
           << m_residual(m_h0.view(0, rf * H), m_tmp1.view(0, rf * H),
                         m_tmp2.view(0, rf * H), rf * H)
                 .dispatch(static_cast<uint>(rf * H))
           << m_layernorm(m_tmp2.view(0, rf * H), m_h1.view(0, rf * H),
                          lw.ln1_gamma.view(), lw.ln1_beta.view(), H, kEmbedLayerNormEps)
                 .dispatch(static_cast<uint>(rf * kNormBlock));

        // FFN (erf-GELU fused in the up-projection epilogue) + residual + LN2.
        // The result returns to m_h0 in place: m_h0's layer input is fully
        // consumed by the first residual_add before LN2 writes back.
        cl << m_gemm_gelu(m_h1.view(0, rf * H), lw.ffn_up.view(),
                          m_ffn.view(0, rf * I), lw.ffn_up_bias.view(),
                          rows, I, H)
                 .dispatch(gemm_dispatch_size(rows, I))
           << m_gemm_bias(m_ffn.view(0, rf * I), lw.ffn_down.view(),
                          m_tmp1.view(0, rf * H), lw.ffn_down_bias.view(),
                          rows, H, I)
                 .dispatch(gemm_dispatch_size(rows, H))
           << m_residual(m_h1.view(0, rf * H), m_tmp1.view(0, rf * H),
                         m_tmp2.view(0, rf * H), rf * H)
                 .dispatch(static_cast<uint>(rf * H))
           << m_layernorm(m_tmp2.view(0, rf * H), m_h0.view(0, rf * H),
                          lw.ln2_gamma.view(), lw.ln2_beta.view(), H, kEmbedLayerNormEps)
                 .dispatch(static_cast<uint>(rf * kNormBlock));

        trace(i + 1u, m_h0.view(0, rf * H));
    }

    //--- dense pooling: CLS row + L2 normalize (kernels #8, #9) ---------------
    for(uint32_t b = 0; b < B; b++) {
        cl << m_slice_cls(m_h0.view(static_cast<size_t>(b) * S * H, H),
                          m_cls.view(0, H), H)
                 .dispatch(H)
           << m_l2norm(m_cls.view(0, H), m_dense_dev.view(static_cast<size_t>(b) * H, H), H)
                 .dispatch(kNormBlock);
    }

    //--- optional retrieval heads (kernel #10) --------------------------------
    if(job.want_sparse) {
        if(job.sparse_out.size() != rf) [[unlikely]] {
            LUISA_ERROR("executor: want_sparse but sparse_out has {} floats (need {}).",
                        job.sparse_out.size(), rf);
        }
        cl << m_sparse_head(m_h0.view(0, rf * H), m_weights.sparse_w().view(),
                            m_weights.sparse_b(), m_sparse_dev.view(0, rf), H)
                 .dispatch(static_cast<uint>(rf * kHeadsBlock));
    }
    if(job.want_colbert) {
        if(job.colbert_out.size() != rf * H) [[unlikely]] {
            LUISA_ERROR("executor: want_colbert but colbert_out has {} floats (need {}).",
                        job.colbert_out.size(), rf * H);
        }
        cl << m_colbert_proj(m_h0.view(0, rf * H), m_weights.colbert_w().view(),
                             m_weights.colbert_b().view(), m_colbert_dev.view(0, rf * H),
                             H, H, 1u)
                 .dispatch(static_cast<uint>(rf * kHeadsBlock));
    }

    //--- readback --------------------------------------------------------------
    cl << m_dense_dev.view(0, static_cast<size_t>(B) * H).copy_to(job.dense_out);
    if(job.want_sparse) {
        cl << m_sparse_dev.view(0, rf).copy_to(job.sparse_out);
    }
    if(job.want_colbert) {
        cl << m_colbert_dev.view(0, rf * H).copy_to(job.colbert_out);
    }
}

void Executor::fire_trace_callbacks() noexcept {
    if(!m_trace_cb) {
        return;
    }
    const auto n = static_cast<size_t>(m_trace_rows) * m_trace_cols;
    for(uint32_t k = 0; k < m_traced_states; k++) {
        m_trace_cb(k, m_trace_slots[k].view(0, n), m_trace_rows, m_trace_cols);
    }
    m_traced_states = 0u;
}

}// namespace luisa::embed
