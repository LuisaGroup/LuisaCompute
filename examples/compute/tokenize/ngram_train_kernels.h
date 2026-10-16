#pragma once

// DSL kernels for n-gram TRAINING and the trained-statistics consumers.
//
// Training = counting. Following the retrieval inference logic exactly
// (see ngram_kernels.h), the count kernel tallies, for every corpus position
// p and every n in [min_n, max_n] with p + n < lib_len (the retriever's own
// match condition -- a match must leave at least one continuation token):
//   - the length-n prefix window L[p..p+n)        -- what K1/K2/K3 match on;
//   - the length-(n+1) continuation window L[p..p+n+1) -- the prefix extended
//     by its next token, i.e. the MLE distribution of the proposed token:
//       P(w | matched n-gram ctx) = count(ctx, w) / count(ctx).
// Both counts range over the same position set, so the denominator identity
// count(ctx) == sum_w count(ctx, w) holds by construction.
//
// The host-built count index (build_ngram_count_index, ngram_library.h) is
// layout-compatible with NgramHashIndex: identical keys, probe order and
// earliest-position values for the shared [min_n, max_n] length range, so
// the existing `hash` retrieval kernel runs on it unchanged. On device the
// index is READ-ONLY (lookups + uint32 atomic adds): no concurrent-insertion
// CAS races, no 64-bit atomics (DX lacks generic uint64 buffer atomics).
//
// Additionally, an optional LM mode (default OFF, demo/tests only) implements
// the classic BOS/EOS/UNK + smoothing + perplexity formulation over a padded
// corpus. It never feeds the retrieval path.

#include "ngram_kernels.h"

#include <luisa/dsl/func.h>
#include <luisa/dsl/resource.h>
#include <luisa/runtime/buffer.h>

namespace tokenize {

// K0: unigram histogram. One thread per corpus token;
// vocab_counts[token] += 1. Used by the LM-mode UNK remapping stage.
//   tokens_buf[len]         corpus token IDs
//   vocab_counts_buf[vocab] per-token occurrence counts (zero-initialized)
using HistogramKernel = luisa::compute::Kernel1D<void(
    luisa::compute::Buffer<uint32_t> /*tokens_buf*/, uint32_t /*len*/,
    luisa::compute::Buffer<uint32_t> /*vocab_counts_buf*/)>;
using HistogramShader = luisa::compute::Shader<1,
    luisa::compute::Buffer<uint32_t>, uint32_t,
    luisa::compute::Buffer<uint32_t>>;

// K4: retrieval-aligned n-gram counting. One thread per corpus position p;
// probes the read-only count index and atomically increments the slot of
//   - each length-n prefix window, n in [min_n, max_n], with p + n < lib_len
//     (the retriever's match condition), and
//   - the length-(max_n+1) continuation window at p when p + max_n < lib_len
//     (its last token may be the corpus's final token).
// Each window is counted once per qualifying position (see the denominator
// identity note on build_ngram_count_index in ngram_library.h). Dispatched
// with lib_len threads.
// `cap_log2` (log2 of the table capacity) is baked into the compiled kernel.
using CountKernel = luisa::compute::Kernel1D<void(
    luisa::compute::Buffer<uint32_t> /*lib_buf*/, uint32_t /*lib_len*/,
    luisa::compute::Buffer<uint64_t> /*keys_buf*/,
    luisa::compute::Buffer<uint32_t> /*pos_buf*/,
    luisa::compute::Buffer<uint32_t> /*counts_buf*/,
    uint32_t /*min_n*/, uint32_t /*max_n*/)>;
using CountShader = luisa::compute::Shader<1,
    luisa::compute::Buffer<uint32_t>, uint32_t,
    luisa::compute::Buffer<uint64_t>,
    luisa::compute::Buffer<uint32_t>,
    luisa::compute::Buffer<uint32_t>,
    uint32_t, uint32_t>;

// K5: draft confidence scoring. One thread per query row; consumes the
// retrieval kernels' outputs (drafts + draft_lens) unchanged. For each draft
// position j the context is the trailing min(max_n, qlen + j) tokens of the
// (query ++ draft) sequence and w is draft[j]; the Add-k probability
//   p = (count(ctx, w) + k) / (count(ctx) + k * V)
// is probed from the count table; an unseen context floors at p = 1 / V.
// Accumulates log2(p) into out_buf[qid]. Adds a learned confidence score to
// the current inference without changing a single proposed token.
using DraftScoreKernel = luisa::compute::Kernel1D<void(
    luisa::compute::Buffer<uint32_t> /*lib_buf*/,
    luisa::compute::Buffer<uint64_t> /*keys_buf*/,
    luisa::compute::Buffer<uint32_t> /*pos_buf*/,
    luisa::compute::Buffer<uint32_t> /*counts_buf*/,
    luisa::compute::Buffer<uint32_t> /*query_buf*/,
    luisa::compute::Buffer<uint32_t> /*qlen_buf*/, uint32_t /*query_stride*/,
    luisa::compute::Buffer<uint32_t> /*draft_buf*/,
    luisa::compute::Buffer<uint32_t> /*draft_len_buf*/, uint32_t /*k*/,
    luisa::compute::Buffer<float> /*out_buf*/,
    uint32_t /*max_n*/, float /*add_k*/, float /*vocab*/)>;
using DraftScoreShader = luisa::compute::Shader<1,
    luisa::compute::Buffer<uint32_t>,
    luisa::compute::Buffer<uint64_t>,
    luisa::compute::Buffer<uint32_t>,
    luisa::compute::Buffer<uint32_t>,
    luisa::compute::Buffer<uint32_t>,
    luisa::compute::Buffer<uint32_t>, uint32_t,
    luisa::compute::Buffer<uint32_t>,
    luisa::compute::Buffer<uint32_t>, uint32_t,
    luisa::compute::Buffer<float>,
    uint32_t, float, float>;

// K6: MLE continuation retrieval (NgramKernelVariant::parallel_mle). Same
// block-per-query longest-match-first scan as the parallel kernel, but among
// all matches of the winning length the block reduces on (continuation count
// desc, position asc) -- implemented as two portable uint32 shared-memory
// reductions (fetch_max on count, then fetch_min on position among the
// max-count holders; no 64-bit/float atomics). The draft extraction (k
// tokens, clamped) is unchanged. On corpora where every continuation is
// unique this degenerates exactly to the parallel kernel's earliest position.
using RetrieveKernelMle = luisa::compute::Kernel1D<void(
    luisa::compute::Buffer<uint32_t>, uint32_t,
    luisa::compute::Buffer<uint32_t>, luisa::compute::Buffer<uint32_t>,
    uint32_t,
    luisa::compute::Buffer<uint32_t>, luisa::compute::Buffer<uint32_t>,
    uint32_t, uint32_t, uint32_t,
    luisa::compute::Buffer<uint64_t> /*keys_buf*/,
    luisa::compute::Buffer<uint32_t> /*pos_buf*/,
    luisa::compute::Buffer<uint32_t> /*counts_buf*/,
    luisa::compute::Buffer<uint32_t> /*req_off_buf*/)>;
using RetrieveShaderMle = luisa::compute::Shader<1,
    luisa::compute::Buffer<uint32_t>, uint32_t,
    luisa::compute::Buffer<uint32_t>, luisa::compute::Buffer<uint32_t>,
    uint32_t,
    luisa::compute::Buffer<uint32_t>, luisa::compute::Buffer<uint32_t>,
    uint32_t, uint32_t, uint32_t,
    luisa::compute::Buffer<uint64_t>, luisa::compute::Buffer<uint32_t>,
    luisa::compute::Buffer<uint32_t>,
    luisa::compute::Buffer<uint32_t>>;

// K7 (LM mode): n-gram counting over the padded corpus. One thread per
// position p; for n in [1, order] with p + n <= len (EOS IS a legitimate
// final token), skipping any window that contains a BOS at offset > 0 (which
// exactly excludes cross-sentence windows, since BOS only follows EOS).
using LmCountKernel = luisa::compute::Kernel1D<void(
    luisa::compute::Buffer<uint32_t> /*corpus_buf*/, uint32_t /*len*/,
    luisa::compute::Buffer<uint64_t> /*keys_buf*/,
    luisa::compute::Buffer<uint32_t> /*pos_buf*/,
    luisa::compute::Buffer<uint32_t> /*counts_buf*/,
    uint32_t /*order*/, uint32_t /*bos_id*/)>;
using LmCountShader = luisa::compute::Shader<1,
    luisa::compute::Buffer<uint32_t>, uint32_t,
    luisa::compute::Buffer<uint64_t>,
    luisa::compute::Buffer<uint32_t>,
    luisa::compute::Buffer<uint32_t>,
    uint32_t, uint32_t>;

// K8 (LM mode): sentence scoring. One thread per row of the row-major batch
// rows_buf[num_rows * stride]; for each position j in [1, len) the context
// length is n = min(order - 1, j) (so count(ctx, w) queries an n-gram of
// length <= order, present in the table) and the predicted token is row[j].
//   add_k mode:   p = (count(ctx, w) + k) / (count(ctx) + k * V)
//   backoff mode: highest order n with count(ctx, w) > 0 wins, p = c / d;
//                 the final floor is the unigram probability
//                 count(w) / total_tokens (always > 0 after the UNK remap).
// order == 1 degenerates to the unigram model (empty context counted as
// total_tokens). Accumulates log2(p) into out_buf[row]. V = vocab + 3
// (includes the <s>/<unk>/<\s> special IDs, matching the Python example's
// explicit V).
using LmScoreKernel = luisa::compute::Kernel1D<void(
    luisa::compute::Buffer<uint32_t> /*corpus_buf*/,
    luisa::compute::Buffer<uint64_t> /*keys_buf*/,
    luisa::compute::Buffer<uint32_t> /*pos_buf*/,
    luisa::compute::Buffer<uint32_t> /*counts_buf*/,
    luisa::compute::Buffer<uint32_t> /*rows_buf*/,
    luisa::compute::Buffer<uint32_t> /*row_lens_buf*/, uint32_t /*stride*/,
    luisa::compute::Buffer<float> /*out_buf*/,
    uint32_t /*order*/, uint32_t /*smoothing*/,
    float /*add_k*/, float /*vocab*/, uint32_t /*total_tokens*/)>;
using LmScoreShader = luisa::compute::Shader<1,
    luisa::compute::Buffer<uint32_t>,
    luisa::compute::Buffer<uint64_t>,
    luisa::compute::Buffer<uint32_t>,
    luisa::compute::Buffer<uint32_t>,
    luisa::compute::Buffer<uint32_t>,
    luisa::compute::Buffer<uint32_t>, uint32_t,
    luisa::compute::Buffer<float>,
    uint32_t, uint32_t, float, float, uint32_t>;

[[nodiscard]] HistogramKernel make_unigram_histogram_kernel() noexcept;
[[nodiscard]] CountKernel make_count_kernel(uint32_t cap_log2) noexcept;
[[nodiscard]] DraftScoreKernel make_draft_score_kernel(uint32_t cap_log2) noexcept;
[[nodiscard]] RetrieveKernelMle make_retrieve_parallel_mle_kernel(
    uint32_t block_size, uint32_t cap_log2) noexcept;
[[nodiscard]] LmCountKernel make_lm_count_kernel(uint32_t cap_log2) noexcept;
[[nodiscard]] LmScoreKernel make_lm_score_kernel(uint32_t cap_log2) noexcept;

}// namespace tokenize
