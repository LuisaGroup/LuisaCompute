#pragma once

// Host-side n-gram model trainer. "Training = counting": the heavy,
// parallel work (unigram histograms, n-gram counting, batch scoring) runs
// on device in DSL kernels (ngram_train_kernels.h/.cpp); the host does I/O,
// hash-index construction, buffer management and final scalar reductions --
// exactly the architecture NgramRetriever uses for retrieval.
//
// The trained count index is RETRIEVAL-ALIGNED (see ngram_library.h): it
// counts exactly the n-grams and continuations the inference logic matches
// on, over the same flat corpus, with the same window rule, the same
// [min_n, max_n] range and the same hash layout, so the trained statistics
// are directly consumable by the existing kernels (hash / parallel_mle) and
// by the draft scorer. The classic BOS/EOS/UNK + perplexity LM formulation
// (lm_mode) is optional, default-off, and never alters the retrieval path.

#include "ngram_kernels.h"
#include "ngram_library.h"
#include "ngram_train_kernels.h"

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace tokenize {

// Build the LM-mode count index over a padded corpus (BOS w1..wL EOS per
// sentence): windows of length n in [1, order] with p + n <= len (EOS is a
// legitimate final token), skipping any window that contains a BOS at
// offset > 0 (exactly the cross-sentence windows). Same table layout, keys
// and probe order as build_ngram_count_index. Declared here for tests.
[[nodiscard]] NgramCountIndex build_lm_count_index(
    luisa::span<const uint32_t> padded, uint32_t order, uint32_t bos_id);

struct NgramTrainOptions {
    uint32_t min_n = 2;           // MUST match the retriever configuration
    uint32_t max_n = 3;           // MUST match the retriever configuration
    float add_k = 0.1f;           // Add-k smoothing for draft scoring
    bool lm_mode = false;         // BOS/EOS/UNK + perplexity (demo/tests only)
    uint32_t order = 2;           // LM-mode model order N; lm_mode only
    uint32_t unk_threshold = 1;   // LM mode: unigram count < threshold -> <unk>
    NgramSmoothing smoothing = NgramSmoothing::add_k;// LM-mode scoring mode
};

class NgramTrainer {
public:
    // Borrows `device`/`stream`/`library` (which must outlive the trainer).
    // Builds the count index on host, uploads it, compiles the kernels and
    // runs the counting dispatch; the model is fully trained when the
    // constructor returns. With options.lm_mode, additionally builds the
    // padded BOS/EOS/UNK corpus and trains the LM count table.
    NgramTrainer(luisa::compute::Device &device,
                 luisa::compute::Stream &stream,
                 NgramLibrary &library,
                 NgramTrainOptions options);

    // ---- retrieval-aligned training products ----

    // The trained count index; its buffers feed the hash / parallel_mle
    // retrieval kernels and the draft scorer.
    [[nodiscard]] const NgramCountIndex &index() const noexcept { return _index; }
    [[nodiscard]] uint32_t min_n() const noexcept { return _options.min_n; }
    [[nodiscard]] uint32_t max_n() const noexcept { return _options.max_n; }
    [[nodiscard]] float add_k() const noexcept { return _options.add_k; }

    // Refresh index().counts_host from the device counts (tests/demo).
    void download_counts();

    // Re-training support for benchmarks: zero the device counts and run one
    // counting dispatch on the trainer's stream (the caller synchronizes).
    void reset_counts();
    void dispatch_count();

    // ---- draft scoring (device) ----
    // Score a batch of retrieval results: row i is query rows_flat[i*stride..]
    // of length query_lens[i] with its draft drafts[i*k..] of length
    // draft_lens[i]; out log2_probs[i] is the sum of the per-token Add-k
    // log2 probabilities (0 for empty drafts). Split into
    // upload/dispatch/download so benchmarks can time the dispatch alone.
    void upload_score_batch(luisa::span<const uint32_t> queries_flat,
                            luisa::span<const uint32_t> query_lens, uint32_t query_stride,
                            luisa::span<const uint32_t> drafts,
                            luisa::span<const uint32_t> draft_lens, uint32_t k);
    void dispatch_score_batch(size_t num_rows);
    void download_score_batch(size_t num_rows, luisa::vector<float> &log2_probs);
    void score_drafts(luisa::span<const uint32_t> queries_flat,
                      luisa::span<const uint32_t> query_lens, uint32_t query_stride,
                      luisa::span<const uint32_t> drafts,
                      luisa::span<const uint32_t> draft_lens, uint32_t k,
                      luisa::vector<float> &log2_probs);

    // ---- LM mode (options.lm_mode == true only) ----
    [[nodiscard]] bool lm_mode() const noexcept { return _options.lm_mode; }
    [[nodiscard]] uint32_t order() const noexcept { return _options.order; }
    [[nodiscard]] uint32_t unk_id() const noexcept { return _unk_id; }
    [[nodiscard]] uint32_t bos_id() const noexcept { return _bos_id; }
    [[nodiscard]] uint32_t eos_id() const noexcept { return _eos_id; }
    // effective vocabulary size including the <unk>/<s>/<\s> special IDs
    [[nodiscard]] uint32_t vocab_effective() const noexcept { return _vocab_eff; }
    [[nodiscard]] uint32_t total_tokens() const noexcept { return _total_tokens; }
    [[nodiscard]] luisa::span<const uint32_t> padded_corpus() const noexcept { return _padded; }
    [[nodiscard]] const NgramCountIndex &lm_index() const noexcept { return _lm_index; }

    // Apply the UNK remap to a raw library token (LM mode): tokens with
    // unigram count < unk_threshold and any out-of-vocabulary ID map to unk.
    [[nodiscard]] uint32_t lm_remap(uint32_t token) const noexcept;

    // Score a batch of sentence rows (device): rows_flat is row-major with
    // the given stride, row_lens[i] is the actual length of row i (>= 1).
    // Rows do NOT need BOS: position j uses context n = min(order - 1, j). A row
    // [<s>, w..., </s>] reproduces the Python sentence_prob, a row [prev, w]
    // reproduces P(w|prev). Out log2_probs[i] = sum_j log2 p(row[j] | ctx).
    void score_sentences(luisa::span<const uint32_t> rows_flat,
                         luisa::span<const uint32_t> row_lens,
                         luisa::vector<float> &log2_probs);

    // Corpus perplexity over the row batch (host reduction):
    //   PPL = 2^( -sum(log2p) / sum(m) ) with m = sum(len_row - 1) predicted
    // tokens. Requires sum(m) > 0.
    [[nodiscard]] double perplexity(luisa::span<const uint32_t> rows_flat,
                                    luisa::span<const uint32_t> row_lens);

    // Convenience over score_sentences: P(token | prefix) under the LM.
    float conditional_prob(luisa::span<const uint32_t> prefix, uint32_t token);

private:
    void _train_lm_pipeline();

    luisa::compute::Device &_device;
    luisa::compute::Stream &_stream;
    NgramLibrary &_lib;
    NgramTrainOptions _options;

    luisa::compute::Buffer<uint32_t> _corpus_buf;// device copy of the corpus
    NgramCountIndex _index;
    luisa::vector<uint32_t> _zeros;// persistent zero-fill for reset_counts()
    CountShader _count_shader;
    DraftScoreShader _score_shader;

    // draft-scoring staging buffers (grow-only)
    size_t _sq_capacity = 0;
    uint32_t _sq_stride = 0;
    uint32_t _sq_k = 0;
    uint32_t _sq_batch_stride = 0;// strides of the last uploaded batch
    uint32_t _sq_batch_k = 0;
    luisa::compute::Buffer<uint32_t> _sq_queries_buf;
    luisa::compute::Buffer<uint32_t> _sq_qlens_buf;
    luisa::compute::Buffer<uint32_t> _sq_drafts_buf;
    luisa::compute::Buffer<uint32_t> _sq_draft_lens_buf;
    luisa::compute::Buffer<float> _sq_out_buf;

    // ---- LM mode state ----
    uint32_t _unk_id = 0;
    uint32_t _bos_id = 0;
    uint32_t _eos_id = 0;
    uint32_t _vocab_eff = 0;   // vocab_size + 3
    uint32_t _total_tokens = 0;// padded corpus length
    luisa::vector<uint32_t> _remap;    // raw token -> remapped token
    luisa::vector<uint32_t> _padded;   // BOS w1..wL EOS per sentence
    luisa::vector<uint32_t> _sent_off; // per-sentence offsets into _padded
    luisa::compute::Buffer<uint32_t> _lm_corpus_buf;
    NgramCountIndex _lm_index;
    LmCountShader _lm_count_shader;
    LmScoreShader _lm_score_shader;

    // LM sentence-scoring staging buffers (grow-only)
    size_t _lm_capacity = 0;
    uint32_t _lm_stride = 0;
    luisa::compute::Buffer<uint32_t> _lm_rows_buf;
    luisa::compute::Buffer<uint32_t> _lm_lens_buf;
    luisa::compute::Buffer<float> _lm_out_buf;
};

}// namespace tokenize
