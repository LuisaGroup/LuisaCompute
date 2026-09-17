// Host-side n-gram model trainer. See ngram_trainer.h for the design; the
// device kernels live in ngram_train_kernels.cpp.

#include "ngram_trainer.h"

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>

#include <algorithm>
#include <cmath>

namespace tokenize {

NgramCountIndex build_lm_count_index(luisa::span<const uint32_t> padded,
                                     uint32_t order, uint32_t bos_id) {
    const uint32_t len = static_cast<uint32_t>(padded.size());
    LUISA_ASSERT(order >= 1u, "order must be >= 1");
    LUISA_ASSERT(len > 0, "padded corpus must not be empty");
    // strict upper bound on the number of distinct windows (BOS-skipped
    // windows included, before filtering)
    uint64_t entries_max = 0;
    for (uint32_t n = 1u; n <= order; ++n) {
        if (n <= len) entries_max += len - n + 1u;
    }
    LUISA_ASSERT(entries_max > 0, "corpus too small for the model order");
    // keep the load factor <= 1/2 so every probe chain reaches an empty slot
    uint32_t cap_log2 = 1;
    while ((uint64_t{1} << cap_log2) < entries_max * 2) ++cap_log2;
    LUISA_ASSERT(cap_log2 < 64u, "table too large");
    const uint64_t cap = uint64_t{1} << cap_log2;
    const uint64_t mask = cap - 1;

    NgramCountIndex index;
    index.cap_log2 = cap_log2;
    index.keys.assign(cap, 0ull);
    index.pos.assign(cap, 0u);

    for (uint32_t n = 1u; n <= order; ++n) {
        if (n > len) break;
        for (uint32_t p = 0; p + n <= len; ++p) {
            // skip windows containing a BOS at offset > 0: exactly the
            // cross-sentence windows (BOS only follows EOS)
            bool has_bos = false;
            for (uint32_t j = 1u; j < n; ++j) {
                has_bos |= padded[p + j] == bos_id;
            }
            if (has_bos) continue;
            uint64_t key = detail::fnv1a64_tokens(padded.data() + p, n);
            if (key == 0ull) key = 1ull;// reserve 0 for empty slots
            uint64_t slot = (key * 0x9E3779B97F4A7C15ull) >> (64 - cap_log2);
            for (;;) {
                const uint64_t k = index.keys[slot];
                if (k == 0ull) {// empty slot: insert (positions grow, first wins)
                    index.keys[slot] = key;
                    index.pos[slot] = p;
                    break;
                }
                if (k == key) {
                    // same hash: same n-gram or an FNV collision -- verify
                    const uint32_t q = index.pos[slot];
                    bool same = true;
                    for (uint32_t j = 0; same && j < n; ++j) {
                        same = padded[q + j] == padded[p + j];
                    }
                    if (same) break;
                }
                slot = (slot + 1ull) & mask;
            }
        }
    }
    return index;
}

NgramTrainer::NgramTrainer(luisa::compute::Device &device,
                           luisa::compute::Stream &stream,
                           NgramLibrary &library,
                           NgramTrainOptions options)
    : _device(device), _stream(stream), _lib(library), _options(options) {
    LUISA_ASSERT(library.finalized(), "library must be finalized before training");
    LUISA_ASSERT(_lib.size() > 0u, "library must not be empty");
    LUISA_ASSERT(options.min_n >= 1u, "min_n must be >= 1");
    LUISA_ASSERT(options.min_n <= options.max_n, "min_n must be <= max_n");
    LUISA_ASSERT(options.add_k > 0.0f, "add_k must be > 0");

    luisa::Clock clock;

    // stage 1 (host): count index, layout-compatible with the retriever's
    // hash index over the SAME flat corpus, window rule and [min_n, max_n]
    clock.tic();
    _index = build_ngram_count_index(luisa::span{_lib.tokens},
                                     _options.min_n, _options.max_n);
    const double index_ms = clock.toc();

    const auto lib_len = static_cast<uint32_t>(_lib.size());
    _corpus_buf = _device.create_buffer<uint32_t>(_lib.tokens.size());
    _index.keys_buf = _device.create_buffer<uint64_t>(_index.keys.size());
    _index.pos_buf = _device.create_buffer<uint32_t>(_index.pos.size());
    _index.counts_buf = _device.create_buffer<uint32_t>(_index.keys.size());
    _stream << _corpus_buf.view().copy_from(luisa::span{_lib.tokens})
            << _index.keys_buf.view().copy_from(luisa::span{_index.keys})
            << _index.pos_buf.view().copy_from(luisa::span{_index.pos});

    // stage 2 (device): counting -- the O(corpus * range) core of training
    reset_counts();
    _count_shader = _device.compile(make_count_kernel(_index.cap_log2));
    _score_shader = _device.compile(make_draft_score_kernel(_index.cap_log2));
    clock.tic();
    dispatch_count();
    _stream << luisa::compute::synchronize();
    const double count_ms = clock.toc();
    LUISA_INFO("n-gram training: corpus {} tokens, index {} slots (2^{}) built "
               "in {:.2f} ms, counted (min_n={}, max_n={}) in {:.2f} ms",
               lib_len, _index.keys.size(), _index.cap_log2, index_ms,
               _options.min_n, _options.max_n, count_ms);

    if (_options.lm_mode) _train_lm_pipeline();
}

void NgramTrainer::reset_counts() {
    // portable zero-init: one upload of a zeroed host vector (no reliance on
    // backend buffer-fill)
    if (_zeros.size() != _index.keys.size()) {
        _zeros.assign(_index.keys.size(), 0u);
    }
    _stream << _index.counts_buf.view().copy_from(luisa::span{_zeros});
}

void NgramTrainer::dispatch_count() {
    _stream << _count_shader(_corpus_buf, static_cast<uint32_t>(_lib.size()),
                             _index.keys_buf, _index.pos_buf, _index.counts_buf,
                             _options.min_n, _options.max_n)
                   .dispatch(static_cast<uint32_t>(_lib.size()));
}

void NgramTrainer::download_counts() {
    _index.counts_host.resize(_index.keys.size());
    _stream << _index.counts_buf.view().copy_to(luisa::span{_index.counts_host})
            << luisa::compute::synchronize();
}

void NgramTrainer::upload_score_batch(luisa::span<const uint32_t> queries_flat,
                                      luisa::span<const uint32_t> query_lens,
                                      uint32_t query_stride,
                                      luisa::span<const uint32_t> drafts,
                                      luisa::span<const uint32_t> draft_lens,
                                      uint32_t k) {
    const auto num_rows = query_lens.size();
    LUISA_ASSERT(num_rows > 0, "score batch requires at least one row");
    LUISA_ASSERT(queries_flat.size() == num_rows * query_stride,
                 "queries_flat size {} != num_rows * query_stride {}",
                 queries_flat.size(), num_rows * query_stride);
    LUISA_ASSERT(draft_lens.size() == num_rows, "draft_lens size mismatch");
    LUISA_ASSERT(drafts.size() == num_rows * k, "drafts size mismatch");
    for (auto len : query_lens) {
        LUISA_ASSERT(len <= query_stride,
                     "query length {} exceeds query_stride {}", len, query_stride);
    }
    for (auto len : draft_lens) {
        LUISA_ASSERT(len <= k, "draft length {} exceeds k {}", len, k);
    }
    // grow-only staging buffers (sized by the maxima seen so far; rows are
    // packed with the batch's own strides, passed to the kernel at dispatch)
    if (num_rows > _sq_capacity || query_stride > _sq_stride || k > _sq_k) {
        _sq_capacity = std::max(num_rows, _sq_capacity);
        _sq_stride = std::max(query_stride, _sq_stride);
        _sq_k = std::max(k, _sq_k);
        _sq_queries_buf = _device.create_buffer<uint32_t>(_sq_capacity * _sq_stride);
        _sq_qlens_buf = _device.create_buffer<uint32_t>(_sq_capacity);
        _sq_drafts_buf = _device.create_buffer<uint32_t>(_sq_capacity * _sq_k);
        _sq_draft_lens_buf = _device.create_buffer<uint32_t>(_sq_capacity);
        _sq_out_buf = _device.create_buffer<float>(_sq_capacity);
    }
    _sq_batch_stride = query_stride;
    _sq_batch_k = k;
    _stream << _sq_queries_buf.view(0, num_rows * query_stride).copy_from(queries_flat)
            << _sq_qlens_buf.view(0, num_rows).copy_from(query_lens)
            << _sq_drafts_buf.view(0, num_rows * k).copy_from(drafts)
            << _sq_draft_lens_buf.view(0, num_rows).copy_from(draft_lens);
}

void NgramTrainer::dispatch_score_batch(size_t num_rows) {
    LUISA_ASSERT(num_rows > 0 && num_rows <= _sq_capacity,
                 "invalid score batch {}", num_rows);
    _stream << _score_shader(_corpus_buf,
                             _index.keys_buf, _index.pos_buf, _index.counts_buf,
                             _sq_queries_buf, _sq_qlens_buf, _sq_batch_stride,
                             _sq_drafts_buf, _sq_draft_lens_buf, _sq_batch_k,
                             _sq_out_buf,
                             _options.max_n, _options.add_k,
                             static_cast<float>(_lib.vocab_size))
                   .dispatch(static_cast<uint32_t>(num_rows));
}

void NgramTrainer::download_score_batch(size_t num_rows,
                                        luisa::vector<float> &log2_probs) {
    LUISA_ASSERT(num_rows > 0 && num_rows <= _sq_capacity,
                 "invalid score batch {}", num_rows);
    log2_probs.resize(num_rows);
    _stream << _sq_out_buf.view(0, num_rows).copy_to(luisa::span{log2_probs})
            << luisa::compute::synchronize();
}

void NgramTrainer::score_drafts(luisa::span<const uint32_t> queries_flat,
                                luisa::span<const uint32_t> query_lens,
                                uint32_t query_stride,
                                luisa::span<const uint32_t> drafts,
                                luisa::span<const uint32_t> draft_lens,
                                uint32_t k,
                                luisa::vector<float> &log2_probs) {
    upload_score_batch(queries_flat, query_lens, query_stride,
                       drafts, draft_lens, k);
    dispatch_score_batch(query_lens.size());
    download_score_batch(query_lens.size(), log2_probs);
}

// ---------- LM mode ----------

void NgramTrainer::_train_lm_pipeline() {
    LUISA_ASSERT(_options.order >= 1u, "LM order must be >= 1");
    const uint32_t vocab = _lib.vocab_size;
    LUISA_ASSERT(vocab > 0u, "LM mode requires a non-empty vocabulary");
    _unk_id = vocab;
    _bos_id = vocab + 1u;
    _eos_id = vocab + 2u;
    _vocab_eff = vocab + 3u;

    luisa::Clock clock;

    // stage LM-1 (device): unigram histogram, drives the UNK remap
    auto hist_buf = _device.create_buffer<uint32_t>(vocab);
    {
        luisa::vector<uint32_t> zeros(vocab, 0u);
        _stream << hist_buf.view().copy_from(luisa::span{zeros});
    }
    auto hist_shader = _device.compile(make_unigram_histogram_kernel());
    clock.tic();
    _stream << hist_shader(_corpus_buf, static_cast<uint32_t>(_lib.size()), hist_buf)
                   .dispatch(static_cast<uint32_t>(_lib.size()));
    luisa::vector<uint32_t> hist(vocab);
    _stream << hist_buf.view().copy_to(luisa::span{hist})
            << luisa::compute::synchronize();
    const double hist_ms = clock.toc();

    // stage LM-2 (host): UNK remap + padded corpus (BOS w1..wL EOS per doc)
    _remap.resize(vocab);
    for (uint32_t t = 0; t < vocab; ++t) {
        _remap[t] = hist[t] < _options.unk_threshold ? _unk_id : t;
    }
    const uint32_t num_sents = _lib.num_docs();
    _padded.reserve(_lib.size() + 2u * num_sents);
    _sent_off.reserve(num_sents);
    for (uint32_t d = 0; d < num_sents; ++d) {
        _sent_off.push_back(static_cast<uint32_t>(_padded.size()));
        _padded.push_back(_bos_id);
        for (uint32_t i = _lib.doc_offsets[d]; i < _lib.doc_offsets[d + 1u]; ++i) {
            _padded.push_back(_remap[_lib.tokens[i]]);
        }
        _padded.push_back(_eos_id);
    }
    _total_tokens = static_cast<uint32_t>(_padded.size());

    // stage LM-3 (host): count index over the padded corpus
    clock.tic();
    _lm_index = build_lm_count_index(luisa::span{_padded}, _options.order, _bos_id);
    const double lm_index_ms = clock.toc();

    _lm_corpus_buf = _device.create_buffer<uint32_t>(_padded.size());
    _lm_index.keys_buf = _device.create_buffer<uint64_t>(_lm_index.keys.size());
    _lm_index.pos_buf = _device.create_buffer<uint32_t>(_lm_index.pos.size());
    _lm_index.counts_buf = _device.create_buffer<uint32_t>(_lm_index.keys.size());
    luisa::vector<uint32_t> zeros(_lm_index.keys.size(), 0u);
    _stream << _lm_corpus_buf.view().copy_from(luisa::span{_padded})
            << _lm_index.keys_buf.view().copy_from(luisa::span{_lm_index.keys})
            << _lm_index.pos_buf.view().copy_from(luisa::span{_lm_index.pos})
            << _lm_index.counts_buf.view().copy_from(luisa::span{zeros});

    // stage LM-4 (device): LM counting
    _lm_count_shader = _device.compile(make_lm_count_kernel(_lm_index.cap_log2));
    _lm_score_shader = _device.compile(make_lm_score_kernel(_lm_index.cap_log2));
    clock.tic();
    _stream << _lm_count_shader(_lm_corpus_buf, _total_tokens,
                                _lm_index.keys_buf, _lm_index.pos_buf,
                                _lm_index.counts_buf,
                                _options.order, _bos_id)
                   .dispatch(_total_tokens);
    _stream << luisa::compute::synchronize();
    const double lm_count_ms = clock.toc();
    LUISA_INFO("LM training: {} sentences, padded {} tokens (V={}), histogram "
               "{:.2f} ms, index {} slots (2^{}) in {:.2f} ms, counted "
               "(order={}) in {:.2f} ms",
               num_sents, _total_tokens, _vocab_eff, hist_ms,
               _lm_index.keys.size(), _lm_index.cap_log2, lm_index_ms,
               _options.order, lm_count_ms);
}

uint32_t NgramTrainer::lm_remap(uint32_t token) const noexcept {
    if (token >= _lib.vocab_size) return _unk_id;// out of vocabulary
    return _remap[token];
}

void NgramTrainer::score_sentences(luisa::span<const uint32_t> rows_flat,
                                   luisa::span<const uint32_t> row_lens,
                                   luisa::vector<float> &log2_probs) {
    LUISA_ASSERT(_options.lm_mode, "score_sentences() requires lm_mode");
    const auto num_rows = row_lens.size();
    LUISA_ASSERT(num_rows > 0, "score_sentences() requires at least one row");
    uint32_t stride = 0;
    for (auto len : row_lens) {
        LUISA_ASSERT(len >= 1u, "sentence rows must have length >= 1");
        stride = std::max(stride, len);
    }
    LUISA_ASSERT(rows_flat.size() == num_rows * stride,
                 "rows_flat size {} != num_rows * stride {}",
                 rows_flat.size(), num_rows * stride);
    if (num_rows > _lm_capacity || stride > _lm_stride) {
        _lm_capacity = std::max(num_rows, _lm_capacity);
        _lm_stride = std::max(stride, _lm_stride);
        _lm_rows_buf = _device.create_buffer<uint32_t>(_lm_capacity * _lm_stride);
        _lm_lens_buf = _device.create_buffer<uint32_t>(_lm_capacity);
        _lm_out_buf = _device.create_buffer<float>(_lm_capacity);
    }
    _stream << _lm_rows_buf.view(0, num_rows * stride).copy_from(rows_flat)
            << _lm_lens_buf.view(0, num_rows).copy_from(row_lens);
    _stream << _lm_score_shader(_lm_corpus_buf,
                                _lm_index.keys_buf, _lm_index.pos_buf,
                                _lm_index.counts_buf,
                                _lm_rows_buf, _lm_lens_buf, stride, _lm_out_buf,
                                _options.order,
                                static_cast<uint32_t>(_options.smoothing),
                                _options.add_k,
                                static_cast<float>(_vocab_eff),
                                _total_tokens)
                   .dispatch(static_cast<uint32_t>(num_rows));
    log2_probs.resize(num_rows);
    _stream << _lm_out_buf.view(0, num_rows).copy_to(luisa::span{log2_probs})
            << luisa::compute::synchronize();
}

double NgramTrainer::perplexity(luisa::span<const uint32_t> rows_flat,
                                luisa::span<const uint32_t> row_lens) {
    luisa::vector<float> log2_probs;
    score_sentences(rows_flat, row_lens, log2_probs);
    double sum = 0.0;
    uint64_t m = 0;
    for (size_t i = 0; i < log2_probs.size(); ++i) {
        sum += static_cast<double>(log2_probs[i]);
        m += row_lens[i] - 1u;
    }
    LUISA_ASSERT(m > 0, "perplexity requires at least one predicted token");
    return std::exp2(-sum / static_cast<double>(m));
}

float NgramTrainer::conditional_prob(luisa::span<const uint32_t> prefix,
                                     uint32_t token) {
    LUISA_ASSERT(_options.lm_mode, "conditional_prob() requires lm_mode");
    LUISA_ASSERT(!prefix.empty(), "prefix must contain at least one token");
    // row 0: prefix ++ [token]; row 1: prefix. The difference of the two
    // scores is the log-probability of the last token given the context.
    const uint32_t plen = static_cast<uint32_t>(prefix.size());
    const uint32_t stride = plen + 1u;
    luisa::vector<uint32_t> rows_flat(2u * stride, 0u);
    std::copy(prefix.begin(), prefix.end(), rows_flat.begin());
    std::copy(prefix.begin(), prefix.end(), rows_flat.begin() + stride);
    rows_flat[plen] = token;
    const luisa::vector<uint32_t> row_lens = {stride, plen};
    luisa::vector<float> log2_probs;
    score_sentences(luisa::span{rows_flat}, luisa::span{row_lens}, log2_probs);
    return std::exp2(log2_probs[0] - log2_probs[1]);
}

}// namespace tokenize
