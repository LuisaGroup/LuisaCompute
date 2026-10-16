// Device-side n-gram training kernels. See ngram_train_kernels.h for the
// semantics; the helpers below MUST stay byte-identical to the host index
// builders (ngram_library.cpp / ngram_trainer.cpp) so hashes and probe
// sequences agree across the host/device boundary.

#include "ngram_train_kernels.h"

#include "ngram_library.h"

#include <luisa/dsl/sugar.h>

namespace tokenize {

using namespace luisa;
using namespace luisa::compute;

// NOTE on DSL control flow (same as ngram_kernels.cpp): runtime branching
// uses $if/$for/$while with Var-typed conditions; there is no break, so loop
// exits use flag variables. Scalar boolean logic uses bitwise & | (the DSL
// && || are vector-only and unary ! is unavailable). The helper functions
// below are plain C++ templates: they run at kernel-definition time and
// inline their AST into the caller, so the probe/hash code exists exactly
// once in source while expanding into every kernel that uses it.

namespace detail {

// FNV-1a 64 over the n tokens produced by read(j), byte-identical to
// detail::fnv1a64_tokens in ngram_library.h.
template<typename Read>
[[nodiscard]] inline Var<uint64_t> fnv1a64_tokens_dsl(Read &&read,
                                                      Var<uint32_t> n) noexcept {
    Var<uint64_t> key = 14695981039346656037ull;
    $for (j, 0u, n) {
        Var<uint64_t> tok = cast<uint64_t>(read(j));
        $for (b, 0u, 4u) {
            key = (key ^ ((tok >> (b * 8)) & 0xFFull)) * 1099511628211ull;
        };
    };
    $if (key == 0ull) {
        key = 1ull;// reserve 0 for empty slots
    };
    return key;
}

// Probe the open-addressing count index for the window of length n whose
// tokens are produced by read_window(j). Identical probe sequence to the
// host builders; FNV collisions between different n-grams are content
// -verified against the corpus via read_corpus(q + j). Returns the slot
// index, or ngram_invalid_id when the window is not indexed (probe reached
// an empty slot; guaranteed to terminate because the load factor <= 1/2).
template<typename ReadWindow, typename ReadCorpus>
[[nodiscard]] inline Var<uint32_t> probe_count_index(
    const BufferVar<uint64_t> &keys_buf, const BufferUInt &pos_buf,
    ReadWindow &&read_window, ReadCorpus &&read_corpus,
    Var<uint32_t> n, Var<uint64_t> key, uint32_t cap_log2) noexcept {
    Var<uint32_t> result = ngram_invalid_id;
    Var<uint32_t> done = 0u;
    Var<uint64_t> slot = (key * 0x9E3779B97F4A7C15ull) >> (64 - cap_log2);
    $while (done == 0u) {
        Var<uint64_t> k2 = keys_buf->read(slot.cast<uint32_t>());
        $if (k2 == key) {
            Var<uint32_t> q = pos_buf->read(slot.cast<uint32_t>());
            Var<bool> same = true;
            $for (j, 0u, n) {
                $if (same) {
                    same = read_corpus(q + j) == read_window(j);
                };
            };
            $if (same) {
                result = slot.cast<uint32_t>();
                done = 1u;
            };
        };
        $if (done == 0u & k2 == 0ull) {
            done = 2u;// empty slot: window not indexed (defensive exit)
        };
        $if (done == 0u) {
            slot = (slot + 1ull) & ((1ull << cap_log2) - 1ull);
        };
    };
    return result;
}

// Look up the count of the window of length n produced by read_window(j);
// 0 when the window is not indexed.
template<typename ReadWindow, typename ReadCorpus>
[[nodiscard]] inline Var<uint32_t> lookup_count(
    const BufferVar<uint64_t> &keys_buf, const BufferUInt &pos_buf,
    const BufferUInt &counts_buf,
    ReadWindow &&read_window, ReadCorpus &&read_corpus,
    Var<uint32_t> n, uint32_t cap_log2) noexcept {
    Var<uint64_t> key = fnv1a64_tokens_dsl(read_window, n);
    Var<uint32_t> slot = probe_count_index(keys_buf, pos_buf,
                                           read_window, read_corpus,
                                           n, key, cap_log2);
    Var<uint32_t> c = 0u;
    $if (slot != ngram_invalid_id) {
        c = counts_buf->read(slot);
    };
    return c;
}

}// namespace detail

HistogramKernel make_unigram_histogram_kernel() noexcept {
    return [](BufferUInt tokens_buf, Var<uint32_t> len,
              BufferUInt vocab_counts_buf) noexcept {
        Var<uint32_t> p = dispatch_x();
        $if (p < len) {
            vocab_counts_buf->atomic(tokens_buf->read(p)).fetch_add(1u);
        };
    };
}

CountKernel make_count_kernel(uint32_t cap_log2) noexcept {
    return [cap_log2](BufferUInt lib_buf, Var<uint32_t> lib_len,
                      BufferVar<uint64_t> keys_buf, BufferUInt pos_buf,
                      BufferUInt counts_buf,
                      Var<uint32_t> min_n, Var<uint32_t> max_n) noexcept {
        Var<uint32_t> p = dispatch_x();
        auto read_corpus = [&](Var<uint32_t> i) noexcept -> Var<uint32_t> {
            return lib_buf->read(i);
        };
        // length-n prefix windows: the retriever's match condition p+n<lib_len
        $for (n, min_n, max_n + 1u) {
            $if (p + n < lib_len) {
                auto read_window = [&](Var<uint32_t> j) noexcept -> Var<uint32_t> {
                    return lib_buf->read(p + j);
                };
                Var<uint64_t> key = detail::fnv1a64_tokens_dsl(read_window, n);
                Var<uint32_t> slot = detail::probe_count_index(
                    keys_buf, pos_buf, read_window, read_corpus,
                    n, key, cap_log2);
                // every window enumerated here IS in the index; the
                // guard is defensive only
                $if (slot != ngram_invalid_id) {
                    counts_buf->atomic(slot).fetch_add(1u);
                };
            };
        };
        // the length-(max_n+1) continuation window at p; its last token may
        // be the corpus's final token (p + max_n + 1 == lib_len allowed)
        $if (p + max_n < lib_len) {
            auto read_window = [&](Var<uint32_t> j) noexcept -> Var<uint32_t> {
                return lib_buf->read(p + j);
            };
            Var<uint64_t> key = detail::fnv1a64_tokens_dsl(read_window, max_n + 1u);
            Var<uint32_t> slot = detail::probe_count_index(
                keys_buf, pos_buf, read_window, read_corpus,
                max_n + 1u, key, cap_log2);
            $if (slot != ngram_invalid_id) {
                counts_buf->atomic(slot).fetch_add(1u);
            };
        };
    };
}

DraftScoreKernel make_draft_score_kernel(uint32_t cap_log2) noexcept {
    return [cap_log2](BufferUInt lib_buf,
                      BufferVar<uint64_t> keys_buf, BufferUInt pos_buf,
                      BufferUInt counts_buf,
                      BufferUInt query_buf, BufferUInt qlen_buf,
                      Var<uint32_t> query_stride,
                      BufferUInt draft_buf, BufferUInt draft_len_buf,
                      Var<uint32_t> k,
                      BufferFloat out_buf,
                      Var<uint32_t> max_n, Var<float> add_k,
                      Var<float> vocab) noexcept {
        Var<uint32_t> qid = dispatch_x();
        Var<uint32_t> qlen = qlen_buf->read(qid);
        Var<uint32_t> dlen = draft_len_buf->read(qid);
        auto read_corpus = [&](Var<uint32_t> i) noexcept -> Var<uint32_t> {
            return lib_buf->read(i);
        };
        // token i of the (query ++ draft) sequence
        auto read_seq = [&](Var<uint32_t> i) noexcept -> Var<uint32_t> {
            Var<uint32_t> tok = 0u;
            $if (i < qlen) {
                tok = query_buf->read(qid * query_stride + i);
            }
            $else {
                tok = draft_buf->read(qid * k + i - qlen);
            };
            return tok;
        };
        Var<float> acc = 0.0f;
        $for (j, 0u, dlen) {
            // predicting draft[j]: the context is the trailing
            // n = min(max_n, qlen + j) tokens of the sequence so far
            Var<uint32_t> n = min(max_n, qlen + j);
            Var<uint32_t> start = qlen + j - n;// context start in the sequence
            auto read_window = [&](Var<uint32_t> t) noexcept -> Var<uint32_t> {
                return read_seq(start + t);
            };
            // d = count(ctx), c = count(ctx, w) from the count table
            Var<uint32_t> d = detail::lookup_count(
                keys_buf, pos_buf, counts_buf, read_window, read_corpus,
                n, cap_log2);
            Var<uint32_t> c = detail::lookup_count(
                keys_buf, pos_buf, counts_buf, read_window, read_corpus,
                n + 1u, cap_log2);
            // Add-k MLE; an unseen context floors at 1 / V
            Var<float> p;
            $if (d == 0u) {
                p = 1.0f / vocab;
            }
            $else {
                p = (cast<float>(c) + add_k) / (cast<float>(d) + add_k * vocab);
            };
            acc += log2(p);
        };
        out_buf->write(qid, acc);
    };
}

RetrieveKernelMle make_retrieve_parallel_mle_kernel(uint32_t block_size,
                                                    uint32_t cap_log2) noexcept {
    return [block_size, cap_log2](BufferUInt lib_buf, Var<uint32_t> lib_len,
                                  BufferUInt query_buf, BufferUInt qlen_buf,
                                  Var<uint32_t> query_stride,
                                  BufferUInt draft_buf, BufferUInt draft_len_buf,
                                  Var<uint32_t> min_n, Var<uint32_t> max_n,
                                  Var<uint32_t> k,
                                  BufferVar<uint64_t> keys_buf, BufferUInt pos_buf,
                                  BufferUInt counts_buf,
                                  BufferUInt req_off_buf) noexcept {
        set_block_size(block_size, 1u, 1u);
        Var<uint32_t> qid = req_off_buf->read(kernel_id()) + block_id().x;
        Var<uint32_t> lane = thread_id().x;
        Var<uint32_t> qlen = qlen_buf->read(qid);

        Shared<uint32_t> s_suffix{ngram_max_suffix_tokens};
        Shared<uint32_t> s_pos{1u};
        Shared<uint32_t> s_count{1u};

        auto read_corpus = [&](Var<uint32_t> i) noexcept -> Var<uint32_t> {
            return lib_buf->read(i);
        };

        Var<uint32_t> best_n = 0u;
        Var<uint32_t> best_pos = 0u;
        Var<uint32_t> done = 0u;

        $if (lane == 0u) {
            $for (i, 0u, k) {
                draft_buf->write(qid * k + i, ngram_invalid_id);
            };
            draft_len_buf->write(qid, 0u);
        };
        sync_block();

        // longest n first: the first length with any match wins (unchanged)
        $for (nn, 0u, max_n - min_n + 1u) {
            Var<uint32_t> n = max_n - nn;
            $if (done == 0u & n <= qlen & n < lib_len) {
                // stage the query suffix q[qlen-n .. qlen) into shared memory
                $for (idx, lane, n, block_size) {
                    s_suffix.write(idx, query_buf->read(qid * query_stride + qlen - n + idx));
                };
                $if (lane == 0u) {
                    s_pos.write(0u, ngram_invalid_id);
                    s_count.write(0u, 0u);
                };
                sync_block();
                // strided scan: thread `lane` checks positions lane, lane+bs, ...
                Var<bool> mine = false;
                Var<uint32_t> my_pos = 0u;
                Var<uint32_t> my_count = 0u;
                $for (p, lane, lib_len - n, block_size) {
                    $if (mine == false) {
                        // first-token filter (same as the parallel kernel)
                        $if (lib_buf->read(p) == s_suffix.read(0)) {
                            Var<bool> match = true;
                            $for (j, 1u, n) {
                                $if (match) {
                                    match = lib_buf->read(p + j) == s_suffix.read(j);
                                };
                            };
                            $if (match) {
                                mine = true;
                                my_pos = p;
                                // continuation count: the (n+1)-gram at p
                                // (always indexed: p + n < lib_len)
                                auto read_window = [&](Var<uint32_t> t) noexcept -> Var<uint32_t> {
                                    return lib_buf->read(p + t);
                                };
                                my_count = detail::lookup_count(
                                    keys_buf, pos_buf, counts_buf,
                                    read_window, read_corpus,
                                    n + 1u, cap_log2);
                            };
                        };
                    };
                };
                // block reduction on (count desc, position asc), two passes
                // of portable uint32 shared-memory atomics
                $if (mine) {
                    s_count.atomic(0u).fetch_max(my_count);
                };
                sync_block();
                Var<uint32_t> top = s_count.read(0u);
                $if (mine & my_count == top) {
                    s_pos.atomic(0u).fetch_min(my_pos);
                };
                sync_block();
                Var<uint32_t> pos = s_pos.read(0u);
                $if (pos != ngram_invalid_id) {
                    done = 1u;
                    best_n = n;
                    best_pos = pos;
                };
            };
        };

        // lane 0 extracts the draft tokens (identical to the parallel kernel)
        $if (done != 0u & lane == 0u) {
            Var<uint32_t> start = best_pos + best_n;
            Var<uint32_t> avail = lib_len - start;// >= 1 by construction
            Var<uint32_t> cnt = min(k, avail);
            $for (i, 0u, cnt) {
                draft_buf->write(qid * k + i, lib_buf->read(start + i));
            };
            draft_len_buf->write(qid, cnt);
        };
    };
}

LmCountKernel make_lm_count_kernel(uint32_t cap_log2) noexcept {
    return [cap_log2](BufferUInt corpus_buf, Var<uint32_t> len,
                      BufferVar<uint64_t> keys_buf, BufferUInt pos_buf,
                      BufferUInt counts_buf,
                      Var<uint32_t> order, Var<uint32_t> bos_id) noexcept {
        Var<uint32_t> p = dispatch_x();
        auto read_corpus = [&](Var<uint32_t> i) noexcept -> Var<uint32_t> {
            return corpus_buf->read(i);
        };
        $for (n, 1u, order + 1u) {
            // EOS is a legitimate final token: windows satisfy p + n <= len
            $if (p + n <= len) {
                // skip windows containing a BOS at offset > 0: exactly the
                // cross-sentence windows (BOS only follows EOS)
                Var<bool> has_bos = false;
                $for (j, 1u, n) {
                    has_bos = has_bos | (corpus_buf->read(p + j) == bos_id);
                };
                $if (has_bos == false) {
                    auto read_window = [&](Var<uint32_t> j) noexcept -> Var<uint32_t> {
                        return corpus_buf->read(p + j);
                    };
                    Var<uint64_t> key = detail::fnv1a64_tokens_dsl(read_window, n);
                    Var<uint32_t> slot = detail::probe_count_index(
                        keys_buf, pos_buf, read_window, read_corpus,
                        n, key, cap_log2);
                    $if (slot != ngram_invalid_id) {
                        counts_buf->atomic(slot).fetch_add(1u);
                    };
                };
            };
        };
    };
}

LmScoreKernel make_lm_score_kernel(uint32_t cap_log2) noexcept {
    return [cap_log2](BufferUInt corpus_buf,
                      BufferVar<uint64_t> keys_buf, BufferUInt pos_buf,
                      BufferUInt counts_buf,
                      BufferUInt rows_buf, BufferUInt row_lens_buf,
                      Var<uint32_t> stride,
                      BufferFloat out_buf,
                      Var<uint32_t> order, Var<uint32_t> smoothing,
                      Var<float> add_k, Var<float> vocab,
                      Var<uint32_t> total_tokens) noexcept {
        Var<uint32_t> row = dispatch_x();
        Var<uint32_t> len = row_lens_buf->read(row);
        auto read_corpus = [&](Var<uint32_t> i) noexcept -> Var<uint32_t> {
            return corpus_buf->read(i);
        };
        auto read_row = [&](Var<uint32_t> i) noexcept -> Var<uint32_t> {
            return rows_buf->read(row * stride + i);
        };
        // count of the window row[start .. start + n) probed from the table
        auto lookup = [&](Var<uint32_t> start, Var<uint32_t> n) noexcept -> Var<uint32_t> {
            auto read_window = [&](Var<uint32_t> t) noexcept -> Var<uint32_t> {
                return read_row(start + t);
            };
            return detail::lookup_count(keys_buf, pos_buf, counts_buf,
                                        read_window, read_corpus,
                                        n, cap_log2);
        };
        Var<float> acc = 0.0f;
        // position j predicts row[j] from the preceding context; the context
        // length is n = min(order - 1, j), so count(ctx, w) queries an
        // n-gram of length <= order (present in the table) and order == 1
        // degenerates to the unigram model. Scoring stops after predicting
        // EOS, so a skipped (EOS, BOS) window never affects a queried
        // denominator.
        $for (j, 1u, len) {
            Var<uint32_t> n_max = min(order - 1u, j);
            Var<float> p = 0.0f;
            $if (smoothing == 0u) {
                // Add-k: unseen context gives (0 + k) / (0 + k * V) == 1 / V;
                // the empty context (unigram model) has count == total_tokens
                Var<uint32_t> c = lookup(j - n_max, n_max + 1u);
                Var<uint32_t> d;
                $if (n_max == 0u) {
                    d = total_tokens;
                }
                $else {
                    d = lookup(j - n_max, n_max);
                };
                p = (cast<float>(c) + add_k) / (cast<float>(d) + add_k * vocab);
            }
            $else {
                // simple backoff: the highest order with count(ctx, w) > 0
                // wins; floor: unigram probability count(w) / total_tokens
                Var<uint32_t> backed = 0u;
                $for (nn, 0u, n_max) {
                    Var<uint32_t> n = n_max - nn;
                    $if (backed == 0u) {
                        Var<uint32_t> c = lookup(j - n, n + 1u);
                        $if (c > 0u) {
                            Var<uint32_t> d = lookup(j - n, n);
                            p = cast<float>(c) / cast<float>(d);
                            backed = 1u;
                        };
                    };
                };
                $if (backed == 0u) {
                    Var<uint32_t> cw = lookup(j, 1u);
                    p = cast<float>(cw) / cast<float>(total_tokens);
                };
            };
            acc += log2(p);
        };
        out_buf->write(row, acc);
    };
}

}// namespace tokenize
