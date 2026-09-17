#pragma once

// Host-side reference implementations of the ported vLLM n-gram retrieval
// semantics. Used ONLY as the test/benchmark oracle for the DSL kernels;
// never part of the retrieval path itself.

#include "ngram_library.h"

#include <luisa/core/stl.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace tokenize {

// Literal implementation of the retrieval semantics (see ngram_kernels.h):
// for each n in [min_n, max_n] (longest first) find the earliest corpus
// position p with p + n < lib_len such that corpus[p..p+n) equals the
// query's trailing n-gram; draft up to k tokens right after the match,
// clamped to the library end. Returns an empty vector on no match.
[[nodiscard]] inline luisa::vector<uint32_t> reference_retrieve(
    luisa::span<const uint32_t> corpus,
    luisa::span<const uint32_t> query,
    uint32_t min_n, uint32_t max_n, uint32_t k) noexcept {
    luisa::vector<uint32_t> draft;
    const auto lib_len = corpus.size();
    const auto qlen = query.size();
    if (k == 0 || qlen < min_n || lib_len == 0) return draft;
    for (uint32_t n = max_n;; --n) {
        if (n <= qlen && n < lib_len) {
            // earliest p with p + n < lib_len
            for (uint32_t p = 0; p + n < lib_len; ++p) {
                bool match = true;
                for (uint32_t j = 0; j < n; ++j) {
                    if (corpus[p + j] != query[qlen - n + j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    const uint32_t start = p + n;// >= 1 continuation by p + n < lib_len
                    const uint32_t cnt = std::min<uint32_t>(k, static_cast<uint32_t>(lib_len - start));
                    draft.assign(corpus.begin() + start, corpus.begin() + start + cnt);
                    return draft;
                }
            }
        }
        if (n == min_n) break;
    }
    return draft;
}

// Faithful C++ port of vLLM's CPU matcher
// (vllm/v1/spec_decode/ngram_proposer.py::
//  _find_longest_matched_ngram_and_propose_tokens): reversed-sequence
// single-pass KMP/LPS scan with the lps array capped at max_n.
//
// Self-retrieval form: `tokens` is simultaneously the library and the
// query, exactly like vLLM where the request's own context is searched.
[[nodiscard]] inline luisa::vector<uint32_t> reference_retrieve_vllm_kmp(
    luisa::span<const uint32_t> tokens,
    uint32_t min_n, uint32_t max_n, uint32_t k) noexcept {
    luisa::vector<uint32_t> draft;
    const uint32_t total = static_cast<uint32_t>(tokens.size());
    if (total < min_n || k == 0) return draft;

    luisa::vector<uint32_t> rev(tokens.rbegin(), tokens.rend());
    luisa::vector<uint32_t> lps(max_n, 0u);

    uint32_t longest_ngram = 0u;
    uint32_t position = 0u;
    uint32_t prev_lps = 0u;
    uint32_t i = 1u;
    while (i < total) {
        if (rev[prev_lps] == rev[i]) {
            prev_lps += 1u;
            // >= (not >): among equal lengths keep the latest reversed
            // position, i.e. the earliest original position.
            if (prev_lps >= longest_ngram) {
                longest_ngram = prev_lps;
                position = i;
            }
            if (i < max_n) lps[i] = prev_lps;
            if (prev_lps == max_n) {
                // do not match n-grams longer than max_n
                prev_lps = lps[max_n - 1u];
            }
            i += 1u;
        } else if (prev_lps != 0u) {
            prev_lps = lps[prev_lps - 1u];
        } else {
            i += 1u;
        }
    }

    if (longest_ngram < min_n) return draft;

    const uint32_t start_position = total - 1u - position + longest_ngram;
    const uint32_t cnt = std::min<uint32_t>(k, total - start_position);
    draft.assign(tokens.begin() + start_position, tokens.begin() + start_position + cnt);
    return draft;
}

// ---------- training oracles (test/benchmark only) ----------

// Host ground-truth n-gram counts, keyed by the raw little-endian bytes of
// the token window. Mirrors the device counting kernel exactly: for every
// position p and n in [min_n, max_n] with p + n < lib_len, count the length-n
// prefix window; additionally count the length-(max_n+1) continuation window
// at every position with p + max_n < lib_len (its last token may be the
// corpus's final token). Each window is counted once per qualifying position.
using NgramCountMap = luisa::unordered_map<luisa::string, uint32_t>;

namespace detail {

[[nodiscard]] inline luisa::string ngram_bytes_key(const uint32_t *tokens,
                                                   uint32_t n) noexcept {
    luisa::string key;
    key.resize(n * sizeof(uint32_t));
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t t = tokens[i]; // explicit little-endian byte order
        for (uint32_t b = 0; b < 4; ++b) {
            key[i * 4 + b] = static_cast<char>((t >> (b * 8)) & 0xFFu);
        }
    }
    return key;
}

}// namespace detail

[[nodiscard]] inline NgramCountMap reference_count_ngrams(
    luisa::span<const uint32_t> corpus, uint32_t min_n, uint32_t max_n) {
    NgramCountMap counts;
    const uint32_t lib_len = static_cast<uint32_t>(corpus.size());
    for (uint32_t n = min_n; n <= max_n; ++n) {
        if (n >= lib_len) break;
        for (uint32_t p = 0; p + n < lib_len; ++p) {
            ++counts[detail::ngram_bytes_key(corpus.data() + p, n)];
        }
    }
    if (max_n < lib_len) {
        for (uint32_t p = 0; p + max_n < lib_len; ++p) {
            ++counts[detail::ngram_bytes_key(corpus.data() + p, max_n + 1u)];
        }
    }
    return counts;
}

[[nodiscard]] inline uint32_t reference_count(const NgramCountMap &counts,
                                              luisa::span<const uint32_t> ngram) noexcept {
    if (ngram.empty()) return 0u;
    auto it = counts.find(detail::ngram_bytes_key(
        ngram.data(), static_cast<uint32_t>(ngram.size())));
    return it == counts.end() ? 0u : it->second;
}

// MLE continuation oracle: same longest-match-first search as
// reference_retrieve, but among all matches of the winning length the draft
// is proposed from the position whose CONTINUATION (the (n+1)-gram at the
// match position) is the most frequent in the corpus -- the maximum
// likelihood estimate argmax_w count(ctx, w). Ties on count resolve to the
// earliest position. `counts` must come from reference_count_ngrams(corpus,
// min_n, max_n). Returns an empty vector on no match.
[[nodiscard]] inline luisa::vector<uint32_t> reference_retrieve_mle(
    luisa::span<const uint32_t> corpus,
    const NgramCountMap &counts,
    luisa::span<const uint32_t> query,
    uint32_t min_n, uint32_t max_n, uint32_t k) noexcept {
    luisa::vector<uint32_t> draft;
    const auto lib_len = corpus.size();
    const auto qlen = query.size();
    if (k == 0 || qlen < min_n || lib_len == 0) return draft;
    for (uint32_t n = max_n;; --n) {
        if (n <= qlen && n < lib_len) {
            uint32_t best_pos = ngram_invalid_id;
            uint32_t best_count = 0u;
            for (uint32_t p = 0; p + n < lib_len; ++p) {
                bool match = true;
                for (uint32_t j = 0; j < n; ++j) {
                    if (corpus[p + j] != query[qlen - n + j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    // continuation count of the (n+1)-gram at p
                    const uint32_t c = reference_count(
                        counts, luisa::span{corpus.data() + p, n + 1u});
                    // (count desc, position asc): strictly greater count wins
                    if (best_pos == ngram_invalid_id || c > best_count) {
                        best_pos = p;
                        best_count = c;
                    }
                }
            }
            if (best_pos != ngram_invalid_id) {
                const uint32_t start = best_pos + n;
                const uint32_t cnt = std::min<uint32_t>(
                    k, static_cast<uint32_t>(lib_len - start));
                draft.assign(corpus.begin() + start, corpus.begin() + start + cnt);
                return draft;
            }
        }
        if (n == min_n) break;
    }
    return draft;
}

// Add-k draft-scoring oracle: mirrors the device draft-score kernel exactly.
// For each draft position j the context is the trailing min(max_n, qlen + j)
// tokens of the (query ++ draft) sequence and w is draft[j];
//   p = (count(ctx, w) + k) / (count(ctx) + k * V),  unseen context -> 1 / V.
// Returns the sum of log2(p) over the draft (0 for an empty draft).
[[nodiscard]] inline float reference_draft_log2prob(
    const NgramCountMap &counts, uint32_t vocab_size,
    luisa::span<const uint32_t> query, luisa::span<const uint32_t> draft,
    uint32_t max_n, float add_k) noexcept {
    float acc = 0.0f;
    const uint32_t qlen = static_cast<uint32_t>(query.size());
    const float vocab = static_cast<float>(vocab_size);
    // token i of the (query ++ draft) sequence
    auto seq = [&](uint32_t i) noexcept {
        return i < qlen ? query[i] : draft[i - qlen];
    };
    luisa::vector<uint32_t> win;
    for (uint32_t j = 0; j < static_cast<uint32_t>(draft.size()); ++j) {
        const uint32_t n = std::min(max_n, qlen + j);
        const uint32_t start = qlen + j - n;
        win.resize(n + 1u);
        for (uint32_t t = 0; t <= n; ++t) win[t] = seq(start + t);
        const uint32_t d = reference_count(counts, luisa::span{win.data(), n});
        const uint32_t c = reference_count(counts, luisa::span{win});
        const float p = d == 0u ? 1.0f / vocab
                                : (static_cast<float>(c) + add_k) /
                                      (static_cast<float>(d) + add_k * vocab);
        acc += std::log2(p);
    }
    return acc;
}

}// namespace tokenize
