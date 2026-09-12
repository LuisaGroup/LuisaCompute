#pragma once

// Host-side reference implementations of the ported vLLM n-gram retrieval
// semantics. Used ONLY as the test/benchmark oracle for the DSL kernels;
// never part of the retrieval path itself.

#include <luisa/core/stl.h>

#include <algorithm>

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

}// namespace tokenize
