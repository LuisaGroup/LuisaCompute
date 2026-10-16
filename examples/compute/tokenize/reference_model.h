#pragma once

// Host-side reference implementation of the LM-mode n-gram language model:
// preprocessing with <s>/</s>/<UNK> markers (the padded corpus is built by
// NgramTrainer), counting / MLE, smoothing (Add-k, simple backoff) and
// perplexity evaluation. Mirrors the reference Python BigramModel example
// (corpus "我 爱 北京 / 我 爱 学习 / 你 爱 北京", P(爱|我), smoothed
// P(学习|你), sentence_prob, perplexity). Test/demo oracle ONLY -- never
// part of the device path or the retrieval path.

#include "ngram_library.h"
#include "reference.h"

#include <algorithm>
#include <cmath>

namespace tokenize {

class ReferenceNgramModel {
public:
    ReferenceNgramModel() noexcept = default;

    // Train over the padded corpus (BOS w1..wL EOS per sentence): count all
    // windows of length n in [1, order] with p + n <= len (EOS is a
    // legitimate final token), skipping any window that contains a BOS at
    // offset > 0 -- exactly the device LM count kernel's rule.
    void train(luisa::span<const uint32_t> padded, uint32_t order,
               uint32_t bos_id) {
        _counts.clear();
        _order = order;
        const uint32_t len = static_cast<uint32_t>(padded.size());
        for (uint32_t n = 1u; n <= order; ++n) {
            if (n > len) break;
            for (uint32_t p = 0; p + n <= len; ++p) {
                bool has_bos = false;
                for (uint32_t j = 1u; j < n; ++j) {
                    has_bos |= padded[p + j] == bos_id;
                }
                if (!has_bos) {
                    ++_counts[detail::ngram_bytes_key(padded.data() + p, n)];
                }
            }
        }
        _total_tokens = len;
    }

    [[nodiscard]] uint32_t order() const noexcept { return _order; }
    [[nodiscard]] uint32_t total_tokens() const noexcept { return _total_tokens; }
    [[nodiscard]] const NgramCountMap &counts() const noexcept { return _counts; }

    // Occurrence count of an n-gram (0 when absent). `order` is the maximum
    // n-gram length (order == 2 is the bigram model of the Python example).
    [[nodiscard]] uint32_t count(luisa::span<const uint32_t> ngram) const noexcept {
        return reference_count(_counts, ngram);
    }

    // Sentence log2 probability: position j in [1, len) predicts row[j] from
    // context length n = min(order - 1, j) (so count(ctx, w) queries an
    // n-gram of length <= order; order == 1 degenerates to the unigram
    // model with the empty context counted as total_tokens).
    //   add_k:   p = (count(ctx, w) + k) / (count(ctx) + k * V)
    //   backoff: highest order with count(ctx, w) > 0, p = c / d; final floor
    //            is the unigram probability count(w) / total_tokens.
    // V is passed explicitly (vocab + 3 for the padded corpus, matching the
    // Python example).
    [[nodiscard]] float sentence_log2prob(luisa::span<const uint32_t> row,
                                          NgramSmoothing smoothing,
                                          float k, float vocab) const noexcept {
        float acc = 0.0f;
        const uint32_t len = static_cast<uint32_t>(row.size());
        for (uint32_t j = 1u; j < len; ++j) {
            const uint32_t n_max = std::min(_order - 1u, j);
            float p = 0.0f;
            if (smoothing == NgramSmoothing::add_k) {
                const uint32_t c = count(row.subspan(j - n_max, n_max + 1u));
                const uint32_t d = n_max == 0u
                                       ? _total_tokens
                                       : count(row.subspan(j - n_max, n_max));
                p = (static_cast<float>(c) + k) /
                    (static_cast<float>(d) + k * vocab);
            } else {
                bool backed = false;
                for (uint32_t n = n_max; n >= 1u; --n) {
                    const uint32_t c = count(row.subspan(j - n, n + 1u));
                    if (c > 0u) {
                        const uint32_t d = count(row.subspan(j - n, n));
                        p = static_cast<float>(c) / static_cast<float>(d);
                        backed = true;
                        break;
                    }
                }
                if (!backed) {
                    p = static_cast<float>(count(row.subspan(j, 1u))) /
                        static_cast<float>(_total_tokens);
                }
            }
            acc += std::log2(p);
        }
        return acc;
    }

    // Perplexity of a batch of rows: PPL = 2^( -sum(log2p) / sum(m) ) with
    // m = sum(len_row - 1) predicted tokens.
    [[nodiscard]] double perplexity(
        luisa::span<const luisa::vector<uint32_t>> rows,
        NgramSmoothing smoothing, float k, float vocab) const noexcept {
        double sum = 0.0;
        uint64_t m = 0;
        for (auto &row : rows) {
            sum += static_cast<double>(
                sentence_log2prob(luisa::span{row}, smoothing, k, vocab));
            m += row.size() - 1u;
        }
        return std::exp2(-sum / static_cast<double>(m));
    }

private:
    NgramCountMap _counts;
    uint32_t _order = 0;
    uint32_t _total_tokens = 0;
};

}// namespace tokenize
