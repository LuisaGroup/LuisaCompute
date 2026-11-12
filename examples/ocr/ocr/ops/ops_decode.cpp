// OCR engine -- decode / embedding / sampling operator implementations
// (Phase 2 kernels + Phase 3 KV append / deterministic host sampling).
#include "ocr/ops/ops_decode.h"

#include <algorithm>
#include <cmath>

#include <luisa/core/stl/format.h>
#include <luisa/dsl/sugar.h>

#include "ocr/ops/ops_common.h"
#include "ocr/quantization.h"
#include "ocr/runtime/shader_cache.h"

namespace luisa::ocr {

using namespace luisa::compute;

[[nodiscard]] EmbeddingLookupKernel make_embedding_lookup_kernel() noexcept {
    EmbeddingLookupKernel kernel = [](BufferFloat table, BufferInt ids, BufferFloat out,
                                      Var<uint> num_ids, Var<uint> hidden) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_embedding_lookup");
        const auto id = ocr_linear_id();
        $if (id < num_ids * hidden) {
            const auto t = id / hidden;
            const auto d = id % hidden;
            const auto row = cast<uint>(ids.read(t));
            out.write(id, table.read(row * hidden + d));
        };
    };
    return kernel;
}

[[nodiscard]] MmScatterKernel make_mm_scatter_kernel() noexcept {
    MmScatterKernel kernel = [](BufferFloat embeddings, BufferFloat features, BufferFloat out,
                                Var<uint> seq, Var<uint> hidden, Var<uint> start, Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_mm_scatter");
        const auto id = ocr_linear_id();
        $if (id < seq * hidden) {
            const auto t = id / hidden;
            const auto d = id % hidden;
            Var<float> value = embeddings.read(id);
            $if (t >= start & t < start + count) {
                value = features.read((t - start) * hidden + d);
            };
            out.write(id, value);
        };
    };
    return kernel;
}

[[nodiscard]] ArgmaxKernel make_argmax_kernel() noexcept {
    ArgmaxKernel kernel = [](BufferFloat logits, BufferInt out, Var<uint> rows, Var<uint> cols) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_argmax");
        const auto row = ocr_linear_id();
        $if (row < rows) {
            const auto base = row * cols;
            Var<uint> best = def(0u);
            Var<float> best_v = def(logits.read(base));
            $for (c, 1u, cols) {
                const auto v = logits.read(base + c);
                $if (v > best_v) {
                    best_v = v;
                    best = c;
                };
            };
            out.write(row, cast<int>(best));
        };
    };
    return kernel;
}

[[nodiscard]] SoftmaxKernel make_softmax_kernel() noexcept {
    SoftmaxKernel kernel = [](BufferFloat logits, BufferFloat out, Var<uint> rows, Var<uint> cols,
                              Var<float> temperature) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_softmax");
        const auto row = ocr_linear_id();
        $if (row < rows) {
            const auto base = row * cols;
            // S5 defensive: invalid temperature -> defined zero output, no NaN.
            $if (temperature > 0.0f) {
                Var<float> max_v = def(-3.0e38f);
                $for (c, cols) { max_v = max(max_v, logits.read(base + c) / temperature); };
                Var<float> sum = def(0.0f);
                $for (c, cols) { sum += exp(logits.read(base + c) / temperature - max_v); };
                $for (c, cols) {
                    out.write(base + c, exp(logits.read(base + c) / temperature - max_v) / sum);
                };
            } $else {
                $for (c, cols) { out.write(base + c, 0.0f); };
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<LaunchPlan1D> plan_embedding_dispatch(uint64_t num_ids, uint64_t hidden) {
    uint64_t count = 0;
    if (!checked_mul(num_ids, hidden, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "embedding count overflow", "num_ids*hidden");
    }
    return plan_threads(count, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_mm_scatter_dispatch(uint64_t seq, uint64_t hidden) {
    uint64_t count = 0;
    if (!checked_mul(seq, hidden, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "mm scatter count overflow", "seq*hidden");
    }
    return plan_threads(count, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_argmax_dispatch(uint64_t rows) {
    return plan_threads(rows, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_softmax_dispatch(uint64_t rows) {
    return plan_threads(rows, 256u);
}

[[nodiscard]] Status validate_softmax_inputs(uint64_t rows, uint64_t cols, float temperature) {
    if (rows == 0u || cols == 0u) {
        return Status::error(StatusCode::invalid_argument, "softmax dimensions must be nonzero", "dims");
    }
    uint64_t count = 0;
    if (!checked_mul(rows, cols, count) || count > UINT32_MAX) {
        return Status::error(StatusCode::invalid_argument, "softmax element count exceeds uint32", "dims");
    }
    if (!std::isfinite(temperature) || temperature <= 0.0f) {
        return Status::error(StatusCode::invalid_argument,
                             "temperature must be finite and positive", "temperature");
    }
    return Status::success();
}

[[nodiscard]] luisa::string decode_variant_key(luisa::string_view op) {
    return make_shader_key("decode", op);
}

// ---------------------------------------------------------------------------
// Phase 3: contiguous-KV append.
// ---------------------------------------------------------------------------

[[nodiscard]] KvAppendKernel make_kv_append_kernel() noexcept {
    KvAppendKernel kernel = [](BufferFloat src_k, BufferFloat src_v, BufferFloat dst_k, BufferFloat dst_v,
                               Var<uint> count, Var<uint> kv_dim, Var<uint> start) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_kv_append");
        const auto id = ocr_linear_id();
        $if (id < count * kv_dim) {
            const auto t = id / kv_dim;
            const auto d = id % kv_dim;
            const auto dst = (start + t) * kv_dim + d;
            dst_k.write(dst, src_k.read(id));
            dst_v.write(dst, src_v.read(id));
        };
    };
    return kernel;
}

[[nodiscard]] Result<LaunchPlan1D> plan_kv_append_dispatch(uint64_t count, uint64_t kv_dim) {
    uint64_t total = 0;
    if (!checked_mul(count, kv_dim, total)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument,
                                             "KV append element count overflow", "count*kv_dim");
    }
    return plan_threads(total, 256u);
}

[[nodiscard]] Status validate_kv_append(uint64_t start, uint64_t count, uint64_t capacity) {
    uint64_t end = 0;
    if (!checked_add(start, count, end) || end > capacity) {
        return Status::error(StatusCode::out_of_memory,
                             luisa::format("KV append [{}, {}) overruns token capacity {}", start, end, capacity),
                             "capacity");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Phase 3: host deterministic sampling building blocks.
// ---------------------------------------------------------------------------

[[nodiscard]] Status DecodeSamplingConfig::validate(uint32_t vocab_size) const {
    return validate_decode_sampling(*this, vocab_size);
}

[[nodiscard]] Status validate_decode_sampling(const DecodeSamplingConfig &config, uint32_t vocab_size) {
    if (vocab_size == 0u) {
        return Status::error(StatusCode::invalid_argument, "sampling vocabulary must be nonzero", "vocab_size");
    }
    if (!config.greedy) {
        if (!std::isfinite(config.temperature) || config.temperature <= 0.0f) {
            return Status::error(StatusCode::invalid_argument,
                                 "sampling temperature must be finite and positive (greedy ignores it)",
                                 "temperature");
        }
        if (config.top_k > vocab_size) {
            return Status::error(StatusCode::invalid_argument,
                                 "top_k exceeds the vocabulary size", "top_k");
        }
        if (!std::isfinite(config.top_p) || config.top_p <= 0.0f || config.top_p > 1.0f) {
            return Status::error(StatusCode::invalid_argument,
                                 "top_p must be in (0, 1]", "top_p");
        }
    }
    if (!std::isfinite(config.repetition_penalty) || config.repetition_penalty <= 0.0f) {
        return Status::error(StatusCode::invalid_argument,
                             "repetition_penalty must be finite and positive", "repetition_penalty");
    }
    if (config.no_repeat_ngram_size > vocab_size) {
        return Status::error(StatusCode::invalid_argument,
                             "no_repeat_ngram_size is larger than the vocabulary", "no_repeat_ngram_size");
    }
    return Status::success();
}

[[nodiscard]] uint64_t decode_splitmix64(uint64_t &state) noexcept {
    state += 0x9E3779B97F4A7C15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30u)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27u)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31u);
}

DeterministicSampler::DeterministicSampler(DecodeSamplingConfig config, uint32_t vocab_size,
                                           luisa::span<const uint32_t> prompt)
    : _config(config), _vocab_size(vocab_size), _rng_state(config.seed) {
    _history.assign(prompt.begin(), prompt.end());
}

void DeterministicSampler::observe(uint32_t token) {
    _history.push_back(token);
}

[[nodiscard]] Result<uint32_t> DeterministicSampler::sample(luisa::span<const float> logits) {
    using R = Result<uint32_t>;
    if (logits.size() != _vocab_size) {
        return R::failure(StatusCode::invalid_argument,
                          "sampler logits row size must equal the vocabulary", "logits");
    }
    auto valid = validate_decode_sampling(_config, _vocab_size);
    if (!valid.ok()) { return R::failure(valid); }

    // Working copy: processors mutate logits before selection.
    luisa::vector<float> work(logits.begin(), logits.end());

    // HF-style repetition penalty over the full observed history.
    if (_config.repetition_penalty != 1.0f && !_history.empty()) {
        luisa::vector<bool> seen(_vocab_size, false);
        for (auto id : _history) {
            if (id < _vocab_size) { seen[id] = true; }
        }
        for (uint32_t i = 0; i < _vocab_size; i++) {
            if (!seen[i]) { continue; }
            work[i] = work[i] > 0.0f ? work[i] / _config.repetition_penalty
                                     : work[i] * _config.repetition_penalty;
        }
    }

    // No-repeat-ngram: ban any token that would complete an n-gram already
    // present in the history. Matches HF generate's _get_ngrams semantics:
    // the last (n-1) tokens form the prefix; any history occurrence of that
    // prefix followed by token t bans t.
    if (_config.no_repeat_ngram_size >= 2u &&
        _history.size() >= _config.no_repeat_ngram_size) {
        const size_t n = _config.no_repeat_ngram_size;
        const size_t prefix_len = n - 1u;
        const size_t hist = _history.size();
        // Compare the trailing prefix against every earlier window.
        for (size_t start = 0; start + n <= hist; start++) {
            bool match = true;
            for (size_t j = 0; j < prefix_len; j++) {
                if (_history[start + j] != _history[hist - prefix_len + j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                const uint32_t banned = _history[start + prefix_len];
                if (banned < _vocab_size) { work[banned] = -std::numeric_limits<float>::infinity(); }
            }
        }
    }

    // Selectable entries are FINITE logits. -inf is an allowed masked/banned
    // entry; NaN and +inf are invalid logits (never silently selectable). At
    // least one finite selectable logit is required, otherwise this is a clear
    // error -- never a silent token 0 or a banned token. The RNG is NOT
    // advanced on any failure path (only on a successful non-greedy draw).
    const auto selectable = [&work](uint32_t i) { return std::isfinite(work[i]); };

    if (_config.greedy) {
        uint32_t best = 0u;
        bool have = false;
        float best_v = 0.0f;
        for (uint32_t i = 0; i < _vocab_size; i++) {
            if (!selectable(i)) { continue; }
            if (!have || work[i] > best_v) {
                best_v = work[i];
                best = i;
                have = true;
            }
        }
        if (!have) {
            return R::failure(StatusCode::invalid_argument,
                              "greedy selection has no finite selectable logit (all NaN/+inf/banned)", "logits");
        }
        observe(best);
        return R::success(best);
    }

    // Temperature scaling (finite entries stay finite; -inf stays -inf).
    for (auto &v : work) { v /= _config.temperature; }

    // Stable descending sort by (logit, index): deterministic tie order.
    // Finite entries sort before -inf; NaN/+inf are excluded from selection.
    luisa::vector<uint32_t> order(_vocab_size);
    for (uint32_t i = 0; i < _vocab_size; i++) { order[i] = i; }
    std::stable_sort(order.begin(), order.end(), [&work](uint32_t a, uint32_t b) {
        return work[a] > work[b];
    });

    // Candidate pool: finite entries only, in sorted order.
    luisa::vector<uint32_t> finite;
    finite.reserve(_vocab_size);
    for (uint32_t i = 0; i < _vocab_size; i++) {
        if (selectable(order[i])) { finite.push_back(order[i]); }
    }
    if (finite.empty()) {
        return R::failure(StatusCode::invalid_argument,
                          "sampling has no finite selectable logit (all NaN/+inf/banned)", "logits");
    }

    // Top-k truncation (0 = disabled) over the finite pool.
    size_t kept = finite.size();
    if (_config.top_k != 0u) { kept = std::min<size_t>(kept, _config.top_k); }

    // Softmax over the kept prefix (max-subtracted, double accumulate). All
    // kept entries are finite, so sum is finite and > 0.
    const float max_v = work[finite[0]];
    luisa::vector<double> probs(kept, 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < kept; i++) {
        const double e = std::exp(static_cast<double>(work[finite[i]]) - static_cast<double>(max_v));
        probs[i] = e;
        sum += e;
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        return R::failure(StatusCode::internal_error,
                          "sampling softmax produced a non-finite normalization", "logits");
    }
    for (auto &p : probs) { p /= sum; }

    // Top-p (nucleus): keep the smallest prefix whose cumulative probability
    // reaches top_p; always keep at least one token.
    if (_config.top_p < 1.0f) {
        double cum = 0.0;
        size_t nucleus = kept;
        for (size_t i = 0; i < kept; i++) {
            cum += probs[i];
            if (cum >= static_cast<double>(_config.top_p)) {
                nucleus = i + 1u;
                break;
            }
        }
        kept = nucleus;
        double renorm = 0.0;
        for (size_t i = 0; i < kept; i++) { renorm += probs[i]; }
        for (size_t i = 0; i < kept; i++) { probs[i] /= renorm; }
    }

    // Categorical draw: exactly one SplitMix64 draw mapped to [0,1), advanced
    // only here on the successful non-greedy path.
    const uint64_t bits = decode_splitmix64(_rng_state);
    const double u = static_cast<double>(bits >> 11u) * (1.0 / 9007199254740992.0);// 2^-53
    double acc = 0.0;
    uint32_t chosen = finite[kept - 1u];
    for (size_t i = 0; i < kept; i++) {
        acc += probs[i];
        if (u < acc) {
            chosen = finite[i];
            break;
        }
    }
    observe(chosen);
    return R::success(chosen);
}

}// namespace luisa::ocr