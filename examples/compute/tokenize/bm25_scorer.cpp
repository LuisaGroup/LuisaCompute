#include "bm25_scorer.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <luisa/core/logging.h>
#include <luisa/core/platform.h>
#include <luisa/core/fiber.h>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/command_list.h>

using namespace luisa;
using namespace luisa::compute;

namespace tokenize {

BM25Scorer::BM25Scorer(const InvertedIndex &index, double k1, double b)
    : _index(index), _k1(k1), _b(b) {
    double avgdl = _index.avgdl();
    if (avgdl == 0.0) return;
    const auto &dl = _index.doc_lengths();
    _denom_base.resize(dl.size());
    luisa::fiber::parallel(static_cast<uint32_t>(dl.size()), [&](uint32_t i) noexcept {
        _denom_base[i] = k1 * ((1.0 - b) + (b / avgdl) * static_cast<double>(dl[i]));
    });
}

BM25Scorer::BM25Scorer(const BM25Scorer &other)
    : _index(other._index), _k1(other._k1), _b(other._b),
      _denom_base(other._denom_base), _tfs_buf(other._tfs_buf), _denom_buf(other._denom_buf) {
    // GPU state is lazily re-initialized on first use; do not copy handles
}

BM25Scorer &BM25Scorer::operator=(const BM25Scorer &other) {
    if (this != &other) {
        // _index is const ref, cannot rebind; but copy assignment is not expected to change _index
        _k1 = other._k1;
        _b = other._b;
        _denom_base = other._denom_base;
        _tfs_buf = other._tfs_buf;
        _denom_buf = other._denom_buf;
        // Reset GPU state so it is re-initialized lazily
        _gpu_ready = false;
        _gpu_device = {};
        _gpu_posting_docs = {};
        _gpu_posting_docs_ptr = {};
        _gpu_posting_tfs = {};
        _gpu_posting_tfs_ptr = {};
        _gpu_denom_base = {};
        _cpu_posting_docs_ptr.clear();
        _cpu_posting_tfs_ptr.clear();
        _gpu_accumulate_shader = {};
        _gpu_extract_nonzero_shader = {};
    }
    return *this;
}

BM25Scorer::~BM25Scorer() = default;

double BM25Scorer::idf(int df, int N) noexcept {
    return std::log(1.0 + (static_cast<double>(N) - static_cast<double>(df) + 0.5) / (static_cast<double>(df) + 0.5));
}

void BM25Scorer::_init_gpu(Device &device, Stream &stream) const {
    if (_gpu_ready) return;

    if (!device) {
        LUISA_WARNING("BM25Scorer: no GPU device provided, GPU scoring disabled");
        _gpu_ready = true;
        return;
    }

    _gpu_device = device;


    // Build flat posting arrays from public index API
    auto &term_to_id = _index.term_to_id();
    int num_terms = static_cast<int>(term_to_id.size());
    luisa::vector<luisa::string> terms_by_id(num_terms);
    for (const auto &[term, tid] : term_to_id) {
        terms_by_id[tid] = term;
    }

    luisa::vector<int> flat_docs;
    luisa::vector<uint16_t> flat_tfs;
    _cpu_posting_docs_ptr.resize(num_terms + 1, 0);
    _cpu_posting_tfs_ptr.resize(num_terms + 1, 0);
    for (int tid = 0; tid < num_terms; ++tid) {
        auto postings = _index.get_postings(terms_by_id[tid]);
        if (postings) {
            for (auto d : postings->docs) flat_docs.push_back(d);
            for (auto t : postings->tfs) flat_tfs.push_back(t);
        }
        _cpu_posting_docs_ptr[tid + 1] = static_cast<int>(flat_docs.size());
        _cpu_posting_tfs_ptr[tid + 1] = static_cast<int>(flat_tfs.size());
    }

    // Upload to GPU
    _gpu_posting_docs = _gpu_device.create_buffer<int>(flat_docs.size());
    _gpu_posting_tfs = _gpu_device.create_buffer<uint16_t>(flat_tfs.size());
    _gpu_posting_docs_ptr = _gpu_device.create_buffer<int>(_cpu_posting_docs_ptr.size());
    _gpu_posting_tfs_ptr = _gpu_device.create_buffer<int>(_cpu_posting_tfs_ptr.size());
    CommandList upload = CommandList::create();
    upload << _gpu_posting_docs.copy_from(luisa::span{flat_docs.data(), flat_docs.size()})
           << _gpu_posting_tfs.copy_from(luisa::span{flat_tfs.data(), flat_tfs.size()})
           << _gpu_posting_docs_ptr.copy_from(luisa::span{_cpu_posting_docs_ptr.data(), _cpu_posting_docs_ptr.size()})
           << _gpu_posting_tfs_ptr.copy_from(luisa::span{_cpu_posting_tfs_ptr.data(), _cpu_posting_tfs_ptr.size()});

    // Upload denom_base as float
    luisa::vector<float> denom_f(_denom_base.size());
    for (size_t i = 0; i < _denom_base.size(); ++i) denom_f[i] = static_cast<float>(_denom_base[i]);
    _gpu_denom_base = _gpu_device.create_buffer<float>(denom_f.size());
    upload << _gpu_denom_base.copy_from(luisa::span{denom_f.data(), denom_f.size()});
    stream << upload.commit();

    // Kernel 1: per-posting-entry BM25 score accumulation
    Kernel1D accumulate_kernel = [](BufferVar<int> posting_docs,
                                     BufferVar<uint16_t> posting_tfs,
                                     BufferVar<float> scores,
                                     BufferVar<float> denom_base,
                                     Var<int> start, Var<int> end,
                                     Var<float> idf_val, Var<float> k1, Var<float> q_weight) noexcept {
        $ idx = dispatch_x();
        $if (idx >= end - start) {
            return;
        };
        Var<int> doc_id = posting_docs.read(start + idx);
        Var<uint16_t> tf_raw = posting_tfs.read(start + idx);
        Var<float> tf = cast<float>(tf_raw);
        Var<float> denom = tf + denom_base.read(doc_id);
        Var<float> score = tf * idf_val * (k1 + 1.0f) * q_weight / denom;
        scores.atomic(doc_id).fetch_add(score);
    };

    // Kernel 2: extract nonzero scores with atomic counter (parallel top-k reduction)
    Kernel1D extract_nonzero_kernel = [](BufferVar<float> scores,
                                          BufferVar<int> out_docs,
                                          BufferVar<float> out_scores,
                                          BufferVar<int> counter,
                                          Var<int> N) noexcept {
        $ idx = dispatch_x();
        $if (idx >= N) {
            return;
        };
        Var<float> s = scores.read(idx);
        $if (s > 0.0f) {
            Var<int> pos = counter.atomic(0).fetch_add(1);
            out_docs.write(pos, cast<int>(idx));
            out_scores.write(pos, s);
        };
    };

    _gpu_accumulate_shader = _gpu_device.compile(accumulate_kernel);
    _gpu_extract_nonzero_shader = _gpu_device.compile(extract_nonzero_kernel);

    _gpu_ready = true;
}

luisa::vector<double> BM25Scorer::gpu_accumulate(Device &device, Stream &stream, const luisa::vector<luisa::string> &query_tokens) const {
    _init_gpu(device, stream);
    if (!_gpu_device) {
        return accumulate(query_tokens, nullptr);
    }

    int N = _index.N();
    if (N == 0 || _denom_base.empty()) {
        return luisa::vector<double>(N, 0.0);
    }

    // Count query tokens
    luisa::unordered_map<luisa::string, int> token_counts;
    for (const auto &t : query_tokens) {
        auto it = token_counts.find(t);
        if (it == token_counts.end()) token_counts.emplace(t, 1);
        else ++(it->second);
    }

    // Build per-term query metadata
    luisa::vector<int> term_ids;
    luisa::vector<float> idfs;
    luisa::vector<float> weights;
    auto &term_to_id = _index.term_to_id();
    for (const auto &[token, q_weight] : token_counts) {
        auto it = term_to_id.find(token);
        if (it == term_to_id.end()) continue;
        int tid = it->second;
        int df = _index.doc_freq(token);
        if (df == 0) continue;
        float idf_val = static_cast<float>(idf(df, N));
        term_ids.push_back(tid);
        idfs.push_back(idf_val);
        weights.push_back(static_cast<float>(q_weight));
    }

    if (term_ids.empty()) {
        return luisa::vector<double>(N, 0.0);
    }

    // Zero scores buffer and accumulate
    auto gpu_scores = _gpu_device.create_buffer<float>(N);
    luisa::vector<float> zero_scores(N, 0.0f);
    CommandList cmdlist = CommandList::create();
    cmdlist << gpu_scores.copy_from(luisa::span{zero_scores.data(), zero_scores.size()});

    float k1f = static_cast<float>(_k1);

    // Launch one dispatch per query term
    for (size_t i = 0; i < term_ids.size(); ++i) {
        int tid = term_ids[i];
        int start = _cpu_posting_docs_ptr[tid];
        int end = _cpu_posting_docs_ptr[tid + 1];
        int count = end - start;
        if (count <= 0) continue;
        cmdlist << _gpu_accumulate_shader(
            _gpu_posting_docs, _gpu_posting_tfs, gpu_scores, _gpu_denom_base,
            start, end, idfs[i], k1f, weights[i]
        ).dispatch(count);
    }
    stream << cmdlist.commit();

    // Download results
    luisa::vector<float> scores_f(N);
    stream << gpu_scores.copy_to(luisa::span{scores_f.data(), scores_f.size()}) << synchronize();

    luisa::vector<double> scores(N);
    for (int i = 0; i < N; ++i) scores[i] = static_cast<double>(scores_f[i]);
    return scores;
}

luisa::vector<std::pair<int, double>> BM25Scorer::gpu_score_topk(Device &device, Stream &stream, const luisa::vector<luisa::string> &query_tokens, int top_k) const {
    _init_gpu(device, stream);
    if (!_gpu_device) {
        return score_topk(query_tokens, top_k, nullptr);
    }

    int N = _index.N();
    if (N == 0 || _denom_base.empty() || top_k <= 0) return {};

    // Count query tokens
    luisa::unordered_map<luisa::string, int> token_counts;
    for (const auto &t : query_tokens) {
        auto it = token_counts.find(t);
        if (it == token_counts.end()) token_counts.emplace(t, 1);
        else ++(it->second);
    }

    // Build per-term query metadata
    luisa::vector<int> term_ids;
    luisa::vector<float> idfs;
    luisa::vector<float> weights;
    auto &term_to_id = _index.term_to_id();
    for (const auto &[token, q_weight] : token_counts) {
        auto it = term_to_id.find(token);
        if (it == term_to_id.end()) continue;
        int tid = it->second;
        int df = _index.doc_freq(token);
        if (df == 0) continue;
        float idf_val = static_cast<float>(idf(df, N));
        term_ids.push_back(tid);
        idfs.push_back(idf_val);
        weights.push_back(static_cast<float>(q_weight));
    }

    if (term_ids.empty()) return {};

    // Accumulate on GPU
    auto gpu_scores = _gpu_device.create_buffer<float>(N);
    luisa::vector<float> zero_scores(N, 0.0f);
    CommandList cmdlist = CommandList::create();
    cmdlist << gpu_scores.copy_from(luisa::span{zero_scores.data(), zero_scores.size()});

    float k1f = static_cast<float>(_k1);
    for (size_t i = 0; i < term_ids.size(); ++i) {
        int tid = term_ids[i];
        int start = _cpu_posting_docs_ptr[tid];
        int end = _cpu_posting_docs_ptr[tid + 1];
        int count = end - start;
        if (count <= 0) continue;
        cmdlist << _gpu_accumulate_shader(
            _gpu_posting_docs, _gpu_posting_tfs, gpu_scores, _gpu_denom_base,
            start, end, idfs[i], k1f, weights[i]
        ).dispatch(count);
    }

    // Extract nonzero candidates on GPU
    auto gpu_out_docs = _gpu_device.create_buffer<int>(N);
    auto gpu_out_scores = _gpu_device.create_buffer<float>(N);
    auto gpu_counter = _gpu_device.create_buffer<int>(1);
    int zero = 0;
    int count = 0;
    cmdlist << gpu_counter.copy_from(luisa::span{&zero, 1})
            << _gpu_extract_nonzero_shader(gpu_scores, gpu_out_docs, gpu_out_scores, gpu_counter, N).dispatch(N)
            << gpu_counter.copy_to(luisa::span{&count, 1});
    stream << cmdlist.commit() << synchronize();

    if (count == 0) return {};

    luisa::vector<int> out_docs(count);
    luisa::vector<float> out_scores_f(count);
    CommandList download = CommandList::create();
    download << gpu_out_docs.view(0, count).copy_to(luisa::span{out_docs.data(), out_docs.size()})
             << gpu_out_scores.view(0, count).copy_to(luisa::span{out_scores_f.data(), out_scores_f.size()});
    stream << download.commit() << synchronize();

    luisa::vector<std::pair<int, double>> items;
    items.reserve(count);
    for (int i = 0; i < count; ++i) {
        items.emplace_back(out_docs[i], static_cast<double>(out_scores_f[i]));
    }

    if (top_k >= static_cast<int>(items.size())) {
        std::sort(items.begin(), items.end(), [](const auto &a, const auto &b) {
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        });
        return items;
    }
    std::partial_sort(items.begin(), items.begin() + top_k, items.end(), [](const auto &a, const auto &b) {
        return a.second > b.second || (a.second == b.second && a.first < b.first);
    });
    items.resize(top_k);
    return items;
}

std::pair<luisa::vector<int>, luisa::vector<double>>
BM25Scorer::token_scores(const luisa::string &token, double q_weight, const luisa::vector<int> *candidate_sorted) const {
    auto postings_opt = _index.get_postings(token);
    if (!postings_opt) return {{}, {}};
    auto docs = postings_opt->docs;
    auto tfs = postings_opt->tfs;
    if (docs.empty()) return {{}, {}};

    luisa::vector<int> filtered_docs;
    luisa::vector<uint16_t> filtered_tfs;
    if (candidate_sorted && !candidate_sorted->empty()) {
        filtered_docs.reserve(docs.size());
        filtered_tfs.reserve(tfs.size());
        for (size_t i = 0; i < docs.size(); ++i) {
            int d = docs[i];
            bool ok = false;
            if (candidate_sorted->size() <= 256) {
                auto it = std::lower_bound(candidate_sorted->begin(), candidate_sorted->end(), d);
                if (it != candidate_sorted->end() && *it == d) ok = true;
            } else {
                ok = std::binary_search(candidate_sorted->begin(), candidate_sorted->end(), d);
            }
            if (ok) {
                filtered_docs.push_back(d);
                filtered_tfs.push_back(tfs[i]);
            }
        }
        if (filtered_docs.empty()) return {{}, {}};
        docs = luisa::span<const int>(filtered_docs.data(), filtered_docs.size());
        tfs = luisa::span<const uint16_t>(filtered_tfs.data(), filtered_tfs.size());
    }

    int df = static_cast<int>(docs.size());
    if (df == 0) return {{}, {}};
    double idf_val = idf(df, _index.N());
    size_t n = docs.size();

    if (_tfs_buf.size() < n) _tfs_buf.resize(n * 2 + 256);
    if (_denom_buf.size() < n) _denom_buf.resize(n * 2 + 256);

    luisa::vector<double> scores;
    scores.resize(n);
    luisa::vector<int> result_docs;
    result_docs.resize(n);
    luisa::fiber::parallel(static_cast<uint32_t>(n), [&](uint32_t i) noexcept {
        double tf = static_cast<double>(tfs[i]);
        double denom = tf + _denom_base[docs[i]];
        scores[i] = tf * idf_val * (_k1 + 1.0) * q_weight / denom;
        result_docs[i] = docs[i];
    });
    return {std::move(result_docs), std::move(scores)};
}

luisa::vector<double> BM25Scorer::accumulate(const luisa::vector<luisa::string> &query_tokens,
                                              const luisa::unordered_set<int> *candidate_docs) const {
    int N = _index.N();
    luisa::vector<double> scores_arr(N, 0.0);
    if (N == 0 || _denom_base.empty()) return scores_arr;
    if (candidate_docs && candidate_docs->empty()) return scores_arr;

    luisa::vector<int> cand_sorted;
    if (candidate_docs) {
        cand_sorted.reserve(candidate_docs->size());
        for (int d : *candidate_docs) cand_sorted.push_back(d);
        std::sort(cand_sorted.begin(), cand_sorted.end());
    }
    const luisa::vector<int> *cand_ptr = cand_sorted.empty() ? nullptr : &cand_sorted;

    luisa::unordered_map<luisa::string, int> token_counts;
    for (const auto &t : query_tokens) {
        auto it = token_counts.find(t);
        if (it == token_counts.end()) token_counts.emplace(t, 1);
        else ++(it->second);
    }

    for (const auto &[token, q_weight] : token_counts) {
        auto [docs, scores] = token_scores(token, static_cast<double>(q_weight), cand_ptr);
        for (size_t i = 0; i < docs.size(); ++i) {
            scores_arr[docs[i]] += scores[i];
        }
    }
    return scores_arr;
}

luisa::unordered_map<int, double> BM25Scorer::accumulate_sparse(const luisa::vector<luisa::string> &query_tokens,
                                                                 const luisa::unordered_set<int> *candidate_docs) const {
    luisa::unordered_map<int, double> scores;
    int N = _index.N();
    if (N == 0 || _denom_base.empty()) return scores;
    if (candidate_docs && candidate_docs->empty()) return scores;

    luisa::vector<int> cand_sorted;
    if (candidate_docs) {
        cand_sorted.reserve(candidate_docs->size());
        for (int d : *candidate_docs) cand_sorted.push_back(d);
        std::sort(cand_sorted.begin(), cand_sorted.end());
    }
    const luisa::vector<int> *cand_ptr = cand_sorted.empty() ? nullptr : &cand_sorted;

    luisa::unordered_map<luisa::string, int> token_counts;
    for (const auto &t : query_tokens) {
        auto it = token_counts.find(t);
        if (it == token_counts.end()) token_counts.emplace(t, 1);
        else ++(it->second);
    }

    for (const auto &[token, q_weight] : token_counts) {
        auto [docs, token_scores_vec] = token_scores(token, static_cast<double>(q_weight), cand_ptr);
        for (size_t i = 0; i < docs.size(); ++i) {
            int d = docs[i];
            auto it = scores.find(d);
            if (it == scores.end()) scores.emplace(d, token_scores_vec[i]);
            else it->second += token_scores_vec[i];
        }
    }
    return scores;
}

luisa::unordered_map<int, double> BM25Scorer::score(const luisa::vector<luisa::string> &query_tokens,
                                                     const luisa::unordered_set<int> *candidate_docs) const {
    int N = _index.N();
    if (N > 50000) {
        return accumulate_sparse(query_tokens, candidate_docs);
    }
    auto scores_arr = accumulate(query_tokens, candidate_docs);
    luisa::unordered_map<int, double> result;
    for (size_t i = 0; i < scores_arr.size(); ++i) {
        if (scores_arr[i] != 0.0) result.emplace(static_cast<int>(i), scores_arr[i]);
    }
    return result;
}

luisa::vector<std::pair<int, double>> BM25Scorer::score_topk(const luisa::vector<luisa::string> &query_tokens,
                                                              int top_k,
                                                              const luisa::unordered_set<int> *candidate_docs) const {
    if (_index.N() == 0 || _denom_base.empty() || top_k <= 0) return {};

    int N = _index.N();
    int cand_size = candidate_docs ? static_cast<int>(candidate_docs->size()) : N;
    bool use_sparse = (N > 50000 && top_k < N / 20) ||
                      (cand_size < 5000 && N > 100000) ||
                      (N > 5000 && top_k < N / 50);

    if (use_sparse) {
        auto scores = accumulate_sparse(query_tokens, candidate_docs);
        if (scores.empty()) return {};
        luisa::vector<std::pair<int, double>> items;
        items.reserve(scores.size());
        for (auto &[d, s] : scores) items.emplace_back(d, s);
        if (top_k >= static_cast<int>(items.size())) {
            std::sort(items.begin(), items.end(), [](const auto &a, const auto &b) {
                return a.second > b.second || (a.second == b.second && a.first < b.first);
            });
            return items;
        }
        std::partial_sort(items.begin(), items.begin() + top_k, items.end(), [](const auto &a, const auto &b) {
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        });
        items.resize(top_k);
        return items;
    }

    auto scores_arr = accumulate(query_tokens, candidate_docs);
    if (top_k >= N) {
        luisa::vector<std::pair<int, double>> result;
        for (size_t i = 0; i < scores_arr.size(); ++i) {
            if (scores_arr[i] > 0.0) result.emplace_back(static_cast<int>(i), scores_arr[i]);
        }
        std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        });
        return result;
    }

    using Item = std::pair<int, double>;
    std::priority_queue<Item, luisa::vector<Item>, std::function<bool(const Item &, const Item &)>> min_heap(
        [](const Item &a, const Item &b) { return a.second > b.second; });

    for (size_t i = 0; i < scores_arr.size(); ++i) {
        if (scores_arr[i] > 0.0) {
            if (static_cast<int>(min_heap.size()) < top_k) {
                min_heap.emplace(static_cast<int>(i), scores_arr[i]);
            } else if (scores_arr[i] > min_heap.top().second) {
                min_heap.pop();
                min_heap.emplace(static_cast<int>(i), scores_arr[i]);
            }
        }
    }

    luisa::vector<Item> result;
    result.reserve(min_heap.size());
    while (!min_heap.empty()) {
        result.push_back(min_heap.top());
        min_heap.pop();
    }
    std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
        return a.second > b.second || (a.second == b.second && a.first < b.first);
    });
    return result;
}

}// namespace tokenize
