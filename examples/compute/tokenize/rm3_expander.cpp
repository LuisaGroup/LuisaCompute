#include "rm3_expander.h"
#include <algorithm>
#include <luisa/core/fiber.h>

namespace tokenize {

RM3Expander::RM3Expander(const InvertedIndex &index,
                         const BM25Scorer &scorer,
                         int fb_docs,
                         int fb_terms,
                         double alpha)
    : _index(index), _scorer(scorer), _fb_docs(fb_docs), _fb_terms(fb_terms), _alpha(alpha) {}

luisa::vector<luisa::string> RM3Expander::expand(const luisa::vector<luisa::string> &query_tokens, int top_k) {
    (void)top_k;
    if (query_tokens.empty()) return {};

    auto results = _scorer.score_topk(query_tokens, _fb_docs);
    if (results.empty()) return query_tokens;

    luisa::unordered_set<int> doc_ids;
    for (const auto &[doc_id, _] : results) doc_ids.insert(doc_id);

    // Parallel feedback document aggregation (thread-local maps + merge)
    luisa::unordered_map<luisa::string, double> term_scores;
    int total_tokens = 0;
    {
        struct LocalAgg {
            luisa::unordered_map<luisa::string, double> scores;
            int total = 0;
        };
        luisa::vector<LocalAgg> locals;
        luisa::fiber::mutex mtx;

        luisa::vector<int> doc_ids_vec;
        doc_ids_vec.reserve(doc_ids.size());
        for (int d : doc_ids) doc_ids_vec.push_back(d);

        luisa::fiber::parallel(static_cast<uint32_t>(doc_ids_vec.size()), [&](uint32_t begin, uint32_t end) noexcept {
            LocalAgg local;
            for (uint32_t i = begin; i < end; ++i) {
                int doc_id = doc_ids_vec[i];
                if (doc_id >= 0 && doc_id < static_cast<int>(_index.doc_term_freqs().size())) {
                    for (const auto &[term, tf] : _index.doc_term_freqs()[doc_id]) {
                        auto it = local.scores.find(term);
                        if (it == local.scores.end()) local.scores.emplace(term, static_cast<double>(tf));
                        else it->second += static_cast<double>(tf);
                        local.total += tf;
                    }
                }
            }
            {
                luisa::fiber::lock lck(mtx);
                locals.push_back(std::move(local));
            }
        });

        for (auto &local : locals) {
            for (auto &[term, score] : local.scores) {
                auto it = term_scores.find(term);
                if (it == term_scores.end()) term_scores.emplace(term, score);
                else it->second += score;
            }
            total_tokens += local.total;
        }
    }

    if (total_tokens == 0) return query_tokens;
    for (auto &[term, score] : term_scores) score /= static_cast<double>(total_tokens);

    luisa::unordered_map<luisa::string, double> expanded;
    for (const auto &token : query_tokens) {
        auto it = expanded.find(token);
        if (it == expanded.end()) expanded.emplace(token, _alpha / static_cast<double>(query_tokens.size()));
        else it->second += _alpha / static_cast<double>(query_tokens.size());
    }

    luisa::vector<std::pair<luisa::string, double>> term_vec;
    term_vec.reserve(term_scores.size());
    for (auto &[term, score] : term_scores) term_vec.emplace_back(term, score);
    std::partial_sort(term_vec.begin(), term_vec.begin() + std::min(static_cast<size_t>(_fb_terms), term_vec.size()), term_vec.end(),
                      [](const auto &a, const auto &b) { return a.second > b.second; });

    for (size_t i = 0; i < term_vec.size() && i < static_cast<size_t>(_fb_terms); ++i) {
        const auto &[term, score] = term_vec[i];
        auto it = expanded.find(term);
        if (it == expanded.end()) expanded.emplace(term, (1.0 - _alpha) * score);
        else it->second += (1.0 - _alpha) * score;
    }

    luisa::vector<luisa::string> result;
    for (const auto &[term, weight] : expanded) {
        int count = std::max(1, static_cast<int>(weight * 10.0));
        for (int i = 0; i < count; ++i) result.push_back(term);
    }
    return result;
}

}// namespace tokenize
