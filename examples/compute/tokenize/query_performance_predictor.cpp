#include "query_performance_predictor.h"

namespace tokenize {

QueryPerformancePredictor::QueryPerformancePredictor(const InvertedIndex &index, const BM25Scorer &scorer)
    : _index(index), _scorer(scorer) {}

luisa::unordered_map<luisa::string, int> QueryPerformancePredictor::batch_doc_freq(const luisa::unordered_set<luisa::string> &tokens) const {
    luisa::unordered_map<luisa::string, int> result;
    for (const auto &token : tokens) {
        auto p = _index.get_postings(token);
        if (p) result.emplace(token, static_cast<int>(p->docs.size()));
    }
    return result;
}

double QueryPerformancePredictor::avg_idf(const luisa::vector<luisa::string> &query_tokens) const {
    luisa::unordered_set<luisa::string> unique_tokens;
    for (const auto &t : query_tokens) unique_tokens.insert(t);
    auto freqs = batch_doc_freq(unique_tokens);
    if (freqs.empty()) return 0.0;
    double sum = 0.0;
    int count = 0;
    int N = _index.N();
    for (const auto &[token, df] : freqs) {
        if (df > 0) {
            sum += _scorer.idf(df, N);
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

double QueryPerformancePredictor::max_idf(const luisa::vector<luisa::string> &query_tokens) const {
    luisa::unordered_set<luisa::string> unique_tokens;
    for (const auto &t : query_tokens) unique_tokens.insert(t);
    auto freqs = batch_doc_freq(unique_tokens);
    if (freqs.empty()) return 0.0;
    double max_val = 0.0;
    int N = _index.N();
    for (const auto &[token, df] : freqs) {
        if (df > 0) max_val = std::max(max_val, _scorer.idf(df, N));
    }
    return max_val;
}

double QueryPerformancePredictor::query_scope(const luisa::vector<luisa::string> &query_tokens) const {
    if (_index.N() == 0) return 0.0;
    luisa::unordered_set<luisa::string> unique_tokens;
    for (const auto &t : query_tokens) unique_tokens.insert(t);
    luisa::unordered_set<int> all_docs;
    for (const auto &t : unique_tokens) {
        if (_index.has_term(t)) {
            auto p = _index.get_postings(t);
            if (p) {
                for (int d : p->docs) all_docs.insert(d);
            }
        }
    }
    if (all_docs.empty()) return 0.0;
    return static_cast<double>(all_docs.size()) / static_cast<double>(_index.N());
}

bool QueryPerformancePredictor::is_hard_query(const luisa::vector<luisa::string> &query_tokens, double avg_idf_threshold) const {
    return avg_idf(query_tokens) < avg_idf_threshold;
}

}// namespace tokenize
