#pragma once

#include "inverted_index.h"
#include "bm25_scorer.h"
#include <luisa/core/stl.h>

namespace tokenize {

class QueryPerformancePredictor {
public:
    QueryPerformancePredictor(const InvertedIndex &index, const BM25Scorer &scorer);

    [[nodiscard]] double avg_idf(const luisa::vector<luisa::string> &query_tokens) const;
    [[nodiscard]] double max_idf(const luisa::vector<luisa::string> &query_tokens) const;
    [[nodiscard]] double query_scope(const luisa::vector<luisa::string> &query_tokens) const;
    [[nodiscard]] bool is_hard_query(const luisa::vector<luisa::string> &query_tokens, double avg_idf_threshold = 2.0) const;

private:
    [[nodiscard]] luisa::unordered_map<luisa::string, int> batch_doc_freq(const luisa::unordered_set<luisa::string> &tokens) const;

    const InvertedIndex &_index;
    const BM25Scorer &_scorer;
};

}// namespace tokenize
