#pragma once

#include "inverted_index.h"
#include "bm25_scorer.h"
#include <luisa/core/stl.h>

namespace tokenize {

class RM3Expander {
public:
    RM3Expander(const InvertedIndex &index,
                const BM25Scorer &scorer,
                int fb_docs = 3,
                int fb_terms = 10,
                double alpha = 0.5);

    [[nodiscard]] luisa::vector<luisa::string> expand(const luisa::vector<luisa::string> &query_tokens, int top_k = 10);

private:
    const InvertedIndex &_index;
    const BM25Scorer &_scorer;
    int _fb_docs;
    int _fb_terms;
    double _alpha;
};

}// namespace tokenize
