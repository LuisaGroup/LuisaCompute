#pragma once

#include "ngram_tokenizer.h"
#include "bm25_scorer.h"
#include "levenshtein_automaton.h"
#include <luisa/core/stl.h>
#include <luisa/core/fiber.h>

namespace tokenize {

class Searcher {
public:
    Searcher(const InvertedIndex &index,
             const NgramTokenizer *tokenizer = nullptr,
             const BM25Scorer *scorer = nullptr,
             double k1 = 1.2,
             double b = 0.75,
             double min_should_match = 0.5,
             luisa::string fuzziness = "AUTO",
             int max_expansions = 50,
             int prefix_length = 1);

    [[nodiscard]] luisa::vector<std::pair<int, double>> search(luisa::string_view query, int top_k = 10);

private:
    [[nodiscard]] static bool is_latin_token(luisa::string_view token);
    [[nodiscard]] luisa::vector<luisa::string> expand_token(luisa::string_view token);

    const InvertedIndex &_index;
    NgramTokenizer _tokenizer;
    BM25Scorer _scorer;
    double _k1;
    double _b;
    double _min_should_match;
    luisa::string _fuzziness;
    int _max_expansions;
    int _prefix_length;
    luisa::unordered_map<luisa::string, luisa::vector<luisa::string>> _expand_cache;
    mutable luisa::fiber::mutex _expand_cache_mtx;
};

}// namespace tokenize
