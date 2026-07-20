#pragma once

#include "inverted_index.h"
#include <luisa/core/stl.h>

namespace tokenize {

class LevenshteinAutomaton {
public:
    LevenshteinAutomaton(luisa::string_view pattern, int max_edits, int prefix_length = 1);

    [[nodiscard]] static int auto_fuzziness(luisa::string_view term) noexcept;
    [[nodiscard]] static int damerau_levenshtein(luisa::string_view s, luisa::string_view t);

    [[nodiscard]] luisa::vector<luisa::string> match(const InvertedIndex &dictionary, int max_expansions = 50) const;

private:
    [[nodiscard]] int freq_lower_bound(luisa::string_view term) const;

    luisa::string _pattern;
    int _max_edits;
    int _prefix_length;
    luisa::unordered_map<char, int> _pattern_counts;
    luisa::vector<std::pair<char, int>> _pattern_counts_items;
    luisa::unordered_set<luisa::string> _pattern_deletes;
};

}// namespace tokenize
