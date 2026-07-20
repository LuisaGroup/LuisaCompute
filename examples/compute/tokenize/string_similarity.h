#pragma once

#include <luisa/core/stl.h>

namespace tokenize {

[[nodiscard]] double jaro_similarity(luisa::string_view s, luisa::string_view t);
[[nodiscard]] double jaro_winkler_similarity(luisa::string_view s, luisa::string_view t, double p = 0.1, int max_prefix = 4);
[[nodiscard]] double sorensen_dice_coefficient(luisa::string_view s, luisa::string_view t);
[[nodiscard]] double ngram_overlap(luisa::string_view s, luisa::string_view t, int n = 2);
[[nodiscard]] double jaccard_similarity_tokens(const luisa::unordered_set<luisa::string> &a,
                                                const luisa::unordered_set<luisa::string> &b);
[[nodiscard]] int hamming_distance(luisa::string_view s, luisa::string_view t);
[[nodiscard]] double cosine_similarity_tfidf(const luisa::unordered_map<luisa::string, double> &vec_a,
                                              const luisa::unordered_map<luisa::string, double> &vec_b);

}// namespace tokenize
