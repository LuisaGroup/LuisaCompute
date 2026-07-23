#pragma once

#include "inverted_index.h"
#include "string_similarity.h"
#include <luisa/core/stl.h>

namespace tokenize {

[[nodiscard]] luisa::vector<std::pair<int, double>> mmr_rerank(
    const luisa::vector<std::pair<int, double>> &results,
    const InvertedIndex &index,
    double lambda_param = 0.5,
    int top_k = -1);

}// namespace tokenize
