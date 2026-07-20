#include "mmr_rerank.h"
#include <cmath>
#include <luisa/core/fiber.h>

namespace tokenize {

luisa::vector<std::pair<int, double>> mmr_rerank(
    const luisa::vector<std::pair<int, double>> &results,
    const InvertedIndex &index,
    double lambda_param,
    int top_k) {
    if (results.empty()) return {};
    if (top_k < 0) top_k = static_cast<int>(results.size());

    // Parallel pre-fetch of token sets (independent per document)
    luisa::vector<luisa::unordered_set<luisa::string>> doc_token_sets(results.size());
    luisa::fiber::parallel(static_cast<uint32_t>(results.size()), [&](uint32_t i) noexcept {
        doc_token_sets[i] = index.doc_token_set(results[i].first);
    });

    luisa::unordered_map<int, size_t> doc_id_to_idx;
    for (size_t i = 0; i < results.size(); ++i) {
        doc_id_to_idx[results[i].first] = i;
    }

    luisa::vector<std::pair<int, double>> selected;
    luisa::vector<std::pair<int, double>> remaining = results;

    while (!remaining.empty() && static_cast<int>(selected.size()) < top_k) {
        // Parallel inner MMR score loop: each remaining doc is independent for a fixed selected set
        luisa::vector<double> scores(remaining.size());
        luisa::fiber::parallel(static_cast<uint32_t>(remaining.size()), [&](uint32_t idx) noexcept {
            const auto &[doc_id, relevance] = remaining[idx];
            double max_sim = 0.0;
            for (const auto &[sel_id, _] : selected) {
                auto doc_it = doc_id_to_idx.find(doc_id);
                auto sel_it = doc_id_to_idx.find(sel_id);
                if (doc_it == doc_id_to_idx.end() || sel_it == doc_id_to_idx.end()) continue;
                double sim = jaccard_similarity_tokens(doc_token_sets[doc_it->second], doc_token_sets[sel_it->second]);
                if (sim > max_sim) max_sim = sim;
            }
            scores[idx] = lambda_param * relevance - (1.0 - lambda_param) * max_sim;
        });

        // Sequential reduction to find best score
        int best_idx = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        for (size_t idx = 0; idx < remaining.size(); ++idx) {
            if (scores[idx] > best_score) {
                best_score = scores[idx];
                best_idx = static_cast<int>(idx);
            }
        }
        if (best_idx < 0) break;
        selected.push_back(remaining[best_idx]);
        remaining.erase(remaining.begin() + best_idx);
    }
    return selected;
}

}// namespace tokenize
