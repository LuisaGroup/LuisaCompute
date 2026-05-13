#include "lambda_mart.h"
#include <algorithm>
#include <cmath>
#include <luisa/core/fiber.h>

namespace tokenize {

double dcg(const luisa::vector<double> &scores) {
    size_t n = scores.size();
    if (n == 0) return 0.0;
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double gain = std::pow(2.0, scores[i]) - 1.0;
        s += gain / std::log2(static_cast<double>(i + 2));
    }
    return s;
}

double ideal_dcg(const luisa::vector<double> &scores) {
    auto sorted = scores;
    luisa::sort(sorted.begin(), sorted.end(), std::greater<double>());
    return dcg(sorted);
}

double ndcg(const luisa::vector<double> &scores) {
    double ideal = ideal_dcg(scores);
    return ideal == 0.0 ? 0.0 : dcg(scores) / ideal;
}

LambdaMART::LambdaMART(int n_iterations, double learning_rate)
    : _n_iterations(n_iterations), _learning_rate(learning_rate) {}

void LambdaMART::fit(const luisa::vector<luisa::vector<luisa::vector<double>>> &X,
                     const luisa::vector<luisa::vector<double>> &y) {
    (void)y;
    if (X.empty() || X[0].empty()) return;
    size_t n_features = X[0][0].size();
    _weights.assign(n_features, 0.0);

    for (int iter = 0; iter < _n_iterations; ++iter) {
        int best_dim = -1;
        double best_delta = 0.0;
        double best_step = 0.0;
        for (size_t dim = 0; dim < n_features; ++dim) {
            for (double step : {-_learning_rate, _learning_rate}) {
                double old_w = _weights[dim];
                _weights[dim] = old_w + step;

                luisa::vector<double> query_ndcg(X.size(), 0.0);
                luisa::fiber::parallel(static_cast<uint32_t>(X.size()), [&](uint32_t q) noexcept {
                    luisa::vector<double> scores;
                    scores.reserve(X[q].size());
                    for (const auto &feat : X[q]) {
                        double score = 0.0;
                        for (size_t i = 0; i < n_features; ++i) score += feat[i] * _weights[i];
                        scores.push_back(score);
                    }
                    query_ndcg[q] = ndcg(scores);
                });

                double ndcg_sum = 0.0;
                for (auto v : query_ndcg) ndcg_sum += v;

                _weights[dim] = old_w;
                if (ndcg_sum > best_delta) {
                    best_delta = ndcg_sum;
                    best_dim = static_cast<int>(dim);
                    best_step = step;
                }
            }
        }
        if (best_dim < 0 || best_delta <= 0.0) break;
        _weights[best_dim] += best_step;
    }
}

luisa::vector<double> LambdaMART::predict(const luisa::vector<luisa::vector<double>> &X) const {
    if (_weights.empty()) return luisa::vector<double>(X.size(), 0.0);
    luisa::vector<double> result(X.size(), 0.0);

    luisa::fiber::parallel(static_cast<uint32_t>(X.size()), [&](uint32_t i) noexcept {
        double score = 0.0;
        for (size_t j = 0; j < _weights.size() && j < X[i].size(); ++j) {
            score += X[i][j] * _weights[j];
        }
        result[i] = score;
    });
    return result;
}

luisa::vector<std::pair<int, double>> LambdaMART::rank(const luisa::vector<std::pair<int, luisa::vector<double>>> &doc_features) const {
    luisa::vector<luisa::vector<double>> feats;
    feats.reserve(doc_features.size());
    luisa::vector<int> ids;
    ids.reserve(doc_features.size());
    for (const auto &[id, feat] : doc_features) {
        ids.push_back(id);
        feats.push_back(feat);
    }
    auto scores = predict(feats);
    luisa::vector<std::pair<int, double>> result;
    result.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) result.emplace_back(ids[i], scores[i]);
    luisa::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
        return a.second > b.second || (a.second == b.second && a.first < b.first);
    });
    return result;
}

}// namespace tokenize
