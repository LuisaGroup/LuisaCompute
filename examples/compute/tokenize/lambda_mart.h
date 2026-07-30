#pragma once

#include <luisa/core/stl.h>

namespace tokenize {

[[nodiscard]] double dcg(const luisa::vector<double> &scores);
[[nodiscard]] double ideal_dcg(const luisa::vector<double> &scores);
[[nodiscard]] double ndcg(const luisa::vector<double> &scores);

class LambdaMART {
public:
    explicit LambdaMART(int n_iterations = 50, double learning_rate = 0.05);

    void fit(const luisa::vector<luisa::vector<luisa::vector<double>>> &X,
             const luisa::vector<luisa::vector<double>> &y);

    [[nodiscard]] luisa::vector<double> predict(const luisa::vector<luisa::vector<double>> &X) const;
    [[nodiscard]] luisa::vector<std::pair<int, double>> rank(const luisa::vector<std::pair<int, luisa::vector<double>>> &doc_features) const;

private:
    int _n_iterations;
    double _learning_rate;
    luisa::vector<double> _weights;
};

}// namespace tokenize
