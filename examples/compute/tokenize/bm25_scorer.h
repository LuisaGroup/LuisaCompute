#pragma once

#include "inverted_index.h"
#include <luisa/core/stl.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/shader.h>

namespace tokenize {

class BM25Scorer {
public:
    BM25Scorer(const InvertedIndex &index, double k1 = 1.2, double b = 0.75);
    BM25Scorer(const BM25Scorer &other);
    BM25Scorer &operator=(const BM25Scorer &other);
    BM25Scorer(BM25Scorer &&other) = delete;
    BM25Scorer &operator=(BM25Scorer &&other) = default;
    ~BM25Scorer();

    [[nodiscard]] static double idf(int df, int N) noexcept;
    [[nodiscard]] luisa::unordered_map<int, double> score(const luisa::vector<luisa::string> &query_tokens,
                                                           const luisa::unordered_set<int> *candidate_docs = nullptr) const;
    [[nodiscard]] luisa::vector<std::pair<int, double>> score_topk(const luisa::vector<luisa::string> &query_tokens,
                                                                    int top_k,
                                                                    const luisa::unordered_set<int> *candidate_docs = nullptr) const;

    // GPU accelerated dense accumulate (returns score for every doc)
    [[nodiscard]] luisa::vector<double> gpu_accumulate(luisa::compute::Device &device,
                                                        luisa::compute::Stream &stream,
                                                        const luisa::vector<luisa::string> &query_tokens) const;

    // GPU accelerated top-k: accumulate on GPU, extract nonzero candidates on GPU, top-k on CPU
    [[nodiscard]] luisa::vector<std::pair<int, double>> gpu_score_topk(luisa::compute::Device &device,
                                                                        luisa::compute::Stream &stream,
                                                                        const luisa::vector<luisa::string> &query_tokens,
                                                                        int top_k) const;

private:
    [[nodiscard]] std::pair<luisa::vector<int>, luisa::vector<double>>
    token_scores(const luisa::string &token, double q_weight, const luisa::vector<int> *candidate_sorted) const;

    [[nodiscard]] luisa::vector<double> accumulate(const luisa::vector<luisa::string> &query_tokens,
                                                    const luisa::unordered_set<int> *candidate_docs) const;
    [[nodiscard]] luisa::unordered_map<int, double> accumulate_sparse(const luisa::vector<luisa::string> &query_tokens,
                                                                       const luisa::unordered_set<int> *candidate_docs) const;

    void _init_gpu(luisa::compute::Device &device, luisa::compute::Stream &stream) const;

    const InvertedIndex &_index;
    double _k1;
    double _b;
    luisa::vector<double> _denom_base;
    mutable luisa::vector<double> _tfs_buf;
    mutable luisa::vector<double> _denom_buf;

    mutable bool _gpu_ready = false;
    mutable luisa::compute::Device _gpu_device;

    mutable luisa::compute::Buffer<int> _gpu_posting_docs;
    mutable luisa::compute::Buffer<int> _gpu_posting_docs_ptr;
    mutable luisa::compute::Buffer<uint16_t> _gpu_posting_tfs;
    mutable luisa::compute::Buffer<int> _gpu_posting_tfs_ptr;
    mutable luisa::compute::Buffer<float> _gpu_denom_base;

    mutable luisa::vector<int> _cpu_posting_docs_ptr;
    mutable luisa::vector<int> _cpu_posting_tfs_ptr;

    mutable luisa::compute::Shader1D<
        luisa::compute::Buffer<int>,
        luisa::compute::Buffer<uint16_t>,
        luisa::compute::Buffer<float>,
        luisa::compute::Buffer<float>,
        int, int, float, float, float> _gpu_accumulate_shader;

    mutable luisa::compute::Shader1D<
        luisa::compute::Buffer<float>,
        luisa::compute::Buffer<int>,
        luisa::compute::Buffer<float>,
        luisa::compute::Buffer<int>,
        int> _gpu_extract_nonzero_shader;
};

}// namespace tokenize
