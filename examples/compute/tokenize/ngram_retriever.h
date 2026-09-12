#pragma once

#include "ngram_kernels.h"
#include "ngram_library.h"

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace tokenize {

// Host-side wrapper around the device state used for retrieval: the
// library buffers, per-batch query/result buffers and the compiled kernel.
//
// The host does ZERO matching logic — it only uploads queries, dispatches
// the selected DSL kernel and downloads the drafts. All decision making
// happens on device (see ngram_kernels.h).
class NgramRetriever {
public:
    // Borrows `library` (which must outlive the retriever), creates its
    // device buffers, uploads them and compiles the kernel for `variant`.
    NgramRetriever(luisa::compute::Device &device,
                   luisa::compute::Stream &stream,
                   NgramLibrary &library,
                   uint32_t min_n, uint32_t max_n, uint32_t k,
                   uint32_t max_query_len, size_t batch_capacity,
                   NgramKernelVariant variant = NgramKernelVariant::naive,
                   uint32_t block_size = 512u);

    [[nodiscard]] const NgramLibrary &library() const noexcept { return _lib; }
    [[nodiscard]] NgramKernelVariant variant() const noexcept { return _variant; }
    [[nodiscard]] uint32_t min_n() const noexcept { return _min_n; }
    [[nodiscard]] uint32_t max_n() const noexcept { return _max_n; }
    [[nodiscard]] uint32_t k() const noexcept { return _k; }
    [[nodiscard]] uint32_t query_stride() const noexcept { return _query_stride; }
    [[nodiscard]] size_t batch_capacity() const noexcept { return _capacity; }
    [[nodiscard]] uint32_t block_size() const noexcept { return _block_size; }

    // Run retrieval for a batch of queries packed row-major into
    // queries_flat with row length query_stride(); query_lens[i] is the
    // actual length of query i. drafts is resized to num_queries * k()
    // (invalid-ID padded) and draft_lens to num_queries.
    void retrieve(luisa::span<const uint32_t> queries_flat,
                  luisa::span<const uint32_t> query_lens,
                  luisa::vector<uint32_t> &drafts,
                  luisa::vector<uint32_t> &draft_lens);

    // Convenience overload taking per-query token vectors.
    void retrieve(luisa::span<const luisa::vector<uint32_t>> queries,
                  luisa::vector<uint32_t> &drafts,
                  luisa::vector<uint32_t> &draft_lens);

    // Split stages of retrieve() so benchmarks can time the kernel dispatch
    // alone: upload_queries() once, then any number of
    // dispatch_queries()/synchronize() rounds, then download_results().
    void upload_queries(luisa::span<const uint32_t> queries_flat,
                        luisa::span<const uint32_t> query_lens);
    void dispatch_queries(size_t num_queries);
    void download_results(size_t num_queries,
                          luisa::vector<uint32_t> &drafts,
                          luisa::vector<uint32_t> &draft_lens);
    void synchronize() { _stream << luisa::compute::synchronize(); }

private:
    luisa::compute::Device &_device;
    luisa::compute::Stream &_stream;
    NgramLibrary &_lib;
    uint32_t _min_n;
    uint32_t _max_n;
    uint32_t _k;
    uint32_t _query_stride;
    size_t _capacity;
    NgramKernelVariant _variant;
    uint32_t _block_size;
    RetrieveShader _shader;
    RetrieveShaderHash _shader_hash;
    NgramHashIndex _hash_index;
    luisa::compute::Buffer<uint32_t> _queries_buf;
    luisa::compute::Buffer<uint32_t> _qlens_buf;
    luisa::compute::Buffer<uint32_t> _drafts_buf;
    luisa::compute::Buffer<uint32_t> _draft_lens_buf;
};

}// namespace tokenize
