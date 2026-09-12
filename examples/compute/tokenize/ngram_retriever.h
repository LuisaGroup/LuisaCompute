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

    // ---- multi-request support ----
    // The retriever's batch capacity can be partitioned into several
    // requests: disjoint row ranges of the shared query/draft buffers.
    // Because stream commit is not thread-safe, every method that touches
    // a stream takes it explicitly; callers running requests on different
    // fibers must give each request its OWN stream (and never share one
    // stream between concurrent fibers).
    //
    // The compiled kernels resolve a thread's global query row as
    // req_off_buf[kernel_id()] + local_row, so:
    //  - a single dispatch needs a 1-entry offsets buffer holding the
    //    request's row base (upload_request_offsets);
    //  - dispatch_requests_multi() needs one N-entry buffer with every
    //    request's row base, and launches ALL requests' grids in ONE
    //    command (sub-dispatch r reports kernel_id() == r), which the
    //    device may execute concurrently.

    // Create a 1-entry offsets buffer holding `row_base`, uploaded on `stream`.
    [[nodiscard]] luisa::compute::Buffer<uint32_t> upload_request_offsets(
        luisa::compute::Stream &stream, uint32_t row_base);
    // Create an N-entry offsets buffer (one row base per request), uploaded on `stream`.
    [[nodiscard]] luisa::compute::Buffer<uint32_t> upload_request_offsets(
        luisa::compute::Stream &stream, luisa::span<const uint32_t> row_bases);

    // Upload one request's query rows [row_base, row_base + lens.size()).
    void upload_request(luisa::compute::Stream &stream,
                        luisa::span<const uint32_t> queries_flat,
                        luisa::span<const uint32_t> query_lens,
                        uint32_t row_base);

    // Single dispatch over one request's rows on `stream`.
    // `off_buf` must hold the request's row base at index 0.
    void dispatch_request(luisa::compute::Stream &stream,
                          const luisa::compute::Buffer<uint32_t> &off_buf,
                          uint32_t row_count);

    // ONE multi-dispatch command over all requests: dispatch_sizes[r] is the
    // logical thread-grid of request r (row_counts[r], or row_counts[r] *
    // block_size for the parallel variant). `off_buf` must hold every
    // request's row base. The caller synchronizes the stream.
    void dispatch_requests_multi(const luisa::compute::Buffer<uint32_t> &off_buf,
                                 luisa::span<const luisa::uint3> dispatch_sizes);

    // Download one request's draft rows [row_base, row_base + row_count).
    void download_request(luisa::compute::Stream &stream, uint32_t row_base,
                          uint32_t row_count,
                          luisa::vector<uint32_t> &drafts,
                          luisa::vector<uint32_t> &draft_lens);

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
    luisa::compute::Buffer<uint32_t> _req_off_buf;// 1-entry {0}: flat-batch rows
};

}// namespace tokenize
