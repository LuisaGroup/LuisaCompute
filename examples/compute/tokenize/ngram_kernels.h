#pragma once

// DSL kernels for n-gram retrieval. ALL matching logic lives here; the
// host (NgramRetriever) only uploads queries and downloads drafts.
//
// Semantics (ported from vLLM's ngram proposer, see
// vllm/v1/spec_decode/ngram_proposer.py):
//   given corpus L[0..lib_len) and query q[0..qlen), for each n in
//   [min_n, max_n] find the EARLIEST position p with p + n < lib_len
//   (the match must leave at least one continuation token) such that
//   L[p..p+n) == q[qlen-n..qlen); pick the LONGEST n with any match
//   (ties on length -> earliest position); the draft is up to k tokens
//   L[p+n .. p+n+k) clamped to the library end. No match -> draft_len 0
//   and every draft slot set to ngram_invalid_id.

#include <luisa/dsl/func.h>
#include <luisa/dsl/resource.h>
#include <luisa/runtime/buffer.h>

namespace tokenize {

// Compile-time capacity of the shared-memory query-suffix staging in the
// parallel kernel. The host must reject max_n > this value for that variant.
inline constexpr uint32_t ngram_max_suffix_tokens = 64u;

// Kernel selection used by NgramRetriever and the benchmark driver.
enum class NgramKernelVariant : uint32_t {
    naive = 0,   // K1: one thread per query, intentionally simple/slow
    parallel = 1,// K2: one block per query, strided scan + shared reduction
    hash = 2,    // K3 (optional): host-built n-gram hash index lookup
};

// Shared kernel signature (all variants). Kernel prototype args use the
// canonical types: Buffer<uint32_t> for buffers, uint32_t for scalars; the
// kernel lambdas in ngram_kernels.cpp take the matching definition types
// (BufferUInt / Var<uint32_t>).
//   lib_buf[lib_len]          corpus token IDs
//   query_buf[num_queries * query_stride]   query token IDs (row-major)
//   qlen_buf[num_queries]     actual query lengths
//   draft_buf[num_queries * k]   draft token IDs (ngram_invalid_id padded)
//   draft_len_buf[num_queries]   number of valid draft tokens per query
//   req_off_buf[num_requests]   row base of each request: the global query
//     row processed by a thread is req_off_buf[kernel_id()] + local_row,
//     where local_row is dispatch_x() (one thread per query) or block_id().x
//     (one block per query). A single dispatch has kernel_id() == 0, so a
//     1-entry {0} buffer reproduces the plain flat-batch behavior; a
//     multi-dispatch (Shader::dispatch(span<const uint3>)) gives every
//     sub-dispatch its own index and thus its own row range, letting the
//     device run several requests' grids from ONE command.
using RetrieveKernel = luisa::compute::Kernel1D<void(
    luisa::compute::Buffer<uint32_t> /*lib_buf*/, uint32_t /*lib_len*/,
    luisa::compute::Buffer<uint32_t> /*query_buf*/, luisa::compute::Buffer<uint32_t> /*qlen_buf*/,
    uint32_t /*query_stride*/,
    luisa::compute::Buffer<uint32_t> /*draft_buf*/, luisa::compute::Buffer<uint32_t> /*draft_len_buf*/,
    uint32_t /*min_n*/, uint32_t /*max_n*/, uint32_t /*k*/,
    luisa::compute::Buffer<uint32_t> /*req_off_buf*/)>;

// Compiled form of RetrieveKernel (Shader is templated on the kernel args).
using RetrieveShader = luisa::compute::Shader<1,
    luisa::compute::Buffer<uint32_t>, uint32_t,
    luisa::compute::Buffer<uint32_t>, luisa::compute::Buffer<uint32_t>,
    uint32_t,
    luisa::compute::Buffer<uint32_t>, luisa::compute::Buffer<uint32_t>,
    uint32_t, uint32_t, uint32_t,
    luisa::compute::Buffer<uint32_t>>;

// K3 signature: RetrieveKernel + the hash index buffers (before req_off_buf).
using RetrieveKernelHash = luisa::compute::Kernel1D<void(
    luisa::compute::Buffer<uint32_t>, uint32_t,
    luisa::compute::Buffer<uint32_t>, luisa::compute::Buffer<uint32_t>,
    uint32_t,
    luisa::compute::Buffer<uint32_t>, luisa::compute::Buffer<uint32_t>,
    uint32_t, uint32_t, uint32_t,
    luisa::compute::Buffer<uint64_t> /*keys_buf*/,
    luisa::compute::Buffer<uint32_t> /*pos_buf*/,
    luisa::compute::Buffer<uint32_t> /*req_off_buf*/)>;

using RetrieveShaderHash = luisa::compute::Shader<1,
    luisa::compute::Buffer<uint32_t>, uint32_t,
    luisa::compute::Buffer<uint32_t>, luisa::compute::Buffer<uint32_t>,
    uint32_t,
    luisa::compute::Buffer<uint32_t>, luisa::compute::Buffer<uint32_t>,
    uint32_t, uint32_t, uint32_t,
    luisa::compute::Buffer<uint64_t>, luisa::compute::Buffer<uint32_t>,
    luisa::compute::Buffer<uint32_t>>;

// K1: naive baseline. One thread per query scans the whole corpus for
// every n-gram length. O(lib_len * max_n^2) per query — simple and slow,
// kept permanently as the device-side ground truth for cross-checks.
[[nodiscard]] RetrieveKernel make_retrieve_naive_kernel() noexcept;

// K2: one block of block_size threads per query. The query suffix is
// staged in shared memory; threads scan corpus positions strided by
// block_size and atomically reduce the earliest match position per
// n-gram length via shared-memory fetch_min.
[[nodiscard]] RetrieveKernel make_retrieve_parallel_kernel(uint32_t block_size) noexcept;

// K3: one thread per query; per n-gram length the query suffix is hashed
// (FNV-1a 64, identical to the host builder) and the earliest corpus
// position is looked up in the host-built open-addressing index. Key
// collisions are disambiguated by on-device content verification. O(max_n)
// work per query instead of an O(lib_len) scan. `cap_log2` (log2 of the
// table capacity) is baked into the compiled kernel.
[[nodiscard]] RetrieveKernelHash make_retrieve_hash_kernel(uint32_t cap_log2) noexcept;

}// namespace tokenize
