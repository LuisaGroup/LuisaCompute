#include "ngram_retriever.h"

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>

#include <algorithm>

namespace tokenize {

NgramRetriever::NgramRetriever(luisa::compute::Device &device,
                               luisa::compute::Stream &stream,
                               NgramLibrary &library,
                               uint32_t min_n, uint32_t max_n, uint32_t k,
                               uint32_t max_query_len, size_t batch_capacity,
                               NgramKernelVariant variant,
                               uint32_t block_size)
    : _device(device), _stream(stream), _lib(library),
      _min_n(min_n), _max_n(max_n), _k(k),
      _query_stride(std::max(max_n, max_query_len)),
      _capacity(batch_capacity), _variant(variant), _block_size(block_size) {
    LUISA_ASSERT(min_n >= 1u, "min_n must be >= 1");
    LUISA_ASSERT(min_n <= max_n, "min_n must be <= max_n");
    LUISA_ASSERT(k >= 1u, "k must be >= 1");
    LUISA_ASSERT(_lib.size() > 0, "library must not be empty");
    LUISA_ASSERT(_lib.size() <= 0xFFFFFFFFu, "library exceeds uint32 indexing");
    LUISA_ASSERT(batch_capacity > 0, "batch capacity must be >= 1");
    if (variant == NgramKernelVariant::parallel) {
        LUISA_ASSERT(max_n <= ngram_max_suffix_tokens,
                     "max_n {} exceeds the parallel kernel suffix capacity {}",
                     max_n, ngram_max_suffix_tokens);
        LUISA_ASSERT(block_size > 0u, "block_size must be >= 1");
    }

    // library buffers
    _lib.tokens_buf = _device.create_buffer<uint32_t>(_lib.tokens.size());
    _lib.offsets_buf = _device.create_buffer<uint32_t>(_lib.doc_offsets.size());
    _lib.lengths_buf = _device.create_buffer<uint32_t>(_lib.doc_lengths.size());

    // batch buffers
    _queries_buf = _device.create_buffer<uint32_t>(_capacity * _query_stride);
    _qlens_buf = _device.create_buffer<uint32_t>(_capacity);
    _drafts_buf = _device.create_buffer<uint32_t>(_capacity * _k);
    _draft_lens_buf = _device.create_buffer<uint32_t>(_capacity);

    // one-shot library upload; retrieval calls chain on the same stream
    _stream << _lib.tokens_buf.view().copy_from(luisa::span{_lib.tokens})
            << _lib.offsets_buf.view().copy_from(luisa::span{_lib.doc_offsets})
            << _lib.lengths_buf.view().copy_from(luisa::span{_lib.doc_lengths});

    switch (_variant) {
        case NgramKernelVariant::naive:
            _shader = _device.compile(make_retrieve_naive_kernel());
            break;
        case NgramKernelVariant::parallel:
            _shader = _device.compile(make_retrieve_parallel_kernel(_block_size));
            break;
        case NgramKernelVariant::hash: {
            luisa::Clock clock;
            clock.tic();
            _hash_index = build_ngram_hash_index(luisa::span{_lib.tokens}, _min_n, _max_n);
            const double build_ms = clock.toc();
            _hash_index.keys_buf = _device.create_buffer<uint64_t>(_hash_index.keys.size());
            _hash_index.pos_buf = _device.create_buffer<uint32_t>(_hash_index.pos.size());
            _stream << _hash_index.keys_buf.view().copy_from(luisa::span{_hash_index.keys})
                    << _hash_index.pos_buf.view().copy_from(luisa::span{_hash_index.pos});
            LUISA_INFO("hash index: {} slots (2^{}), built in {:.2f} ms",
                       _hash_index.keys.size(), _hash_index.cap_log2, build_ms);
            _shader_hash = _device.compile(make_retrieve_hash_kernel(_hash_index.cap_log2));
            break;
        }
    }
}

void NgramRetriever::upload_queries(luisa::span<const uint32_t> queries_flat,
                                    luisa::span<const uint32_t> query_lens) {
    const auto num_queries = query_lens.size();
    LUISA_ASSERT(num_queries > 0, "upload_queries() requires at least one query");
    LUISA_ASSERT(num_queries <= _capacity,
                 "query batch {} exceeds capacity {}", num_queries, _capacity);
    LUISA_ASSERT(queries_flat.size() == num_queries * _query_stride,
                 "queries_flat size {} != num_queries * query_stride {}",
                 queries_flat.size(), num_queries * _query_stride);
    for (auto len : query_lens) {
        LUISA_ASSERT(len <= _query_stride,
                     "query length {} exceeds query_stride {}", len, _query_stride);
    }
    _stream << _queries_buf.view(0, num_queries * _query_stride)
                   .copy_from(queries_flat)
            << _qlens_buf.view(0, num_queries).copy_from(query_lens);
}

void NgramRetriever::dispatch_queries(size_t num_queries) {
    LUISA_ASSERT(num_queries > 0, "dispatch_queries() requires at least one query");
    LUISA_ASSERT(num_queries <= _capacity,
                 "query batch {} exceeds capacity {}", num_queries, _capacity);
    switch (_variant) {
        case NgramKernelVariant::naive:
            _stream << _shader(_lib.tokens_buf, static_cast<uint32_t>(_lib.size()),
                               _queries_buf, _qlens_buf, _query_stride,
                               _drafts_buf, _draft_lens_buf,
                               _min_n, _max_n, _k)
                           .dispatch(static_cast<uint32_t>(num_queries));
            break;
        case NgramKernelVariant::parallel:
            _stream << _shader(_lib.tokens_buf, static_cast<uint32_t>(_lib.size()),
                               _queries_buf, _qlens_buf, _query_stride,
                               _drafts_buf, _draft_lens_buf,
                               _min_n, _max_n, _k)
                           .dispatch(static_cast<uint32_t>(num_queries) * _block_size);
            break;
        case NgramKernelVariant::hash:
            _stream << _shader_hash(_lib.tokens_buf, static_cast<uint32_t>(_lib.size()),
                                    _queries_buf, _qlens_buf, _query_stride,
                                    _drafts_buf, _draft_lens_buf,
                                    _min_n, _max_n, _k,
                                    _hash_index.keys_buf, _hash_index.pos_buf)
                           .dispatch(static_cast<uint32_t>(num_queries));
            break;
    }
}

void NgramRetriever::download_results(size_t num_queries,
                                      luisa::vector<uint32_t> &drafts,
                                      luisa::vector<uint32_t> &draft_lens) {
    LUISA_ASSERT(num_queries > 0 && num_queries <= _capacity,
                 "invalid query batch {}", num_queries);
    drafts.resize(num_queries * _k);
    draft_lens.resize(num_queries);
    _stream << _drafts_buf.view(0, num_queries * _k).copy_to(luisa::span{drafts})
            << _draft_lens_buf.view(0, num_queries).copy_to(luisa::span{draft_lens})
            << luisa::compute::synchronize();
}

void NgramRetriever::retrieve(luisa::span<const uint32_t> queries_flat,
                              luisa::span<const uint32_t> query_lens,
                              luisa::vector<uint32_t> &drafts,
                              luisa::vector<uint32_t> &draft_lens) {
    upload_queries(queries_flat, query_lens);
    dispatch_queries(query_lens.size());
    download_results(query_lens.size(), drafts, draft_lens);
}

void NgramRetriever::retrieve(luisa::span<const luisa::vector<uint32_t>> queries,
                              luisa::vector<uint32_t> &drafts,
                              luisa::vector<uint32_t> &draft_lens) {
    luisa::vector<uint32_t> flat(queries.size() * _query_stride);
    luisa::vector<uint32_t> lens(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
        lens[i] = static_cast<uint32_t>(queries[i].size());
        std::copy_n(queries[i].begin(), std::min(queries[i].size(), size_t(_query_stride)),
                    flat.begin() + i * _query_stride);
    }
    retrieve(luisa::span{flat}, luisa::span{lens}, drafts, draft_lens);
}

}// namespace tokenize
