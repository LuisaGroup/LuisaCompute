// Serialized command-service boundary shared by the retrieval REPL and JSON-RPC
// server (retrieval plan phase 2; hybrid encoder wiring phase H2). Frontends
// exchange only structured DTOs, so the console grammar and the RPC schema are
// identical for a dense-only and for a hybrid service.
#pragma once

#include <mutex>
#include <utility>

#include "retrieval_index.h"
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>

namespace luisa::embed {

class RetrievalService {
public:
    // Dense-only service (unchanged behavior): the encoder returns one dense
    // vector per text and scores are plain dense dot products.
    explicit RetrievalService(RetrievalEncodeCallback encode) noexcept
        : _index{std::move(encode)} {}
    // Hybrid service (phase H2): the encoder returns the dense vector plus the
    // BGE-M3 sparse lexical weights, and scores fuse both halves. Which overload
    // a call selects is decided by the callback's return type
    // (RetrievalEmbedding vs luisa::vector<float>), never by the data: a dense
    // callback cannot silently become a hybrid one.
    explicit RetrievalService(RetrievalHybridEncodeCallback encode) noexcept
        : _index{std::move(encode)} {}
    ~RetrievalService() noexcept = default;

    // Optional disk cache for directory import; nullptr disables it. Not owned,
    // must outlive the service. Call before serving requests.
    void set_embed_cache(RetrievalEmbedCache *cache) noexcept { _index.set_embed_cache(cache); }

    RetrievalService(const RetrievalService &) = delete;
    RetrievalService(RetrievalService &&) = delete;
    RetrievalService &operator=(const RetrievalService &) = delete;
    RetrievalService &operator=(RetrievalService &&) = delete;

    [[nodiscard]] RetrievalLoadResult load(const luisa::filesystem::path &root);
    [[nodiscard]] RetrievalLoadResult load(
        const luisa::filesystem::path &root,
        luisa::span<const luisa::string> extensions);
    [[nodiscard]] RetrievalUnloadResult unload(const luisa::filesystem::path &root);
    [[nodiscard]] RetrievalUpdateResult update(const luisa::filesystem::path &path);
    [[nodiscard]] RetrievalQueryResult query(
        luisa::string_view text,
        uint32_t top_k = kRetrievalDefaultTopK);

private:
    RetrievalIndex _index;
    std::mutex _mutex;
};

}// namespace luisa::embed
