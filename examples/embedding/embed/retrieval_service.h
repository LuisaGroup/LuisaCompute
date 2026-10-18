// Serialized command-service boundary shared by the retrieval REPL and JSON-RPC
// server (retrieval plan phase 2). Frontends exchange only structured DTOs.
#pragma once

#include <mutex>

#include "retrieval_index.h"
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>

namespace luisa::embed {

class RetrievalService {
public:
    explicit RetrievalService(RetrievalEncodeCallback encode) noexcept
        : _index{std::move(encode)} {}
    ~RetrievalService() noexcept = default;

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
