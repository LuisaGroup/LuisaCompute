// Shared retrieval command service (retrieval plan phase 2). Every operation,
// including long indexing calls, is serialized to keep corpus transitions
// consistent across console and TCP clients.

#include "retrieval_service.h"

namespace luisa::embed {

RetrievalLoadResult RetrievalService::load(const std::filesystem::path &root) {
    std::lock_guard lock{_mutex};
    return _index.load(root);
}

RetrievalLoadResult RetrievalService::load(
    const std::filesystem::path &root,
    luisa::span<const luisa::string> extensions) {
    std::lock_guard lock{_mutex};
    return _index.load(root, extensions);
}

RetrievalUnloadResult RetrievalService::unload(const std::filesystem::path &root) {
    std::lock_guard lock{_mutex};
    return _index.unload(root);
}

RetrievalUpdateResult RetrievalService::update(const std::filesystem::path &path) {
    std::lock_guard lock{_mutex};
    return _index.update(path);
}

RetrievalQueryResult RetrievalService::query(luisa::string_view text, uint32_t top_k) {
    std::lock_guard lock{_mutex};
    return _index.query(text, top_k);
}

}// namespace luisa::embed
