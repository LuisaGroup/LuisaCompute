// Shared retrieval command service (retrieval plan phase 2). Every operation,
// including long indexing calls, is serialized to keep corpus transitions
// consistent across console and TCP clients.

#include "retrieval_service.h"
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>

namespace luisa::embed {

RetrievalLoadResult RetrievalService::load(const luisa::filesystem::path &root) {
    std::lock_guard lock{_mutex};
    return _index.load(root);
}

RetrievalLoadResult RetrievalService::load(
    const luisa::filesystem::path &root,
    luisa::span<const luisa::string> extensions) {
    std::lock_guard lock{_mutex};
    return _index.load(root, extensions);
}

RetrievalLoadResult RetrievalService::load(
    const luisa::filesystem::path &root,
    luisa::span<const luisa::string> extensions,
    const ChunkOptions &options,
    const RetrievalProgress *progress) {
    std::lock_guard lock{_mutex};
    return _index.load(root, extensions, options, progress);
}

RetrievalUnloadResult RetrievalService::unload(const luisa::filesystem::path &root) {
    std::lock_guard lock{_mutex};
    return _index.unload(root);
}

RetrievalUpdateResult RetrievalService::update(const luisa::filesystem::path &path) {
    std::lock_guard lock{_mutex};
    return _index.update(path);
}

RetrievalQueryResult RetrievalService::query(luisa::string_view text, uint32_t top_k) {
    std::lock_guard lock{_mutex};
    return _index.query(text, top_k);
}

RetrievalQueryResult RetrievalService::query(
    luisa::string_view text,
    uint32_t top_k,
    const RetrievalQueryOptions *options) {
    std::lock_guard lock{_mutex};
    return _index.query(text, top_k, options);
}

RetrievalDocumentResult RetrievalService::add_document(
    luisa::string doc_id,
    luisa::string_view text,
    luisa::string_view metadata_json,
    const ChunkOptions &options,
    const RetrievalProgress *progress) {
    std::lock_guard lock{_mutex};
    return _index.add_document(std::move(doc_id), text, metadata_json, options, progress);
}

RetrievalDocumentResult RetrievalService::remove_document(luisa::string_view doc_id) {
    std::lock_guard lock{_mutex};
    return _index.remove_document(doc_id);
}

RetrievalDocumentResult RetrievalService::update_document(
    luisa::string doc_id,
    luisa::string_view text,
    luisa::string_view metadata_json,
    const ChunkOptions &options,
    const RetrievalProgress *progress) {
    std::lock_guard lock{_mutex};
    return _index.update_document(std::move(doc_id), text, metadata_json, options, progress);
}

RetrievalStatus RetrievalService::save(const luisa::filesystem::path &path) {
    std::lock_guard lock{_mutex};
    return _index.save(path);
}

RetrievalStatus RetrievalService::load_snapshot(const luisa::filesystem::path &path) {
    std::lock_guard lock{_mutex};
    return _index.load_snapshot(path);
}

}// namespace luisa::embed
