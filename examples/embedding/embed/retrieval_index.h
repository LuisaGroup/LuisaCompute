// Host-side whole-file retrieval index for the interactive embedding retrieval
// example (retrieval plan phases 1 and 4.1-4.3). No device or DSL types cross
// this interface; the caller supplies the dense-embedding callback.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

#include <luisa/core/stl/functional.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

namespace luisa::embed {

inline constexpr uint32_t kRetrievalDefaultTopK = 5u;
inline constexpr uint32_t kRetrievalMaxTopK = 100u;

using RetrievalEncodeCallback = luisa::function<luisa::vector<float>(luisa::string_view utf8)>;

enum class RetrievalErrorCode {
    none,
    invalid_argument,
    not_found,
    operational,
};

struct RetrievalStatus {
    RetrievalErrorCode code{RetrievalErrorCode::none};
    luisa::string message;

    [[nodiscard]] bool ok() const noexcept { return code == RetrievalErrorCode::none; }
};

struct RetrievalHit {
    luisa::string path;
    double score{};
};

struct RetrievalLoadResult {
    RetrievalStatus status;
    luisa::string root;
    luisa::vector<luisa::string> extensions;
    size_t added{};
    size_t skipped{};
    size_t failed{};
    size_t total_documents{};
};

struct RetrievalUnloadResult {
    RetrievalStatus status;
    luisa::string root;
    size_t removed_documents{};
    size_t total_documents{};
};

struct RetrievalUpdateResult {
    RetrievalStatus status;
    luisa::string path;
    size_t added{};
    size_t updated{};
    size_t removed{};
    size_t unchanged{};
    size_t failed{};
    size_t total_documents{};
};

struct RetrievalQueryResult {
    RetrievalStatus status;
    luisa::vector<RetrievalHit> hits;
};

class RetrievalIndex {
public:
    explicit RetrievalIndex(RetrievalEncodeCallback encode) noexcept;
    ~RetrievalIndex() noexcept = default;

    RetrievalIndex(const RetrievalIndex &) = delete;
    RetrievalIndex(RetrievalIndex &&) = delete;
    RetrievalIndex &operator=(const RetrievalIndex &) = delete;
    RetrievalIndex &operator=(RetrievalIndex &&) = delete;

    [[nodiscard]] static luisa::vector<luisa::string> default_extensions();
    // Converts UTF-8 bytes without the Windows narrow-locale path constructor.
    // Invalid UTF-8 and embedded NUL bytes are recoverable input errors.
    [[nodiscard]] static RetrievalStatus path_from_utf8(
        luisa::string_view utf8,
        std::filesystem::path &path);
    [[nodiscard]] static RetrievalStatus normalize_extensions(
        luisa::span<const luisa::string> input,
        luisa::vector<luisa::string> &normalized);
    [[nodiscard]] static RetrievalStatus parse_extension_list(
        luisa::string_view comma_separated,
        luisa::vector<luisa::string> &normalized);

    [[nodiscard]] RetrievalLoadResult load(const std::filesystem::path &root);
    [[nodiscard]] RetrievalLoadResult load(
        const std::filesystem::path &root,
        luisa::span<const luisa::string> extensions);
    [[nodiscard]] RetrievalUnloadResult unload(const std::filesystem::path &root);
    [[nodiscard]] RetrievalUpdateResult update(const std::filesystem::path &path);
    [[nodiscard]] RetrievalQueryResult query(
        luisa::string_view text,
        uint32_t top_k = kRetrievalDefaultTopK);

    [[nodiscard]] size_t document_count() const noexcept { return _documents.size(); }
    [[nodiscard]] size_t root_count() const noexcept { return _roots.size(); }

private:
    struct RootRecord {
        std::filesystem::path path;
        std::filesystem::path comparison_path;
        luisa::vector<luisa::string> extensions;
    };

    struct DocumentRecord {
        std::filesystem::path path;
        std::filesystem::path comparison_path;
        luisa::string root_key;
        luisa::string display_path;
        luisa::string text;
        luisa::vector<float> embedding;
    };

    struct DiscoveryResult {
        luisa::vector<std::filesystem::path> files;
        size_t skipped{};
        size_t failed{};
    };

    enum class FileOutcome {
        added,
        updated,
        unchanged,
    };

    [[nodiscard]] static RetrievalStatus _normalize_path(
        const std::filesystem::path &input,
        bool require_existing,
        std::filesystem::path &display_path,
        std::filesystem::path &comparison_path,
        luisa::string &comparison_key);
    [[nodiscard]] static luisa::string _path_key(const std::filesystem::path &path);
    [[nodiscard]] static luisa::string _display_path(const std::filesystem::path &path);
    [[nodiscard]] static bool _is_within(
        const std::filesystem::path &path,
        const std::filesystem::path &root) noexcept;
    [[nodiscard]] static bool _extension_admitted(
        const std::filesystem::path &path,
        luisa::span<const luisa::string> extensions);
    [[nodiscard]] static DiscoveryResult _discover(
        const std::filesystem::path &root,
        luisa::span<const luisa::string> extensions);
    [[nodiscard]] static RetrievalStatus _read_text_file(
        const std::filesystem::path &path,
        luisa::string &text);

    [[nodiscard]] RootRecord *_find_owning_root(const std::filesystem::path &comparison_path);
    [[nodiscard]] const RootRecord *_find_owning_root(const std::filesystem::path &comparison_path) const;
    [[nodiscard]] RetrievalStatus _encode(
        luisa::string_view text,
        luisa::vector<float> &embedding) const;
    [[nodiscard]] RetrievalStatus _apply_file(
        const luisa::string &root_key,
        const std::filesystem::path &file,
        FileOutcome &outcome);
    [[nodiscard]] RetrievalUpdateResult _update_file(
        const std::filesystem::path &display_path,
        const luisa::string &comparison_key,
        RootRecord &root,
        const luisa::string &root_key);
    [[nodiscard]] RetrievalUpdateResult _update_directory(
        const std::filesystem::path &display_path,
        const std::filesystem::path &comparison_path,
        RootRecord &root,
        const luisa::string &root_key);

    RetrievalEncodeCallback _encode_callback;
    luisa::unordered_map<luisa::string, RootRecord> _roots;
    luisa::unordered_map<luisa::string, DocumentRecord> _documents;
    size_t _dimension{};
};

}// namespace luisa::embed
