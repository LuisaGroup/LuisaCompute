// Host-side whole-file retrieval index for the interactive embedding retrieval
// example (retrieval plan phases 1 and 4.1-4.3; hybrid fusion phase H1). No
// device or DSL types cross this interface; the caller supplies either the
// dense-embedding callback (unchanged dense behavior) or a hybrid callback that
// also returns the BGE-M3 sparse lexical (token_id, weight) pairs.
//
// Hybrid scoring (phase H1):
//   score = kRetrievalDenseWeight * <dense dot> + kRetrievalSparseWeight * <sparse cosine>
// where the sparse term is the cosine of two L2-normalized, token-sorted, unique
// sparse vectors (so the term is already in [0, 1]). Dense-mode scoring is
// untouched: it stays the plain dense dot product, bit for bit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <utility>

#include <luisa/core/stl/functional.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/filesystem.h>

namespace luisa::embed {

inline constexpr uint32_t kRetrievalDefaultTopK = 5u;
inline constexpr uint32_t kRetrievalMaxTopK = 100u;

// Hybrid fusion weights (phase H1), applied to the [0, 1]-normalized parts of a
// hybrid score. They add up to 1, so a document that matches the query on both
// halves scores at most 1.
inline constexpr double kRetrievalDenseWeight = 0.7;
inline constexpr double kRetrievalSparseWeight = 0.3;

using RetrievalEncodeCallback = luisa::function<luisa::vector<float>(luisa::string_view utf8)>;

// Host-only dense + sparse retrieval embedding (phase H1). `dense` is the BGE-M3
// CLS embedding; `sparse` holds the sparse-head lexical weights as
// (token_id, weight) pairs. The index canonicalizes `sparse` on every path
// (see RetrievalIndex::canonicalize_sparse); an empty vector is valid and means
// "no lexical evidence". std::pair is not an STL container, so it is allowed
// inside luisa::vector (see scheduler.h for the same convention).
struct RetrievalEmbedding {
    luisa::vector<float> dense;
    luisa::vector<std::pair<uint32_t, float>> sparse;
};

using RetrievalHybridEncodeCallback =
    luisa::function<RetrievalEmbedding(luisa::string_view utf8)>;

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
    size_t cached{};
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
    size_t cached{};
    size_t failed{};
    size_t total_documents{};
};

struct RetrievalQueryResult {
    RetrievalStatus status;
    luisa::vector<RetrievalHit> hits;
};

// Optional disk-cache hook for directory import (retrieval example): consulted
// per file before the encoder runs, so files whose content hash is already
// cached skip the encode and are served from disk instead. The index calls
// begin/end_directory around every directory session (load/update); all
// methods run on the service thread with the index lock held.
//
// Hybrid caches (phase H1) additionally implement the sparse sidecar hooks
// below. They are non-pure with safe defaults, so existing dense-only caches
// stay source-compatible: a hybrid index then never hits and always re-encodes.
class RetrievalEmbedCache {
public:
    virtual ~RetrievalEmbedCache() noexcept = default;

    // Starts a directory session. `root` is the canonical display root.
    virtual void begin_directory(
        const luisa::filesystem::path &root,
        luisa::span<const luisa::string> extensions) noexcept = 0;
    // Serves the cached embedding for `contents` of `file`; false on a miss.
    [[nodiscard]] virtual bool load(
        const luisa::filesystem::path &file,
        luisa::string_view contents,
        luisa::vector<float> &embedding) noexcept = 0;
    // Persists a freshly encoded embedding for `file`.
    virtual void store(
        const luisa::filesystem::path &file,
        luisa::string_view contents,
        luisa::span<const float> embedding) noexcept = 0;
    // Serves the cached sparse lexical pairs for `contents` of `file`. Returns
    // false when this cache has no (valid) sparse sidecar for the entry. A hit
    // is only usable together with a dense hit: the index re-encodes both when
    // either half is missing or invalid. The default implementation reports a
    // miss, which keeps dense-only caches working in hybrid mode.
    [[nodiscard]] virtual bool load_sparse(
        const luisa::filesystem::path &file,
        luisa::string_view contents,
        luisa::vector<std::pair<uint32_t, float>> &sparse) noexcept {
        (void)file;
        (void)contents;
        (void)sparse;
        return false;
    }
    // Persists the sparse canonical pairs for a freshly hybrid-encoded `file`.
    // Only called after a successful hybrid encode, so dense-only usage never
    // creates sidecars. The default implementation does nothing.
    virtual void store_sparse(
        const luisa::filesystem::path &file,
        luisa::string_view contents,
        luisa::span<const std::pair<uint32_t, float>> sparse) noexcept {
        (void)file;
        (void)contents;
        (void)sparse;
    }
    // Ends the directory session; persist meta information here.
    virtual void end_directory() noexcept = 0;
};

class RetrievalIndex {
public:
    // Dense-only index (unchanged behavior): scores are plain dense dot products.
    explicit RetrievalIndex(RetrievalEncodeCallback encode) noexcept;
    // Hybrid index (phase H1): scores fuse the dense dot product with the sparse
    // cosine using kRetrievalDenseWeight / kRetrievalSparseWeight.
    explicit RetrievalIndex(RetrievalHybridEncodeCallback encode) noexcept;
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
        luisa::filesystem::path &path);
    [[nodiscard]] static RetrievalStatus normalize_extensions(
        luisa::span<const luisa::string> input,
        luisa::vector<luisa::string> &normalized);
    [[nodiscard]] static RetrievalStatus parse_extension_list(
        luisa::string_view comma_separated,
        luisa::vector<luisa::string> &normalized);
    // Canonicalizes a sparse (token_id, weight) vector: non-finite and negative
    // weights are rejected as recoverable errors, exact zeros are dropped,
    // duplicate token ids are combined with a double sum (also rejected when the
    // sum is not finite), the result is sorted by token_id, and its weights are
    // L2-normalized in double, so a dot product of two canonical vectors is
    // their sparse cosine in [0, 1]. An empty or all-zero input is valid and
    // yields an empty vector (norm zero). Safe to apply to encoder output and
    // to cache-loaded values alike.
    [[nodiscard]] static RetrievalStatus canonicalize_sparse(
        luisa::span<const std::pair<uint32_t, float>> input,
        luisa::vector<std::pair<uint32_t, float>> &output);

    [[nodiscard]] RetrievalLoadResult load(const luisa::filesystem::path &root);
    [[nodiscard]] RetrievalLoadResult load(
        const luisa::filesystem::path &root,
        luisa::span<const luisa::string> extensions);
    [[nodiscard]] RetrievalUnloadResult unload(const luisa::filesystem::path &root);
    [[nodiscard]] RetrievalUpdateResult update(const luisa::filesystem::path &path);
    [[nodiscard]] RetrievalQueryResult query(
        luisa::string_view text,
        uint32_t top_k = kRetrievalDefaultTopK);

    [[nodiscard]] size_t document_count() const noexcept { return _documents.size(); }
    [[nodiscard]] size_t root_count() const noexcept { return _roots.size(); }

    // Optional disk cache; nullptr (default) disables caching. Not owned.
    void set_embed_cache(RetrievalEmbedCache *cache) noexcept { _embed_cache = cache; }

    // True when this index was built from a hybrid encoder.
    [[nodiscard]] bool hybrid() const noexcept { return _hybrid; }

private:
    struct RootRecord {
        luisa::filesystem::path path;
        luisa::filesystem::path comparison_path;
        luisa::vector<luisa::string> extensions;
    };

    struct DocumentRecord {
        luisa::filesystem::path path;
        luisa::filesystem::path comparison_path;
        luisa::string root_key;
        luisa::string display_path;
        luisa::string text;
        // Dense always populated; sparse only in hybrid mode (dense mode leaves
        // it empty, which keeps dense-only ranking untouched).
        RetrievalEmbedding embedding;
    };

    struct DiscoveryResult {
        luisa::vector<luisa::filesystem::path> files;
        size_t skipped{};
        size_t failed{};
    };

    enum class FileOutcome {
        added,
        updated,
        unchanged,
    };

    [[nodiscard]] static RetrievalStatus _normalize_path(
        const luisa::filesystem::path &input,
        bool require_existing,
        luisa::filesystem::path &display_path,
        luisa::filesystem::path &comparison_path,
        luisa::string &comparison_key);
    [[nodiscard]] static luisa::string _path_key(const luisa::filesystem::path &path);
    [[nodiscard]] static luisa::string _display_path(const luisa::filesystem::path &path);
    [[nodiscard]] static bool _is_within(
        const luisa::filesystem::path &path,
        const luisa::filesystem::path &root) noexcept;
    [[nodiscard]] static bool _extension_admitted(
        const luisa::filesystem::path &path,
        luisa::span<const luisa::string> extensions);
    [[nodiscard]] static DiscoveryResult _discover(
        const luisa::filesystem::path &root,
        luisa::span<const luisa::string> extensions);
    [[nodiscard]] static RetrievalStatus _read_text_file(
        const luisa::filesystem::path &path,
        luisa::string &text);

    [[nodiscard]] RootRecord *_find_owning_root(const luisa::filesystem::path &comparison_path);
    [[nodiscard]] const RootRecord *_find_owning_root(const luisa::filesystem::path &comparison_path) const;
    [[nodiscard]] RetrievalStatus _validate_dense(
        luisa::span<const float> dense) const;
    [[nodiscard]] RetrievalStatus _encode(
        luisa::string_view text,
        RetrievalEmbedding &embedding) const;
    // Two-pointer merge over canonical (sorted, unique, L2-normalized) sparse
    // vectors; equals their cosine because both norms are 1.
    [[nodiscard]] static double _sparse_dot(
        luisa::span<const std::pair<uint32_t, float>> lhs,
        luisa::span<const std::pair<uint32_t, float>> rhs) noexcept;
    [[nodiscard]] RetrievalStatus _apply_file(
        const luisa::string &root_key,
        const luisa::filesystem::path &file,
        FileOutcome &outcome,
        bool &cache_hit);
    [[nodiscard]] RetrievalUpdateResult _update_file(
        const luisa::filesystem::path &display_path,
        const luisa::string &comparison_key,
        RootRecord &root,
        const luisa::string &root_key);
    [[nodiscard]] RetrievalUpdateResult _update_directory(
        const luisa::filesystem::path &display_path,
        const luisa::filesystem::path &comparison_path,
        RootRecord &root,
        const luisa::string &root_key);

    RetrievalEncodeCallback _encode_callback;
    RetrievalHybridEncodeCallback _hybrid_encode_callback;
    bool _hybrid{false};
    RetrievalEmbedCache *_embed_cache{nullptr};
    luisa::unordered_map<luisa::string, RootRecord> _roots;
    luisa::unordered_map<luisa::string, DocumentRecord> _documents;
    size_t _dimension{};
};

}// namespace luisa::embed
