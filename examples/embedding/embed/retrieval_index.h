// Host-side retrieval index for the interactive embedding retrieval
// example (retrieval plan phases 1 and 4.1-4.3; hybrid fusion phase H1; RAG
// growth report.md §2-§6). No device or DSL types cross this interface; the
// caller supplies either the dense-embedding callback (unchanged dense
// behavior) or a hybrid callback that also returns the BGE-M3 sparse lexical
// (token_id, weight) pairs.
//
// Hybrid scoring (phase H1, default weights):
//   score = kRetrievalDenseWeight * <dense dot> + kRetrievalSparseWeight * <sparse cosine>
// where the sparse term is the cosine of two L2-normalized, token-sorted, unique
// sparse vectors (so the term is already in [0, 1]). Dense-mode scoring is
// untouched: it stays the plain dense dot product, bit for bit. Per-query
// weight overrides arrive through RetrievalQueryOptions (report.md §4).
//
// RAG ingestion (report.md §2): every document is split into chunks
// (chunker.h); the index stores one DocumentRecord per chunk, grouped by the
// caller-chosen doc_id, so a hit carries the exact byte range of the passage.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <utility>

#include <luisa/core/stl/functional.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/filesystem.h>

#include "chunker.h"

namespace luisa::embed {

inline constexpr uint32_t kRetrievalDefaultTopK = 5u;
inline constexpr uint32_t kRetrievalMaxTopK = 100u;

// Hybrid fusion weights (phase H1), applied to the [0, 1]-normalized parts of a
// hybrid score. They add up to 1, so a document that matches the query on both
// halves scores at most 1. These remain the DEFAULT only; RetrievalQueryOptions
// overrides them per query (report.md §4).
inline constexpr double kRetrievalDenseWeight = 0.7;
inline constexpr double kRetrievalSparseWeight = 0.3;

// Per-call encode controls handed to the encoder callbacks (engine-mode
// max_seq_len pass-through and the opt-in ColBERT head; report.md §4/§5).
// Callback mode hosts that cannot honor them may ignore the request.
struct RetrievalEncodeRequest {
    uint32_t max_seq_len = 0u;// 0 = encoder default (512)
    bool want_colbert = false;
};

using RetrievalEncodeCallback =
    luisa::function<luisa::vector<float>(luisa::string_view utf8, const RetrievalEncodeRequest &request)>;

// Host-only dense + sparse (+ optional ColBERT) retrieval embedding. `dense` is
// the BGE-M3 CLS embedding; `sparse` holds the sparse-head lexical weights as
// (token_id, weight) pairs; `colbert` is the optional late-interaction matrix
// [(rows) * H] with per-row L2-normalized vectors (rows == 0 when absent).
// The index canonicalizes `sparse` on every path (see
// RetrievalIndex::canonicalize_sparse); an empty vector is valid and means
// "no lexical evidence". `truncated` reports that the text exceeded the
// requested max_seq_len and was truncated by the tokenizer. std::pair is not
// an STL container, so it is allowed inside luisa::vector (see scheduler.h
// for the same convention).
struct RetrievalEmbedding {
    luisa::vector<float> dense;
    luisa::vector<std::pair<uint32_t, float>> sparse;
    luisa::vector<float> colbert;
    bool truncated = false;
};

using RetrievalHybridEncodeCallback =
    luisa::function<RetrievalEmbedding(luisa::string_view utf8, const RetrievalEncodeRequest &request)>;

enum class RetrievalErrorCode {
    none,
    invalid_argument,
    not_found,
    operational,
};

// Structured failure sub-codes (report.md §6): which subsystem failed, so
// orchestrators can retry/reload appropriately without breaking the flat enum.
enum class RetrievalSubCode {
    none = 0,
    device = 1,
    model = 2,
    io = 3,
    encode = 4,
    index = 5,
    cancelled = 6,
};

struct RetrievalStatus {
    RetrievalErrorCode code{RetrievalErrorCode::none};
    luisa::string message;
    RetrievalSubCode sub_code{RetrievalSubCode::none};

    [[nodiscard]] bool ok() const noexcept { return code == RetrievalErrorCode::none; }
};

// One ranked passage (report.md §2/§4). `path` is the canonical display path
// for file-sourced documents and empty for memory-only documents; `doc_id`
// always identifies the document (caller-chosen for eer_add_document, the
// display path for directory loads). `score` is the fused ranking score;
// `dense_score` / `sparse_score` / `colbert_score` break it down (sparse and
// colbert are 0 in dense mode and when not requested, respectively).
struct RetrievalHit {
    luisa::string path;
    luisa::string doc_id;
    double score{};
    double dense_score{};
    double sparse_score{};
    double colbert_score{};
    uint32_t chunk_index{};
    uint64_t byte_offset{};
    uint64_t byte_length{};
};

struct RetrievalLoadResult {
    RetrievalStatus status;
    luisa::string root;
    luisa::vector<luisa::string> extensions;
    size_t added{};
    size_t cached{};
    size_t skipped{};
    size_t failed{};
    size_t total_documents{};// chunk records (one per chunk)
    bool cancelled = false;
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
    bool truncated = false;
};

// Result of add/remove/update_document (report.md §2). Chunk counters describe
// what the call did; `total_documents` counts chunk records afterwards and
// `total_docs` the distinct doc_ids.
struct RetrievalDocumentResult {
    RetrievalStatus status;
    luisa::string doc_id;
    size_t chunks_added{};
    size_t chunks_removed{};
    size_t chunks_unchanged{};
    size_t total_documents{};
    size_t total_docs{};
};

// Optional progress + cancellation hook for long ingestion (report.md §5).
// `on_progress` is invoked with (done, total) after every completed unit;
// `cancel_flag` (owned by the caller) is polled between units: when non-zero
// the step stops at a consistent boundary and reports the work done so far
// with RetrievalSubCode::cancelled. All NULL = the historical behavior.
struct RetrievalProgress {
    void (*on_progress)(void *user_data, uint64_t done, uint64_t total) = nullptr;
    void *user_data = nullptr;
    const volatile uint32_t *cancel_flag = nullptr;
};

// Per-query ranking controls (report.md §4). NaN or negative weights select
// the compiled-in defaults (kRetrievalDenseWeight / kRetrievalSparseWeight).
struct RetrievalQueryOptions {
    double dense_weight = std::numeric_limits<double>::quiet_NaN();
    double sparse_weight = std::numeric_limits<double>::quiet_NaN();
    luisa::string filter_json;          // empty = no filter
    uint32_t max_seq_len = 0u;          // 0 = engine default (512)
    double diversity = 0.0;             // 0..1 MMR re-ranking, 0 = off
    bool report_truncation = false;     // populate RetrievalQueryResult::truncated
    luisa::string instruction_prefix;   // prepended to the query text; empty = none
    bool want_colbert = false;          // ColBERT MaxSim re-ranking (engine mode)
};

// Production statistics snapshot (report.md §6).
struct RetrievalStats {
    size_t document_count = 0;  // distinct doc_ids
    size_t chunk_count = 0;     // chunk records
    uint32_t dimension = 0;
    bool hybrid = false;
    double cache_hit_ratio = 0.0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t encode_count = 0;  // chunks actually encoded (not cache-served)
    double mean_encode_ms = 0.0;
};

// Optional disk-cache hook for directory import (retrieval example): consulted
// per file/chunk before the encoder runs, so content whose embedding is
// already cached skips the encode and is served from disk instead. The index
// calls begin/end_directory around every directory session (load/update); all
// methods run on the service thread with the index lock held.
//
// Hybrid caches (phase H1) additionally implement the sparse sidecar hooks
// below. They are non-pure with safe defaults, so existing dense-only caches
// stay source-compatible: a hybrid index then never hits and always re-encodes.
//
// Chunked ingestion (report.md §2) addresses entries through the `_variant`
// overloads: `variant` carries (chunk_index, chunk_size, chunk_overlap, by),
// so the cache keys on (content hash, chunk_index, chunk params). The default
// implementations keep source compatibility: a non-empty variant misses, an
// empty variant forwards to the legacy whole-file hooks (byte-identical v1
// behavior).
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
    // Variant (chunk) addressing; see the class comment.
    [[nodiscard]] virtual bool load_variant(
        const luisa::filesystem::path &file,
        luisa::string_view contents,
        luisa::string_view variant,
        luisa::vector<float> &embedding) noexcept {
        return variant.empty() ? load(file, contents, embedding) : false;
    }
    virtual void store_variant(
        const luisa::filesystem::path &file,
        luisa::string_view contents,
        luisa::string_view variant,
        luisa::span<const float> embedding) noexcept {
        if (variant.empty()) { store(file, contents, embedding); }
    }
    [[nodiscard]] virtual bool load_sparse_variant(
        const luisa::filesystem::path &file,
        luisa::string_view contents,
        luisa::string_view variant,
        luisa::vector<std::pair<uint32_t, float>> &sparse) noexcept {
        return variant.empty() ? load_sparse(file, contents, sparse) : false;
    }
    virtual void store_sparse_variant(
        const luisa::filesystem::path &file,
        luisa::string_view contents,
        luisa::string_view variant,
        luisa::span<const std::pair<uint32_t, float>> sparse) noexcept {
        if (variant.empty()) { store_sparse(file, contents, sparse); }
    }
    // Ends the directory session; persist meta information here.
    virtual void end_directory() noexcept = 0;
};

class RetrievalIndex {
public:
    // Dense-only index (unchanged behavior): scores are plain dense dot products.
    explicit RetrievalIndex(RetrievalEncodeCallback encode) noexcept;
    // Hybrid index (phase H1): scores fuse the dense dot product with the sparse
    // cosine using kRetrievalDenseWeight / kRetrievalSparseWeight (defaults).
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
    // Validates that `json` is a well-formed JSON object (or empty). Used for
    // document metadata at ingest and for query filters (which additionally
    // go through parse_filter).
    [[nodiscard]] static RetrievalStatus validate_metadata_json(
        luisa::string_view json,
        luisa::string_view what);
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

    // Metadata filter predicate tree (report.md §4): equality and
    // $gte/$lte/$gt/$lt/$in leaves on top-level keys, combined with $and/$or.
    // Public so the JSON parser helpers can build it; not part of the C ABI.
    struct Filter {
        enum class Op : uint32_t {
            eq, gte, lte, gt, lt, in_list, logical_and, logical_or,
        };
        Op op = Op::logical_and;
        luisa::string key;    // leaf: top-level metadata key
        bool is_number = false;
        double number = 0.0;
        luisa::string text;   // leaf string value
        luisa::vector<double> numbers;
        luisa::vector<luisa::string> texts;
        luisa::vector<luisa::unique_ptr<Filter>> children;

        [[nodiscard]] bool matches(luisa::string_view metadata_json) const;
    };

    // Parses the metadata-filter JSON grammar (report.md §4): top-level keys
    // with scalar equality values, {key: {"$gte": ...}} comparison leaves and
    // "$in" lists, combined by implicit/explicit $and and $or.
    [[nodiscard]] static RetrievalStatus parse_filter(
        luisa::string_view json,
        luisa::unique_ptr<Filter> &out);

    // Whole-file directory load (historical behavior; chunk_by == off).
    [[nodiscard]] RetrievalLoadResult load(const luisa::filesystem::path &root);
    [[nodiscard]] RetrievalLoadResult load(
        const luisa::filesystem::path &root,
        luisa::span<const luisa::string> extensions);
    // Chunked directory load (report.md §2): files are split per `options`
    // before encoding. `progress` may be null.
    [[nodiscard]] RetrievalLoadResult load(
        const luisa::filesystem::path &root,
        luisa::span<const luisa::string> extensions,
        const ChunkOptions &options,
        const RetrievalProgress *progress = nullptr);
    [[nodiscard]] RetrievalUnloadResult unload(const luisa::filesystem::path &root);
    [[nodiscard]] RetrievalUpdateResult update(const luisa::filesystem::path &path);

    // In-memory document ingestion (report.md §2). `doc_id` is caller-chosen
    // (non-empty, valid UTF-8, no NUL bytes); re-ingesting the same doc_id
    // replaces its chunks transactionally. `metadata_json` must be empty or a
    // JSON object and is kept verbatim for filters and snapshots.
    [[nodiscard]] RetrievalDocumentResult add_document(
        luisa::string doc_id,
        luisa::string_view text,
        luisa::string_view metadata_json,
        const ChunkOptions &options,
        const RetrievalProgress *progress = nullptr);
    [[nodiscard]] RetrievalDocumentResult update_document(
        luisa::string doc_id,
        luisa::string_view text,
        luisa::string_view metadata_json,
        const ChunkOptions &options,
        const RetrievalProgress *progress = nullptr) {
        return add_document(std::move(doc_id), text, metadata_json, options, progress);
    }
    // Drops every chunk of `doc_id`; unknown ids are a not_found error.
    [[nodiscard]] RetrievalDocumentResult remove_document(luisa::string_view doc_id);

    // Plain query (historical behavior: default weights, no filter, no MMR).
    [[nodiscard]] RetrievalQueryResult query(
        luisa::string_view text,
        uint32_t top_k = kRetrievalDefaultTopK);
    // Query with ranking controls (report.md §4). `options` may be null.
    [[nodiscard]] RetrievalQueryResult query(
        luisa::string_view text,
        uint32_t top_k,
        const RetrievalQueryOptions *options);

    // Corpus-level snapshot persistence (report.md §3): one JSON manifest at
    // `path` plus a packed-float32 blob at `path` + ".bin". Restores roots,
    // documents, chunks, metadata and embeddings with no re-encode; memory-only
    // documents are serialized with their text. A mismatching or corrupt
    // snapshot is a clean operational error and leaves the index untouched.
    [[nodiscard]] RetrievalStatus save(const luisa::filesystem::path &path);
    [[nodiscard]] RetrievalStatus load_snapshot(const luisa::filesystem::path &path);

    [[nodiscard]] size_t document_count() const noexcept { return _doc_chunks.size(); }
    [[nodiscard]] size_t chunk_count() const noexcept { return _documents.size(); }
    [[nodiscard]] RetrievalStats stats() const noexcept;

    // Optional disk cache; nullptr (default) disables caching. Not owned.
    void set_embed_cache(RetrievalEmbedCache *cache) noexcept { _embed_cache = cache; }

    // True when this index was built from a hybrid encoder.
    [[nodiscard]] bool hybrid() const noexcept { return _hybrid; }

      // Opt-in ColBERT matrix storage (report.md §5; bounds RAM). When false,
      // encode-time colbert payloads are dropped and want_colbert queries fail.
      // Records without a stored matrix (e.g. restored from a snapshot that
      // predates ColBERT serialization) degrade to the fused dense/sparse
      // score instead of failing the query.
    void set_store_colbert(bool enabled) noexcept { _want_colbert = enabled; }
    [[nodiscard]] bool store_colbert() const noexcept { return _want_colbert; }

private:
    struct RootRecord {
        luisa::filesystem::path path;
        luisa::filesystem::path comparison_path;
        luisa::vector<luisa::string> extensions;
        ChunkOptions chunk_options;
    };

    // One chunk record; chunk 0 of a document additionally carries the full
    // document text (the others keep theirs empty to bound RAM).
    struct DocumentRecord {
        luisa::filesystem::path path;         // empty for memory-only documents
        luisa::filesystem::path comparison_path;
        luisa::string root_key;               // empty for memory-only documents
        luisa::string display_path;           // empty for memory-only documents
        luisa::string doc_id;
        luisa::string metadata_json;          // raw JSON, validated at ingest
        luisa::string text;                   // full document text (chunk 0 only)
        uint32_t chunk_index = 0;
        uint64_t byte_offset = 0;
        uint64_t byte_length = 0;
        ChunkOptions chunk_options;// chunk 0 only (idempotent re-ingest check)
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
        uint32_t max_seq_len,
        bool want_colbert,
        RetrievalEmbedding &embedding);
    // Two-pointer merge over canonical (sorted, unique, L2-normalized) sparse
    // vectors; equals their cosine because both norms are 1.
    [[nodiscard]] static double _sparse_dot(
        luisa::span<const std::pair<uint32_t, float>> lhs,
        luisa::span<const std::pair<uint32_t, float>> rhs) noexcept;
    // ColBERT MaxSim: sum over query rows of the max dot product against the
    // document rows (both row-normalized). `colbert` is [(rows)*dimension].
    [[nodiscard]] static double _colbert_maxsim(
        luisa::span<const float> query,
        luisa::span<const float> document,
        uint32_t dimension) noexcept;
    // Encoded-chunk accounting + latency sampling (report.md §6).
    void _record_encode(double elapsed_ms, bool cache_hit) noexcept;
    [[nodiscard]] RetrievalStatus _apply_file(
        const luisa::string &root_key,
        const luisa::filesystem::path &file,
        FileOutcome &outcome,
        bool &cache_hit,
        const RetrievalProgress *progress);
    // Shared ingestion core: chunks + encodes + transactionally replaces the
    // document's chunk records. `base_key` is the comparison key (files) or
    // empty (memory-only); `path`/`display` are empty for memory-only docs.
    [[nodiscard]] RetrievalStatus _ingest_document(
        luisa::string doc_id,
        const luisa::filesystem::path &path,
        const luisa::filesystem::path &comparison_path,
        const luisa::string &root_key,
        luisa::string_view text,
        luisa::string_view metadata_json,
        const ChunkOptions &options,
        luisa::vector<luisa::string> &removed_keys,
        luisa::vector<luisa::string> &added_keys,
        size_t &chunks_unchanged,
        size_t &served_count,
        const RetrievalProgress *progress);
    [[nodiscard]] static luisa::string _chunk_key(
        const luisa::string &base_key,
        uint32_t chunk_index);
    [[nodiscard]] static luisa::string _chunk_variant(
        uint32_t chunk_index,
        const ChunkOptions &options);
    void _drop_document(luisa::string_view doc_id) noexcept;
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
    bool _want_colbert{false};
    RetrievalEmbedCache *_embed_cache{nullptr};
    luisa::unordered_map<luisa::string, RootRecord> _roots;
    // Chunk records keyed by _chunk_key (whole-file docs keep the legacy
    // comparison key so pre-chunking behavior is untouched).
    luisa::unordered_map<luisa::string, DocumentRecord> _documents;
    // doc_id -> ordered chunk keys (index 0 is chunk 0).
    luisa::unordered_map<luisa::string, luisa::vector<luisa::string>> _doc_chunks;
    size_t _dimension{};
    // Statistics counters (report.md §6).
    uint64_t _encode_count{0};
    uint64_t _cache_hits{0};
    uint64_t _cache_misses{0};
    double _encode_time_total_ms{0.0};
};

}// namespace luisa::embed
