/*
 * example_embed_retrieval_lib -- C ABI for the bge-m3 whole-file embedding
 * retrieval engine (examples/embedding/embed).
 *
 * The executable `example_embed_retrieval` drives the engine through an
 * interactive console or a JSON-RPC server. This library exposes exactly the
 * same steps -- initialize, load a directory, query, update, unload, encode,
 * shut down -- as a flat, stable C ABI, so the retriever can be hosted by any
 * language or process through its FFI (Python ctypes/cffi, C# DllImport, Rust
 * bindgen, Java JNA, ...). No C++ type, STL container, or LuisaCompute handle
 * crosses the boundary: every entry point takes and returns only C scalars,
 * `const char *` UTF-8 strings, and plain C structs.
 *
 * Layout of this header
 * ---------------------
 *   1.  Configuration (version, limits, export macro)
 *   2.  Status codes (+ 1.2 structured sub-codes)
 *   3.  Opaque context handle
 *   4.  Value types (hits, results, errors, 1.2 options structs)
 *   5.  Host-encoder callbacks (optional, GPU-free mode; 1.2 sparse pair)
 *   6.  Lifecycle:  eer_context_create / eer_context_destroy
 *   7.  Steps:      eer_load_dir(_ex) / eer_query(_ex) / eer_update_path /
 *                    eer_unload_dir / eer_add_document / eer_update_document /
 *                    eer_remove_document
 *   8.  Extras:     eer_encode_text / eer_encode_batch / eer_encode_colbert /
 *                    eer_index_save / eer_index_load / eer_stats / ping /
 *                    device memory / park-resume / introspection
 *   9.  Memory management: every `_free` entry point
 *
 * Ownership and memory
 * --------------------
 *   - `eer_context *` is created by eer_context_create() and released by
 *     eer_context_destroy(). It owns the backend device, the model session and
 *     the retrieval index.
 *   - Every payload struct (eer_load_result, eer_unload_result,
 *     eer_update_result, eer_query_result, eer_error) is caller-allocated
 *     (usually on the stack). The library fills it in and may allocate strings
 *     or a hit array inside it; those allocations belong to the caller from
 *     then on and MUST be released with the matching `_free` function:
 *
 *         eer_query_result result;
 *         if (eer_query(&ctx, text, len, 5u, &result) == EER_STATUS_OK) { ... }
 *         eer_query_result_free(&result);   // always, success or failure
 *
 *     The `_free` functions are idempotent (a zeroed struct is a no-op), so
 *     `_free` may safely be called after a failed call. Every allocation comes
 *     from the library's own allocator (not necessarily the C runtime `malloc`),
 *     so payload memory MUST be released with the matching `_free` function --
 *     passing it to C `free()` / the host language's deallocator is undefined
 *     behavior across a shared-library boundary.
 *   - No pointer returned by this library stays valid across another call on
 *     the same context, except `eer_version()`, `eer_status_string()` and
 *     `eer_default_extensions()`, which return process-lifetime constants, and
 *     `eer_context_backend()`, whose string lives as long as the context.
 *
 * Threading
 * ---------
 *   All step functions (load/query/update/unload/encode) are safe to call
 *   concurrently from multiple host threads on the same context: the engine
 *   serializes index mutation and batching internally, exactly like the RPC
 *   server does. eer_context_create() and eer_context_destroy() are NOT
 *   thread-safe: no other call may be in flight while they run.
 *
 * Strings
 * -------
 *   Every textual argument is UTF-8. Paths are converted with the same
 *   UTF-8 -> filesystem conversion the executable uses, so non-ASCII paths work
 *   on Windows (no ANSI code-page round trip) and embedded NUL bytes are
 *   rejected as invalid input instead of truncating silently. Query text is
 *   passed as an explicit (pointer, byte length) pair because it may legally
 *   contain any UTF-8 byte sequence and need not be NUL-terminated.
 *
 * Build / link
 * ------------
 *   The library is built by xmake as the target `example_embed_retrieval_lib`
 *   (kind = shared) and lands next to the other binaries, e.g.
 *   `bin/release/example_embed_retrieval_lib.dll` on Windows and
 *   `bin/release/libexample_embed_retrieval_lib.so` elsewhere. Consumers must
 *   also ship / install the LuisaCompute runtime and the selected backend
 *   modules next to the host executable (see eer_create_info::program_path).
 *
 * Minimal example (engine mode, one directory, one query)
 * -------------------------------------------------------
 *     eer_create_info info = {0};
 *     info.struct_size  = sizeof(info);
 *     info.model_path   = "D:/models/bge-m3/model.bgm";
 *     info.tokenizer_path = "D:/models/bge-m3/tokenizer.json";
 *     info.backend      = "vk";
 *
 *     eer_context *ctx = NULL;
 *     eer_error error = {0};
 *     if (eer_context_create(&info, &ctx, &error) != EER_STATUS_OK) {
 *         fprintf(stderr, "init failed: %s\n", error.message);
 *         eer_error_free(&error);
 *         return 1;
 *     }
 *
 *     eer_load_result load = {0};
 *     if (eer_load_dir(ctx, "D:/docs", NULL, &load) == EER_STATUS_OK) {
 *         printf("indexed %zu document(s) from %s\n", load.total_documents, load.root);
 *     }
 *     eer_load_result_free(&load);
 *
 *     eer_query_result hits = {0};
 *     if (eer_query(ctx, "how do I mount a texture?", 25u, 5u, &hits) == EER_STATUS_OK) {
 *         for (size_t i = 0; i < hits.hit_count; i++) {
 *             printf("%zu %.6f %s\n", i + 1u, hits.hits[i].score, hits.hits[i].path);
 *         }
 *     }
 *     eer_query_result_free(&hits);
 *
 *     eer_context_destroy(ctx);
 *
 * See examples/embedding/api/README.md for the per-step contract, the
 * GPU-free callback mode, runnable C / Python examples and the error table.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

/* ==========================================================================
 * 1. Configuration
 * ========================================================================== */

/* Semantic version of this C ABI. Bumped on any incompatible change; adding
 * new entry points or new trailing struct fields with a default meaning is
 * compatible and does not bump the major number. */
#define EER_VERSION_MAJOR 1
/* 1.1: result payloads are now allocated with the library's own allocator and
 * MUST be released through the matching eer_*_free (C free() is no longer
 * guaranteed to work across the shared-library boundary); num_workers == 0
 * now selects the hardware concurrency instead of a fixed count of 2.
 * 1.2 (RAG growth, report.md §2-§6): chunked document ingestion with passage
 * byte ranges (eer_chunk_options / eer_add_document / eer_update_document /
 * eer_remove_document / eer_load_dir_ex), corpus snapshots (eer_index_save /
 * eer_index_load), ranking control (eer_query_ex: per-query hybrid weights,
 * metadata filter_json, MMR diversity, per-query max_seq_len + truncation
 * reporting, instruction prefix, ColBERT re-ranking), batch encoding
 * (eer_encode_batch), ColBERT encoding (eer_encode_colbert), ingestion
 * progress + cancellation (eer_progress), production observability
 * (structured EER_SUB_* sub-codes on every result payload, eer_stats,
 * eer_context_ping, eer_context_device_memory, eer_context_park /
 * eer_context_resume), sparse host-encoder callback (hybrid callback mode)
 * and the EER_CREATE_WANT_COLBERT create flag. Everything is additive:
 * new trailing struct fields are gated on struct_size, new entry points
 * extend the export table, and a zero-initialized options struct keeps the
 * historical behavior. */
#define EER_VERSION_MINOR 2
#define EER_VERSION_PATCH 0

/* Default and maximum `top_k` accepted by eer_query(). A `top_k` of 0 in a
 * query request selects EER_DEFAULT_TOP_K. */
#define EER_DEFAULT_TOP_K 5u
#define EER_MAX_TOP_K 100u

/* Extension rule used when a load request passes extensions == NULL.
 * (The same default the console documents.) */
#define EER_DEFAULT_EXTENSIONS ".txt,.md"

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(EXAMPLE_EMBED_RETRIEVAL_LIB_BUILD)
#define EER_API __declspec(dllexport)
#else
#define EER_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define EER_API __attribute__((visibility("default")))
#else
#define EER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * 2. Status codes
 * ========================================================================== */

/* Every entry point that can fail reports through this code. Steps that also
 * produce a payload return the status as their return value and write the
 * payload (plus a diagnostic message on failure) into the caller's out struct. */
typedef enum eer_status {
    /* The requested step completed; all out fields are valid. */
    EER_STATUS_OK = 0,
    /* A caller-supplied argument was rejected (NULL pointer where a value is
     * required, empty backend name, extension rule that is not a plain,
     * case-insensitive extension list, `top_k` outside [1, EER_MAX_TOP_K],
     * `struct_size` too small, ...). The engine was not touched. */
    EER_STATUS_INVALID_ARGUMENT = 1,
    /* A referenced path is missing or is not the expected kind of file: a model
     * or tokenizer required by eer_context_create(), a root that is not
     * currently loaded (eer_unload_dir), or an update path outside every
     * loaded root (eer_update_path). */
    EER_STATUS_NOT_FOUND = 2,
    /* The step was well formed but the engine could not complete it: no GPU
     * backend available, encoder failure, unreadable file inside a directory,
     * empty corpus, dimension mismatch, disk-cache I/O error, ... The message
     * describes the specific cause. */
    EER_STATUS_OPERATIONAL = 3,
    /* The `eer_context *` argument is NULL -- or refers to an already destroyed
     * context, which the library detects on a best-effort basis only. Using a
     * destroyed handle is undefined behavior; this code exists to turn the
     * common mistakes into a diagnostic instead of a crash. Always a
     * programming error on the caller side. */
    EER_STATUS_INVALID_HANDLE = 4
} eer_status;

/* Stable, human-readable name of a status code ("ok", "invalid_argument",
 * ...). Never NULL; unknown values map to "unknown". Process-lifetime string. */
EER_API const char *eer_status_string(eer_status status);

/* Structured failure sub-codes (report.md §6): which subsystem failed, so
 * orchestrators can retry/reload appropriately without breaking the flat
 * eer_status enum. Every result payload carries an OPTIONAL trailing
  * `int32_t sub_code` field (append-only ABI growth): it holds one of these
  * values on failure and EER_SUB_NONE (0) otherwise / on success. Every
  * failure path writes it explicitly (plain argument/handle errors report
  * EER_SUB_NONE), so it is never left unset. Callers compiled against the
  * 1.1 header simply do not see the field. */
typedef enum eer_sub_code {
    EER_SUB_NONE = 0,     /* no sub-code (success or a plain argument error) */
    EER_SUB_DEVICE = 1,   /* GPU/backend device creation or loss */
    EER_SUB_MODEL = 2,    /* model / tokenizer load or validation failure */
    EER_SUB_IO = 3,       /* file system / snapshot / cache I/O */
    EER_SUB_ENCODE = 4,   /* the encoder (session or host callback) failed */
    EER_SUB_INDEX = 5,    /* index inconsistency (dimensions, corrupt data) */
    EER_SUB_CANCELLED = 6 /* the caller cancelled a long-running step */
} eer_sub_code;

/* Stable, human-readable name of a sub-code ("none", "device", ...).
 * Process-lifetime string, never NULL; unknown values map to "unknown". */
EER_API const char *eer_sub_code_string(int32_t sub_code);

/* Version of the loaded library as "major.minor.patch". Use this at runtime to
 * confirm that the binary you loaded matches the header you compiled against.
 * Process-lifetime string, never NULL. */
EER_API const char *eer_version(void);

/* EER_DEFAULT_EXTENSIONS as a string, so bindings do not have to hard-code the
 * rule. Process-lifetime string, never NULL. */
EER_API const char *eer_default_extensions(void);

/* ==========================================================================
 * 3. Opaque context handle
 * ========================================================================== */

/* One initialized retriever: backend device + model session + document index.
 * Created by eer_context_create(), destroyed by eer_context_destroy(); treat
 * the pointer as opaque. */
typedef struct eer_context eer_context;

/* ==========================================================================
 * 4. Value types
 * ========================================================================== */

/* Diagnostics of a call that returns a handle or plain value instead of a
 * result struct (eer_context_create(), eer_index_save/load, eer_encode_text,
 * eer_encode_batch, eer_encode_colbert, eer_context_ping/device_memory/
 * park/resume). `message` is a library-allocated UTF-8 string on failure and
 * NULL on success; release it with eer_error_free(). `sub_code` (1.2,
 * appended) is EER_SUB_NONE on success. */
typedef struct eer_error {
    char *message;
    int32_t sub_code;
} eer_error;

/* One ranked document returned by eer_query(). `path` is the canonical display
 * path the index tracks for the file (UTF-8, library-allocated, owned by the enclosing
 * eer_query_result). `score` is the cosine similarity between the query
 * embedding and the document embedding; dense embeddings are L2-normalized by
 * the engine, so the score is in [-1, 1] and results are sorted by descending
 * score, ties broken by ascending path. */
/* One ranked PASSAGE returned by eer_query()/eer_query_ex() (1.2 turns the
 * whole-file hit into a located chunk, report.md §2/§4). `path` is the
 * canonical display path for file-sourced documents and NULL for memory-only
 * documents (eer_add_document); `doc_id` always identifies the document
 * (caller-chosen for eer_add_document, the display path for directory loads;
 * for a plain whole-file document both name the file). `chunk_index`,
 * `byte_offset` and `byte_length` locate the passage: the chunk text is
 * source[byte_offset, byte_offset + byte_length). `score` is the fused
 * ranking score; `dense_score` / `sparse_score` / `colbert_score` break it
 * down (sparse is 0 in dense mode; colbert is 0 unless ColBERT re-ranking
 * was requested). All strings are library-allocated and owned by the
 * enclosing eer_query_result. Results are sorted by descending score, ties
 * broken by ascending (doc_id, chunk_index). */
typedef struct eer_hit {
    char *path;
    double score;
    char *doc_id;       /* 1.2, appended (library-allocated, never NULL) */
    uint32_t chunk_index;
    uint64_t byte_offset;
    uint64_t byte_length;
    double dense_score;
    double sparse_score;
    double colbert_score;
} eer_hit;

/* Result of eer_load_dir(): a directory import.
 *
 *   added            documents encoded and inserted by this call
 *   cached           documents served from the on-disk embedding cache
 *                    (.lc_embed_cache*) instead of being re-encoded
 *   skipped          regular files that did not match the extension rule
 *   failed           files that matched but could not be read, decoded or
 *                    encoded; the successfully indexed siblings are kept
 *   total_documents  documents tracked by the index after the call (all roots)
 *   root             canonical display path of the newly loaded root
 *   extensions       normalized, comma-separated extension rule that now
 *                    governs this root (".md,.txt")
 *   message          failure diagnostics, NULL on success
 *
 * A root owns its extension rule; duplicate, nested or overlapping roots are
 * rejected with EER_STATUS_INVALID_ARGUMENT. Release with
 * eer_load_result_free(). */
typedef struct eer_load_result {
    char *message;
    char *root;
    char *extensions;
    size_t added;
    size_t cached;
    size_t skipped;
    size_t failed;
    size_t total_documents;
    int32_t sub_code;   /* 1.2, appended; EER_SUB_CANCELLED marks a cancelled
                           partial load (counters still report the work done) */
} eer_load_result;

/* Result of eer_unload_dir(): removal of exactly one loaded root.
 *
 *   removed_documents  documents dropped by this call
 *   total_documents    documents still tracked afterwards
 *   root               canonical display path of the removed root
 *   message            failure diagnostics, NULL on success
 *
 * Release with eer_unload_result_free(). */
typedef struct eer_unload_result {
    char *message;
    char *root;
    size_t removed_documents;
    size_t total_documents;
    int32_t sub_code;   /* 1.2, appended */
} eer_unload_result;

/* Result of eer_update_path(): re-index one file or one subdirectory in place.
 *
 *   added      new files inserted
 *   updated    files whose content changed, re-encoded and replaced
 *   removed    tracked files that no longer exist, dropped
 *   unchanged  byte-identical files, not re-encoded
 *   cached     files whose embedding came from the disk cache
 *   failed     files that could not be read/encoded; the previous retrievable
 *              version is retained (updates are transactional per file)
 *   total_documents  documents tracked after the call
 *   path       canonical display path that was updated
 *   message    failure diagnostics, NULL on success
 *
 * Release with eer_update_result_free(). */
typedef struct eer_update_result {
    char *message;
    char *path;
    size_t added;
    size_t updated;
    size_t removed;
    size_t unchanged;
    size_t cached;
    size_t failed;
    size_t total_documents;
    int32_t sub_code;   /* 1.2, appended */
} eer_update_result;

/* Result of eer_query(): the ranked hit list.
 *
 *   hit_count  number of entries in `hits` (<= the requested top_k)
 *   hits       library-allocated array of hit_count entries, best match first; NULL
 *              when hit_count is 0
 *   message    failure diagnostics, NULL on success
 *
 * Querying an empty corpus is a recoverable EER_STATUS_OPERATIONAL error (no
 * hits, message explains), as is an empty or blank query text. Release with
 * eer_query_result_free(). */
typedef struct eer_query_result {
    char *message;
    size_t hit_count;
    eer_hit *hits;
    uint32_t truncated; /* 1.2, appended: 1 when the query text was cut to
                           max_seq_len (only when requested via flags bit0) */
    int32_t sub_code;   /* 1.2, appended */
} eer_query_result;

/* Chunking request for directory loads and document ingestion (1.2,
 * report.md §2). Fill with zeros, set struct_size = sizeof(...), then set
 * the fields you need. `chunk_by == 0` (or chunk_size == 0) keeps the
 * historical whole-file behavior: one chunk covering the entire document.
 * Chunk boundaries never split a UTF-8 code point; each hit reports the
 * exact byte range of its passage. */
typedef struct eer_chunk_options {
    size_t struct_size;      /* sizeof(eer_chunk_options); fields below are
                                read only up to this size */
    uint32_t chunk_size;     /* target maximum chunk payload in bytes; 0 = off */
    uint32_t chunk_overlap;  /* bytes repeated from the previous chunk's tail */
    uint32_t chunk_by;       /* 0 = off (whole doc), 1 = paragraph (blank-line),
                                2 = sentence (UTF-8 aware), 3 = token (word
                                boundaries, the bge-m3 Metaspace segmentation) */
    uint32_t max_seq_len;    /* 0 = engine default (512); per-chunk encode
                                truncation length (engine mode) */
} eer_chunk_options;

/* Result of eer_add_document()/eer_update_document()/eer_remove_document()
 * (1.2, report.md §2). Chunk counters describe what the call did:
 * re-ingesting a document with identical text/metadata/chunking is reported
 * as chunks_unchanged (idempotent); a replace counts the old chunks in
 * chunks_removed and the new ones in chunks_added. `total_documents` counts
 * chunk records afterwards, `total_docs` the distinct doc_ids. Release with
 * eer_document_result_free(). */
typedef struct eer_document_result {
    char *message;
    char *doc_id;
    size_t chunks_added;
    size_t chunks_removed;
    size_t chunks_unchanged;
    size_t total_documents;
    size_t total_docs;
    int32_t sub_code;   /* EER_SUB_CANCELLED: nothing was applied */
} eer_document_result;

/* Progress + cancellation hook for long ingestion (1.2, report.md §5).
 * Zero-initialize and set struct_size. All fields NULL/0 selects the
 * historical behavior (no progress, no cancel). `on_progress` is invoked on
 * the calling thread with (done, total) after every completed unit; total
 * is the chunk count for document ingestion and the file count for
 * eer_load_dir_ex. `cancel_flag` (owned by the caller, read between units)
 * aborts the step at a consistent boundary: already-ingested content stays,
 * the in-flight document is applied all-or-nothing, the result carries
 * EER_SUB_CANCELLED and a "cancelled" message, and the return status is
 * EER_STATUS_OPERATIONAL (a clean, recoverable partial state). */
typedef struct eer_progress {
    size_t struct_size;
    void (*on_progress)(void *user_data, uint64_t done, uint64_t total);
    void *user_data;
    volatile uint32_t *cancel_flag;
} eer_progress;

/* One text for eer_encode_batch() (1.2): (pointer, byte length), NUL bytes
 * allowed, no NUL-termination required. */
typedef struct eer_text {
    const char *utf8;
    size_t length;
} eer_text;

/* Production statistics snapshot (1.2, report.md §6), filled by eer_stats().
 * `document_count` counts distinct doc_ids, `chunk_count` chunk records;
 * `hybrid` is 1 when the index fuses dense + sparse scores; the cache ratio
 * and mean encode latency cover the index lifetime. No pointers: releasing
 * is a no-op beyond zeroing (eer_stats_data_free() exists for ABI symmetry). */
typedef struct eer_stats_data {
    size_t struct_size;     /* sizeof(eer_stats_data) as passed in */
    uint64_t document_count;
    uint64_t chunk_count;
    uint32_t dimension;
    uint32_t hybrid;
    double cache_hit_ratio;
    double mean_encode_ms;
    uint64_t encode_count;
    uint64_t cache_hits;
    uint64_t cache_misses;
} eer_stats_data;

/* ==========================================================================
 * 5. Host-encoder callback (optional, GPU-free mode)
 * ========================================================================== */

/* Return codes for eer_encode_fn. */
typedef enum eer_encode_status {
    /* `embedding_dimension` floats were written to `embedding`. */
    EER_ENCODE_OK = 0,
    /* `embedding_capacity` was too small: the required size was written to
     * `embedding_dimension` and the library retries once with exactly that
     * many floats. */
    EER_ENCODE_BUFFER_TOO_SMALL = 1,
    /* The text cannot be encoded (any other value is treated this way). The
     * library reports the enclosing step as EER_STATUS_OPERATIONAL and counts
     * the file as failed. */
    EER_ENCODE_FAILED = 2
} eer_encode_status;

/* Optional user-supplied dense encoder. Supplying one switches the context to
 * GPU-free "callback mode": no model, no tokenizer, no backend device and no
 * LuisaCompute plugin are needed, and every embedding comes from this
 * function. This is the intended way to host the retriever in a process that
 * already computes embeddings elsewhere, and it is how the API is unit-tested
 * without a GPU.
 *
 * Contract
 *   - `utf8_text` is NOT NUL-terminated; use `text_length` bytes.
 *   - Write up to `embedding_capacity` floats into `embedding` and store the
 *     number written in `*embedding_dimension`; return EER_ENCODE_OK.
 *   - If the buffer is too small, store the required size in
 *     `*embedding_dimension` and return EER_ENCODE_BUFFER_TOO_SMALL. The
 *     library calls again with that capacity. Requesting more than 16384
 *     floats is rejected.
 *   - Every value must be finite and every text must produce the same
 *     dimension within one index; otherwise the document counts as failed.
 *   - The callback runs on the thread that called the step function, inside
 *     the index lock, so it must not call back into the same context.
 *   - It must not throw or longjmp; report failures through the return code. */
typedef int32_t (*eer_encode_fn)(void *user_data,
                                 const char *utf8_text,
                                 size_t text_length,
                                 float *embedding,
                                 size_t embedding_capacity,
                                 size_t *embedding_dimension);

/* Optional user-supplied SPARSE lexical encoder (1.2, report.md §5): the
 * callback-mode counterpart of the BGE-M3 sparse head. When BOTH `encode`
 * and `sparse_encode` are set in eer_create_info, the context builds a HYBRID
 * index over the two host callbacks: scores fuse dense and sparse halves
 * exactly like engine mode (default 0.7 / 0.3, per-query overridable through
 * eer_query_ex). Setting only `encode` keeps the historical dense-only
 * callback mode; `sparse_encode` without `encode` is ignored.
 *
 * Contract
 *   - `utf8_text` is NOT NUL-terminated; use `text_length` bytes.
 *   - Write up to `pair_capacity` BYTES of packed (uint32 token_id,
 *     float weight) records (8 bytes each, host byte order) into `pairs` and
 *     store the record count in `*pair_count`; return EER_ENCODE_OK.
 *   - If the buffer is too small, store the required record count in
 *     `*pair_count` and return EER_ENCODE_BUFFER_TOO_SMALL; the library
 *     retries once with pair_capacity == pair_count * 8.
 *   - Every weight must be finite and >= 0; the index canonicalizes
 *     (sorts, deduplicates, L2-normalizes) the pairs itself.
 *   - Same threading rules as eer_encode_fn: called on the step thread inside
 *     the index lock; must not call back into the same context, throw or
 *     longjmp. */
typedef int32_t (*eer_sparse_encode_fn)(void *user_data,
                                        const char *utf8_text,
                                        size_t text_length,
                                        uint8_t *pairs,
                                        size_t pair_capacity,
                                        size_t *pair_count);

/* ==========================================================================
 * 6. Lifecycle
 * ========================================================================== */

/* Bit flags for eer_create_info::flags. Zero selects the defaults, i.e. the
 * on-disk embedding cache is enabled and a warm-up encode runs. */
enum {
    /* Do not persist/consult .lc_embed_cache*; every admitted file is encoded
     * on every load/update. */
    EER_CREATE_DISABLE_DISK_CACHE = 1u << 0,
    /* Skip the single warm-up encode that fully initializes and drains the GPU
     * execution path before the first request. Skipping it makes creation
     * cheaper but moves that cost (and any teardown-order surprise) into the
     * first real request. Ignored in callback mode. */
    EER_CREATE_SKIP_WARMUP = 1u << 1,
    /* Opt-in ColBERT late-interaction storage (1.2, report.md §5): the index
     * keeps the per-chunk ColBERT token matrices so eer_query_ex can re-rank
     * by MaxSim and eer_encode_colbert can serve them. Off by default to
     * bound RAM (one H=1024 float32 row per real token per chunk). Engine
     * mode only; ignored in callback mode. */
    EER_CREATE_WANT_COLBERT = 1u << 2
};

/* Initialization request for eer_context_create().
 *
 * Fill the struct with zeros, set `struct_size = sizeof(eer_create_info)`, then
 * set the fields you need. All strings are UTF-8 and are copied; they may be
 * released as soon as eer_context_create() returns.
 *
 * Engine mode (encode == NULL) requires model_path AND tokenizer_path.
 * Callback mode (encode != NULL) ignores model_path, tokenizer_path, backend
 * and program_path, and never touches a GPU. */
typedef struct eer_create_info {
    /* Size of the struct the caller compiled against, i.e.
     * sizeof(eer_create_info). Required so that future versions can grow the
     * struct without breaking old binaries; 0 is accepted and means "current
     * size". A value smaller than this version's size is rejected. */
    size_t struct_size;

    /* Bitwise OR of EER_CREATE_*; 0 selects the defaults. */
    uint32_t flags;

    /* Optional. Directory that contains both this library and the backend
     * modules (luisa-vk.dll / luisa-dx.dll / luisa-cuda.dll / ...). Pass the
     * host process' executable path or its directory when the modules sit next
     * to it; NULL or "" resolves to the current executable, which is what a
     * plain executable would pass as argv[0]. */
    const char *program_path;

    /* Backend plugin name, e.g. "vk" (default when NULL or ""), "dx", "cuda",
     * "fallback", "simd", "metal". The list of installed names is reported in
     * the error message when the requested one cannot be created. */
    const char *backend;

    /* Packed bge-m3 model (model.bgm), required in engine mode. */
    const char *model_path;

    /* tokenizer.json for the same model, required in engine mode. */
    const char *tokenizer_path;

    /* Scheduler worker count (continuous-batching GPU workers). 0 selects the
     * default: the hardware concurrency (number of CPU cores/threads the OS
     * reports, at least 1). */
    uint32_t num_workers;

    /* Optional host encoder; see eer_encode_fn. Non-NULL selects callback
     * mode. */
    eer_encode_fn encode;

    /* Opaque pointer handed back to `encode` on every call. */
    void *encode_user_data;

    /* 1.2 (appended, read only when struct_size covers them): optional sparse
     * lexical encoder, see eer_sparse_encode_fn. When both `encode` and
     * `sparse_encode` are set, callback mode becomes hybrid-capable. */
    eer_sparse_encode_fn sparse_encode;

    /* Opaque pointer handed back to `sparse_encode` on every call. */
    void *sparse_encode_user_data;
} eer_create_info;

/* Initializes a retriever.
 *
 * Both out pointers are implemented defensively for FFI callers:
 *   - `out_context` receives the new handle, or NULL on failure. Required.
 *   - `out_error` is optional; when non-NULL it receives a library-allocated diagnostic
 *     message on failure (release with eer_error_free()) and a NULL message on
 *     success.
 *
 * Creation validates every path up front and returns EER_STATUS_NOT_FOUND /
 * EER_STATUS_INVALID_ARGUMENT before touching the GPU, so callers can report
 * configuration mistakes without risking a device loss. Remaining failures
 * (device creation, session construction) surface as EER_STATUS_OPERATIONAL;
 * a *corrupt* model still terminates the process inside the engine's own
 * invariant checks (LUISA_ERROR), exactly like the executable does.
 *
 * Creation is expensive in engine mode (weight upload + kernel compilation,
 * plus the warm-up encode unless skipped). Create one context per process and
 * keep it; create/destroy per request is not supported. */
EER_API eer_status eer_context_create(const eer_create_info *info,
                                      eer_context **out_context,
                                      eer_error *out_error);

/* Releases every resource owned by the context: the document index, the
 * session (which drains and joins its scheduler workers), the device and the
 * backend context. Passing NULL is a no-op. After the call the handle is
 * dangling: no other call on it is valid. Not thread-safe. */
EER_API void eer_context_destroy(eer_context *context);

/* Name of the backend the context runs on ("vk", "dx", ...), or NULL in
 * callback mode (no device) and for an invalid handle. The returned string
 * lives as long as the context. */
EER_API const char *eer_context_backend(const eer_context *context);

/* 1 when the on-disk embedding cache is active for this context, 0 otherwise
 * (disabled by flag, callback mode with no cache directory, or an invalid
 * handle). */
EER_API uint32_t eer_context_disk_cache_enabled(const eer_context *context);

/* ==========================================================================
 * 7. Retrieval steps -- the C-FFI counterpart of the console commands
 * ========================================================================== */

/* `load <directory> [<extension-list>]` -- recursively indexes a directory.
 *
 *   root        UTF-8 path of the directory to index. Required.
 *   extensions  Optional comma-separated extension rule, e.g. ".txt,.md" or
 *               "TXT,.Md". Case-insensitive, normalized to a leading dot and
 *               lowercase; empty entries, separators, wildcards and malformed
 *               rules are rejected with EER_STATUS_INVALID_ARGUMENT. NULL or ""
 *               selects EER_DEFAULT_EXTENSIONS.
 *   result      Caller-allocated payload; zeroed and filled by the library.
 *               Release with eer_load_result_free() in all cases.
 *
 * Unchanged files of a previously loaded root are not re-encoded: their
 * embeddings are served from the on-disk cache when it is enabled. A root owns
 * its rule and cannot be loaded twice or nested inside another loaded root;
 * unload it first to change the rule.
 *
 * A root is canonicalized before use, so a path that does not exist, cannot be
 * resolved, or is not a directory is reported as
 * EER_STATUS_INVALID_ARGUMENT (not EER_STATUS_NOT_FOUND) by the engine. */
EER_API eer_status eer_load_dir(eer_context *context,
                                const char *root,
                                const char *extensions,
                                eer_load_result *result);

/* `load <directory> [<extension-list>]` with chunking, progress and
 * cancellation (1.2, report.md §2/§5). Identical to eer_load_dir() plus:
 *
 *   chunk_options  Optional; NULL or chunk_by == 0 keeps whole-file behavior.
 *                  Files are then split into passages before encoding, every
 *                  hit carries its byte range, and re-ingestion keys on
 *                  (content hash, chunk_index, chunk params) in the disk
 *                  cache (a second, chunk-level cache layer; the per-file
 *                  .lc_embed_cache* layout is untouched).
 *   progress       Optional; NULL = no progress, no cancel. When the cancel
 *                  flag is set between files, the load stops cleanly: files
 *                  already indexed stay, the result counts the work done so
 *                  far, sub_code is EER_SUB_CANCELLED, message is
 *                  "cancelled" and the return status is
 *                  EER_STATUS_OPERATIONAL (recoverable; re-load without the
 *                  flag to continue).
 *
 * `result` is caller-allocated; release with eer_load_result_free(). */
EER_API eer_status eer_load_dir_ex(eer_context *context,
                                   const char *root,
                                   const char *extensions,
                                   const eer_chunk_options *chunk_options,
                                   eer_load_result *result,
                                   const eer_progress *progress);

/* `unload <loaded-root-directory>` -- drops exactly one loaded root and every
 * document it owns. Descendant paths are not accepted as aliases for their
 * root. `result` is caller-allocated; release with eer_unload_result_free(). */
EER_API eer_status eer_unload_dir(eer_context *context,
                                  const char *root,
                                  eer_unload_result *result);

/* `update <file-or-subdirectory>` -- re-indexes one path in place using the
 * extension rule of the root that owns it. Accepts the loaded root itself, one
 * descendant subdirectory, or one descendant file. Byte-identical files are
 * reported as `unchanged`, modified files are re-encoded, new files are added
 * and deleted tracked files are removed. Updates are transactional per file: a
 * read/encode failure keeps the previous retrievable version and increments
 * `failed`. A path outside every loaded root is EER_STATUS_NOT_FOUND.
 * `result` is caller-allocated; release with eer_update_result_free(). */
EER_API eer_status eer_update_path(eer_context *context,
                                   const char *path,
                                   eer_update_result *result);

/* `query [--top <N>] <query text>` -- ranks indexed documents against a query.
 *
 *   utf8_text    Query text; `text_length` bytes are read (NUL bytes inside the
 *                text are allowed, the string need not be NUL-terminated).
 *   text_length  Byte length of the query. 0 is accepted for an empty input,
 *                which the engine reports as EER_STATUS_INVALID_ARGUMENT;
 *                empty/blank text never matches everything.
 *   top_k        Number of hits to return; 0 selects EER_DEFAULT_TOP_K, values
 *                above EER_MAX_TOP_K are rejected.
 *   result       Caller-allocated payload; release with
 *                eer_query_result_free() in all cases.
 *
 * The query text is encoded with the same encoder the documents used (session
 * or host callback) and scored by dot product of the L2-normalized vectors,
 * i.e. cosine similarity. */
EER_API eer_status eer_query(eer_context *context,
                             const char *utf8_text,
                             size_t text_length,
                             uint32_t top_k,
                             eer_query_result *result);

/* Per-query ranking controls (1.2, report.md §4). Zero-initialize, set
 * struct_size, then set the fields you need; NULL selects all defaults.
 *   dense_weight / sparse_weight  NaN or negative selects the engine defaults
 *                                 (0.7 / 0.3); applied as
 *                                 score = w_d * dense + w_s * sparse.
 *   filter_json                   Optional Mongo-style metadata predicate
 *                                 applied BEFORE top-k selection (never
 *                                 under-fills pages):
 *                                   {"tenant": "acme", "year": {"$gte": 2023}}
 *                                 Operators: equality (string/number/bool),
 *                                 $eq/$gte/$lte/$gt/$lt, $in (array),
 *                                 $and/$or (array of objects). Top-level keys
 *                                 combine implicitly with AND. Only
 *                                 documents ingested with metadata_json can
 *                                 match.
 *   max_seq_len                   0 = engine default (512). Larger values
 *                                 use bge-m3's long-context headroom.
 *   diversity                     0..1 MMR re-ranking over the fused scores;
 *                                 0 (default) disables it. Higher values
 *                                 suppress same-document chunk repetition.
 *   flags                         bit0: report query truncation in
 *                                 eer_query_result::truncated.
 * instruction_prefix Prepended verbatim to the query text before
 * tokenization (engine mode). NULL/empty
 * keeps the query raw: 1.2 does NOT
 * auto-prepend the bge-m3 instruction, so
 * 1.1 query encodings stay bit-identical.
 * For retrieval quality pass
 * "Represent this sentence for searching
 * relevant passages:" explicitly for short
 * queries.
 *   want_colbert                  1 = re-rank the dense/sparse shortlist by
 *                                 ColBERT MaxSim (token-level late
 *                                 interaction). Requires engine mode and the
 * EER_CREATE_WANT_COLBERT create flag; the
 * MaxSim value becomes the hit score and is
 * reported in colbert_score. A record without
 * a stored matrix (restored from a snapshot
 * that predates ColBERT serialization) keeps
 * its fused score for that entry instead of
 * failing the query. Last field: read only
 * when struct_size covers it. */
  typedef struct eer_query_options {
    size_t struct_size;
    double dense_weight;
    double sparse_weight;
    const char *filter_json;
    uint32_t max_seq_len;
    double diversity;
    uint32_t flags;
    const char *instruction_prefix;
    uint32_t want_colbert;
} eer_query_options;

/* `query` with ranking controls (1.2, report.md §4). Identical text/top_k
 * contract to eer_query(); `options` NULL reproduces eer_query() exactly.
 * `result` is caller-allocated; release with eer_query_result_free(). */
EER_API eer_status eer_query_ex(eer_context *context,
                                const char *utf8_text,
                                size_t text_length,
                                uint32_t top_k,
                                const eer_query_options *options,
                                eer_query_result *result);

/* --- In-memory documents (1.2, report.md §2) --------------------------------
 *
 * Index content that never exists as a file: database rows, chat history,
 * web pages. `doc_id` is caller-chosen (non-empty UTF-8 without NUL bytes)
 * and is the dedup/idempotency key: re-ingesting the same doc_id replaces
 * its chunks transactionally (old chunks are dropped only after every new
 * chunk encoded), and re-ingesting identical text/metadata/chunking is a
 * no-op reported in chunks_unchanged. `metadata_json` must be empty or a
 * JSON object; it is kept verbatim for eer_query_ex filters and snapshots.
 * `chunk_options` NULL keeps the document as one whole chunk.
 * `progress` is optional (per-chunk on_progress; cancel between chunks keeps
 * the previous version of the document intact). `result` is caller-allocated;
 * release with eer_document_result_free(). All step functions stay safe to
 * call concurrently on one context. */
EER_API eer_status eer_add_document(eer_context *context,
                                    const char *doc_id,
                                    const char *utf8_text,
                                    size_t text_length,
                                    const char *metadata_json,
                                    const eer_chunk_options *chunk_options,
                                    eer_document_result *result,
                                    const eer_progress *progress);

/* Same as eer_add_document() (replace semantics with the same doc_id). */
EER_API eer_status eer_update_document(eer_context *context,
                                       const char *doc_id,
                                       const char *utf8_text,
                                       size_t text_length,
                                       const char *metadata_json,
                                       const eer_chunk_options *chunk_options,
                                       eer_document_result *result,
                                       const eer_progress *progress);

/* Drops every chunk of `doc_id`. Unknown ids are EER_STATUS_NOT_FOUND. */
EER_API eer_status eer_remove_document(eer_context *context,
                                       const char *doc_id,
                                       eer_document_result *result);

/* ==========================================================================
 * 8. Encoding and introspection
 * ========================================================================== */

/* Encodes one text into its dense embedding, without touching the index.
 *
 *   utf8_text         Text bytes; `text_length` bytes are read. NULL is allowed
 *                     only when `text_length` is 0.
 *   text_length       Byte length (NUL-termination not required).
 *   embedding         Caller buffer of at least `embedding_capacity` floats.
 *   embedding_capacity Capacity in floats. Pass 0 with `embedding == NULL` to
 *                     ask for the dimension only: the function then returns
 *                     EER_STATUS_OK with `*embedding_dimension` set and writes
 *                     nothing.
 *   embedding_dimension Out: number of floats written (or required). Required.
 *   out_error         Optional; receives the library-allocated diagnostic message on
 *                     failure (release with eer_error_free()).
 *
 * Use it to check the embedding dimension, to pre-compute query vectors, or to
 * verify that a host encoder behaves as expected in callback mode. */
EER_API eer_status eer_encode_text(eer_context *context,
                                   const char *utf8_text,
                                   size_t text_length,
                                   float *embedding,
                                   size_t embedding_capacity,
                                   size_t *embedding_dimension,
                                   eer_error *out_error);

/* Encodes many texts in one round trip (1.2, report.md §5): engine mode
 * submits the batch through the session's continuous-batching scheduler;
 * callback mode invokes the host encoder per text. Output layout is
 * row-major [count][dimension] in `out_vectors`; every text must produce the
 * same dimension (validated; a mismatch is EER_STATUS_OPERATIONAL with
 * EER_SUB_ENCODE).
 *
 *   texts           Array of `count` (pointer, length) entries; NULL entries
 *                   with length 0 are encoded as empty strings.
 *   count           Number of entries (0 is rejected).
 *   out_vectors     Caller buffer of `capacity` floats, or NULL with
 *                   capacity 0 for a dimension probe: the texts are still
 *                   encoded and validated, *out_dimension receives the
 *                   uniform dimension and nothing is written.
 *   capacity        Capacity of out_vectors in floats; must be >=
 *                   count * dimension when writing.
 *   out_dimension   Required; receives the uniform dimension.
 *
 * Engine mode honors the same max_seq_len=512 default as eer_encode_text. */
EER_API eer_status eer_encode_batch(eer_context *context,
                                    const eer_text *texts,
                                    size_t count,
                                    float *out_vectors,
                                    size_t capacity,
                                    size_t *out_dimension,
                                    eer_error *out_error);

/* Encodes one text into its ColBERT late-interaction matrix (1.2,
 * report.md §5), without touching the index. Engine mode only (the host
 * callback has no ColBERT head): EER_STATUS_OPERATIONAL with EER_SUB_ENCODE
 * in callback mode. The matrix is `out_rows` rows of `dimension` float32
 * (row-major), per-row L2-normalized; `*out_values` receives rows *
 * dimension. Pass `out == NULL` with `capacity == 0` to probe rows/dimension
 * only. */
EER_API eer_status eer_encode_colbert(eer_context *context,
                                      const char *utf8_text,
                                      size_t text_length,
                                      float *out,
                                      size_t capacity,
                                      size_t *out_values,
                                      uint32_t *out_rows,
                                      eer_error *out_error);

/* --- Corpus snapshots (1.2, report.md §3) -----------------------------------
 *
 * O(1) cold start for an already-ingested corpus: eer_index_save() writes a
 * JSON manifest at `path` plus a packed-float32 blob at `path` + ".bin"
 * (dense vectors in manifest order, sparse pairs and document/chunk metadata
 * inline; memory-only documents carry their text). When the corpus stores
 * ColBERT matrices, a second blob `path` + ".colbert.bin" holds them with the
 * per-record row counts in the manifest; eer_index_load() restores them only
 * into a context created with EER_CREATE_WANT_COLBERT (the create flag's RAM
 * bound applies to snapshots too). eer_index_load() restores roots, documents,
 * chunks, metadata and embeddings with NO re-encode. The pair is relocatable
 * as a unit; loading on a different backend (dx vs vk) is supported (the
 * blob is host byte order). A version/dimension/hybrid mismatch, a blob whose
 * size does not match the manifest (checked with overflow-safe arithmetic, so
 * a corrupt manifest cannot wrap the comparison), or an unreadable file is a
 * clean EER_STATUS_OPERATIONAL error that leaves the current index untouched.
 * The per-file .lc_embed_cache* layer is independent and untouched. */
EER_API eer_status eer_index_save(eer_context *context,
                                  const char *path,
                                  eer_error *out_error);
EER_API eer_status eer_index_load(eer_context *context,
                                  const char *path,
                                  eer_error *out_error);

/* --- Serving surface (1.2, report.md §6) ------------------------------------
 *
 * Production observability over the same context; all safe to call
 * concurrently with the steps. */

/* Counters and mode flags (see eer_stats_data). struct_size must be set;
 * never allocates (eer_stats_data_free() zeroes the struct for symmetry). */
EER_API eer_status eer_stats(eer_context *context, eer_stats_data *out_stats);
EER_API void eer_stats_data_free(eer_stats_data *stats);

/* Readiness probe: one tiny encode through the full pipeline (session or
 * host callback), so a load balancer can mark the worker warm without a
 * full query. EER_STATUS_OPERATIONAL with EER_SUB_ENCODE/EER_SUB_DEVICE
 * when the pipeline is broken. */
EER_API eer_status eer_context_ping(eer_context *context, eer_error *out_error);

/* Device memory pressure: *out_used_bytes / *out_total_bytes for the
 * context's GPU device where the backend reports them. LuisaCompute does not
 * expose a device memory-stats API, so every context currently reports the
 * honest 0/0 with EER_STATUS_OK (callback mode included); the entry point
 * exists so hosts can code against it unchanged when a backend starts
 * reporting. */
EER_API eer_status eer_context_device_memory(eer_context *context,
                                             uint64_t *out_used_bytes,
                                             uint64_t *out_total_bytes);

/* Park/resume: release and re-create device buffers/session around idle
 * periods while the index stays in host RAM (report.md §6). NOT SUPPORTED in
 * this build: the Session lifetime is tied to the device context member
 * order documented on eer_context, and a clean unload/reload proved too
 * invasive. Both entry points exist so hosts can code against them; they
 * return EER_STATUS_OPERATIONAL with message "park/resume is not supported
 * in this build" and EER_SUB_NONE. */
EER_API eer_status eer_context_park(eer_context *context, eer_error *out_error);
EER_API eer_status eer_context_resume(eer_context *context, eer_error *out_error);

/* ==========================================================================
 * 9. Memory management
 * ========================================================================== */

/* Frees the strings allocated inside the payloads above and zeroes the struct,
 * so a second call is a no-op. Passing NULL is a no-op. Every result payload
 * must be passed here exactly once, whether the step succeeded or failed. */
EER_API void eer_error_free(eer_error *error);
EER_API void eer_load_result_free(eer_load_result *result);
EER_API void eer_unload_result_free(eer_unload_result *result);
EER_API void eer_update_result_free(eer_update_result *result);
EER_API void eer_query_result_free(eer_query_result *result);
EER_API void eer_document_result_free(eer_document_result *result);

#ifdef __cplusplus
}// extern "C"
#endif
