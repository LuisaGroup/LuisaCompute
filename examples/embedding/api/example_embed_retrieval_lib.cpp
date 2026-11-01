// C ABI implementation for the bge-m3 embedding retrieval engine
// (example_embed_retrieval_lib.h; retrieval plan phases 2-4; RAG growth
// report.md §2-§6, ABI 1.2). This translation unit is the only place where
// the C boundary meets LuisaCompute / engine types: every entry point
// validates its arguments, converts UTF-8 into the engine's path/string
// types, runs one engine step, and copies the result back into caller-owned
// C structs. The steps mirror example_embed_retrieval.cpp
// (load/unload/query/update) plus the RAG surface: chunked documents with
// passage byte ranges, corpus snapshots, per-query ranking controls, batch
// and ColBERT encoding, progress/cancel, stats/ping and hybrid callback mode.
// Engine mode scores hybrid (0.7 * dense dot + 0.3 * sparse cosine by
// default, per-query overridable); callback mode is dense-only unless a
// sparse host encoder is supplied (hybrid callback mode).

#include "example_embed_retrieval_lib.h"

#include <cstdlib>
#include <cstring>
#include <system_error>
#include <thread>

#include <luisa/core/fiber.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>

#include "embedding/embed/retrieval_disk_cache.h"
#include "embedding/embed/retrieval_service.h"
#include "embedding/embed/session.h"

// One initialized retriever. Member order is the destruction order the engine
// requires: the service (and its disk-cache hook) must die before the session,
// the session before the device, and the device before the backend context
// that created it.
struct eer_context {
    // Best-effort use-after-destroy detection; cleared by eer_context_destroy().
    static constexpr uint64_t live_magic = 0x4545522D43545801ull;// "EER-CTX\1"

    uint64_t magic{live_magic};
    luisa::unique_ptr<luisa::compute::Context> context;
    luisa::unique_ptr<luisa::compute::Device> device;
    // Engine mode binds a fiber scheduler for Session's worker pool. Declared
    // before session so the session is destroyed first (reverse member order).
    luisa::unique_ptr<luisa::fiber::scheduler> fiber_scheduler;
    luisa::unique_ptr<luisa::embed::Session> session;
    luisa::unique_ptr<luisa::embed::DiskEmbedCache> disk_cache;
    luisa::unique_ptr<luisa::embed::RetrievalService> service;
    eer_encode_fn encode_fn{nullptr};
    void *encode_user_data{nullptr};
    // 1.2: optional sparse lexical host encoder (hybrid callback mode).
    eer_sparse_encode_fn sparse_encode_fn{nullptr};
    void *sparse_encode_user_data{nullptr};
    bool want_colbert{false};// EER_CREATE_WANT_COLBERT
    luisa::string backend_name;// empty in callback mode
    bool cache_enabled{false};
};

// The entry points below are plain C functions; inside this translation unit
// they use the engine's own vocabulary.
using namespace luisa;
using namespace luisa::compute;

namespace {

// --- tunables ---------------------------------------------------------------

// First buffer size offered to a host encoder; the callback may ask for more.
constexpr size_t kEncoderInitialCapacity = 1024u;
// Hard cap on the dense dimension a host encoder may request (sanity bound).
constexpr size_t kEncoderMaxCapacity = 16384u;
// First buffer size (in pairs) offered to a sparse host encoder.
constexpr size_t kSparseInitialPairCapacity = 512u;
// Hard cap on the sparse pair count a host encoder may request.
constexpr size_t kSparseMaxPairCapacity = 65536u;
// Struct field availability floors for the 1.2 options structs (a caller may
// pass a struct from an EARLIER 1.2.x header that ends before our last field).
constexpr size_t kChunkOptionsBaseSize = offsetof(eer_chunk_options, max_seq_len);
// want_colbert is the last field of eer_query_options and is read only when
// struct_size covers it, so a caller whose struct ends before it (an earlier
// 1.2 build, or a caller that stopped at the base layout) is never read past.
constexpr size_t kQueryOptionsBaseSize = offsetof(eer_query_options, want_colbert);
constexpr size_t kQueryOptionsColbertSize = kQueryOptionsBaseSize + sizeof(uint32_t);

// Scheduler worker count used when the caller passes 0: the hardware
// concurrency (CPU core/thread count), falling back to 1 when the OS cannot
// report it.
[[nodiscard]] uint32_t default_worker_count() noexcept {
    const auto hardware = std::thread::hardware_concurrency();
    return hardware == 0u ? 1u : hardware;
}
// Backend used when the caller does not name one (same default as the console).
constexpr char kDefaultBackend[] = "vk";
// The executable performs exactly this warm-up encode; keep the two in sync.
constexpr char kWarmupText[] = "retrieval session warmup";
// Tiny encode for eer_context_ping().
constexpr char kPingText[] = "ping";

#define EER_STRINGIFY_INNER(x) #x
#define EER_STRINGIFY(x) EER_STRINGIFY_INNER(x)
constexpr char kVersionString[] =
    EER_STRINGIFY(EER_VERSION_MAJOR) "." EER_STRINGIFY(EER_VERSION_MINOR) "." EER_STRINGIFY(EER_VERSION_PATCH);

// --- small helpers ----------------------------------------------------------

// Allocates a NUL-terminated copy of `text` with the engine allocator (never
// the bare CRT malloc, so the memory can safely cross the shared-library
// boundary and is only ever released through the matching eer_*_free entry
// point); nullptr means out of memory, which callers report as a failed (but
// non-fatal) result field.
[[nodiscard]] char *allocate_string(string_view text) noexcept {
    auto *buffer = allocate_with_allocator<char>(text.size() + 1u);
    if (buffer == nullptr) { return nullptr; }
    if (!text.empty()) { std::memcpy(buffer, text.data(), text.size()); }
    buffer[text.size()] = '\0';
    return buffer;
}

[[nodiscard]] string_view text_or_empty(const char *text) noexcept {
    return text == nullptr ? string_view{} : string_view{text};
}

[[nodiscard]] string path_text(const luisa::filesystem::path &path) {
    const auto utf8 = path.generic_u8string();
    return string{reinterpret_cast<const char *>(utf8.data()), utf8.size()};
}

[[nodiscard]] bool context_is_live(const eer_context *context) noexcept {
    return context != nullptr && context->magic == eer_context::live_magic;
}

[[nodiscard]] eer_status from_error_code(embed::RetrievalErrorCode code) noexcept {
    switch (code) {
        case embed::RetrievalErrorCode::none: return EER_STATUS_OK;
        case embed::RetrievalErrorCode::invalid_argument: return EER_STATUS_INVALID_ARGUMENT;
        case embed::RetrievalErrorCode::not_found: return EER_STATUS_NOT_FOUND;
        case embed::RetrievalErrorCode::operational: break;
    }
    return EER_STATUS_OPERATIONAL;
}

// Engine sub-codes map 1:1 onto the C enum (same numeric values).
[[nodiscard]] int32_t to_c_sub(embed::RetrievalSubCode code) noexcept {
    return static_cast<int32_t>(code);
}

// Fills a result's message slot and returns the status, so a step can write
// `return fail(result->message, code, "...")`. When the caller passes
// `out_sub` (every payload that carries an appended sub_code field), the
// sub-code is written through it, so a failure NEVER leaves sub_code
// unpopulated: argument/handle errors report EER_SUB_NONE explicitly, and
// subsystem failures pass their code. Payloads are zero-initialized at step
// entry, so the discipline is defensive, not load-bearing.
[[nodiscard]] eer_status fail(char *&message, eer_status code, string_view text,
                              int32_t *out_sub = nullptr,
                              int32_t sub_code = EER_SUB_NONE) noexcept {
    message = allocate_string(text);
    if (out_sub != nullptr) { *out_sub = sub_code; }
    return code;
}

// Failure helper for steps whose payload carries a trailing sub_code: fills
// message + sub_code and returns the mapped status.
template<typename Result>
[[nodiscard]] eer_status fail_result(Result *result, eer_status code,
                                     string_view text, embed::RetrievalSubCode sub) noexcept {
    result->sub_code = to_c_sub(sub);
    return fail(result->message, code, text);
}

[[nodiscard]] string join_extensions(span<const string> extensions) {
    string result;
    for (size_t i = 0u; i < extensions.size(); i++) {
        if (i != 0u) { result.push_back(','); }
        result += extensions[i];
    }
    return result;
}

// Converts a UTF-8 path argument with the engine's own converter (so Windows
// never round-trips through the ANSI code page) and requires a regular file.
[[nodiscard]] eer_status resolve_regular_file(
    string_view utf8,
    luisa::filesystem::path &path,
    string &message) {
    auto status = embed::RetrievalIndex::path_from_utf8(utf8, path);
    if (!status.ok()) {
        message = std::move(status.message);
        return from_error_code(status.code);
    }
    std::error_code ec;
    if (!luisa::filesystem::is_regular_file(path, ec) || ec) {
        message = luisa::format("'{}' does not exist or is not a regular file", path_text(path));
        return EER_STATUS_NOT_FOUND;
    }
    return EER_STATUS_OK;
}

// --- encoder plumbing -------------------------------------------------------

// Runs the caller's host encoder, retrying once when it asks for a bigger
// buffer. Returns an empty vector on any failure; the index turns that into a
// documented "encoder returned an empty dense embedding" operational error.
[[nodiscard]] luisa::vector<float> encode_with_host_callback(
    eer_context *context,
    string_view text) noexcept {
    luisa::vector<float> embedding;
    if (context->encode_fn == nullptr) { return embedding; }
    auto capacity = kEncoderInitialCapacity;
    for (;;) {
        embedding.resize(capacity);
        size_t dimension = 0u;
        const auto status = context->encode_fn(
            context->encode_user_data,
            text.empty() ? nullptr : text.data(),
            text.size(),
            embedding.data(),
            capacity,
            &dimension);
        if (status == EER_ENCODE_OK) {
            if (dimension == 0u || dimension > capacity) { break; }
            embedding.resize(dimension);
            return embedding;
        }
        // Misbehaving callback (asked for more without reporting a required
        // size, or beyond the sanity bound): fail instead of looping.
        if (status != EER_ENCODE_BUFFER_TOO_SMALL ||
            dimension == 0u || dimension > kEncoderMaxCapacity ||
            dimension <= capacity) {
            break;
        }
        capacity = dimension;
    }
    embedding.clear();
    return embedding;
}

// Runs the caller's sparse host encoder with the same retry discipline;
// pairs are packed (uint32 token_id, float weight) by the host.
[[nodiscard]] luisa::vector<std::pair<uint32_t, float>> sparse_with_host_callback(
    eer_context *context,
    string_view text) noexcept {
    luisa::vector<std::pair<uint32_t, float>> pairs;
    if (context->sparse_encode_fn == nullptr) { return pairs; }
    auto capacity = kSparseInitialPairCapacity;
    for (;;) {
        pairs.resize(capacity);
        size_t count = 0u;
        const auto status = context->sparse_encode_fn(
            context->sparse_encode_user_data,
            text.empty() ? nullptr : text.data(),
            text.size(),
            reinterpret_cast<uint8_t *>(pairs.data()),
            capacity * sizeof(std::pair<uint32_t, float>),
            &count);
        if (status == EER_ENCODE_OK) {
            if (count > capacity) { break; }
            pairs.resize(count);
            return pairs;
        }
        if (status != EER_ENCODE_BUFFER_TOO_SMALL ||
            count == 0u || count > kSparseMaxPairCapacity || count <= capacity) {
            break;
        }
        capacity = count;
    }
    pairs.clear();
    return pairs;
}

// The single dense encoder the public C ABI sees: the host callback in callback
// mode, the model session otherwise. eer_encode_text keeps calling this, so its
// dense-only output contract is unchanged by hybrid retrieval.
[[nodiscard]] luisa::vector<float> encode_text(eer_context *context, string_view text) {
    if (context->encode_fn != nullptr) { return encode_with_host_callback(context, text); }
    auto embedding = context->session->encode(text);
    return std::move(embedding.dense);
}

// Model-backed dense + sparse (+ optional ColBERT) encode (retrieval plan
// phase H2 + report.md §5). Only ever used in engine mode, where a Session
// exists: the sparse head is switched on and both halves are moved into the
// retrieval embedding. The RetrievalEncodeRequest carries the per-call
// max_seq_len (0 = session default) and the opt-in ColBERT head.
[[nodiscard]] embed::RetrievalEmbedding encode_hybrid_text(
    eer_context *context, string_view text, const embed::RetrievalEncodeRequest &request) {
    embed::EncodeOptions options;
    options.want_sparse = true;
    options.want_colbert = request.want_colbert;
    if (request.max_seq_len != 0u) { options.max_seq_len = request.max_seq_len; }
    auto encoded = context->session->encode(text, options);
    embed::RetrievalEmbedding embedding;
    embedding.dense = std::move(encoded.dense);
    embedding.sparse = std::move(encoded.sparse);
    embedding.colbert = std::move(encoded.colbert);
    embedding.truncated = encoded.truncated;
    return embedding;
}

// Dense-only service (callback mode without a sparse encoder: unchanged
// behavior): the host encoder produces one dense vector per text and the
// index scores by plain dense dot product. Wrapping the encoder in
// RetrievalEncodeCallback pins the dense RetrievalService constructor, so an
// empty-sparse host callback can never be mistaken for a hybrid encoder.
[[nodiscard]] luisa::unique_ptr<embed::RetrievalService> make_dense_service(eer_context *context) {
    return luisa::make_unique<embed::RetrievalService>(
        embed::RetrievalEncodeCallback{[context](string_view text,
                                                 const embed::RetrievalEncodeRequest &) {
            return encode_text(context, text);
        }});
}

// Hybrid service (engine mode, phase H2): scores are
// kRetrievalDenseWeight * dense dot + kRetrievalSparseWeight * sparse cosine
// by default, per-query overridable through eer_query_ex.
[[nodiscard]] luisa::unique_ptr<embed::RetrievalService> make_hybrid_service(eer_context *context) {
    return luisa::make_unique<embed::RetrievalService>(
        embed::RetrievalHybridEncodeCallback{[context](
                                                 string_view text,
                                                 const embed::RetrievalEncodeRequest &request) {
            return encode_hybrid_text(context, text, request);
        }});
}

// Hybrid service in CALLBACK mode (1.2, report.md §5): both halves come from
// host callbacks, so a GPU-free host stays hybrid-capable.
[[nodiscard]] luisa::unique_ptr<embed::RetrievalService> make_hybrid_callback_service(eer_context *context) {
    return luisa::make_unique<embed::RetrievalService>(
        embed::RetrievalHybridEncodeCallback{[context](
                                                 string_view text,
                                                 const embed::RetrievalEncodeRequest &) {
            embed::RetrievalEmbedding embedding;
            embedding.dense = encode_with_host_callback(context, text);
            embedding.sparse = sparse_with_host_callback(context, text);
            return embedding;
        }});
}

// --- options conversion -----------------------------------------------------

// Converts the C chunk options; NULL / struct_size below the base layout
// selects the historical whole-file behavior. Note a plain
// `embed::ChunkOptions{}` does NOT: its in-class defaults mean paragraph
// chunking at 512 bytes, so the NULL path must ask for whole_file() explicitly
// (ABI 1.1 compatibility: legacy callers never chunked).
[[nodiscard]] eer_status convert_chunk_options(
    const eer_chunk_options *options,
    embed::ChunkOptions &out,
    string &message) noexcept {
    out = embed::ChunkOptions::whole_file();
    if (options == nullptr) { return EER_STATUS_OK; }
    if (options->struct_size != 0u && options->struct_size < kChunkOptionsBaseSize) {
        message = format("eer_chunk_options::struct_size {} is smaller than the base size {}",
                         options->struct_size, kChunkOptionsBaseSize);
        return EER_STATUS_INVALID_ARGUMENT;
    }
    out.chunk_size = options->chunk_size;
    out.chunk_overlap = options->chunk_overlap;
    out.by = static_cast<embed::ChunkOptions::By>(options->chunk_by);
    if (options->chunk_by > 3u) {
        message = format("chunk_by must be 0..3, got {}", options->chunk_by);
        return EER_STATUS_INVALID_ARGUMENT;
    }
    if (options->struct_size == 0u ||
        options->struct_size >= offsetof(eer_chunk_options, max_seq_len) + sizeof(uint32_t)) {
        out.max_seq_len = options->max_seq_len;
    }
    return EER_STATUS_OK;
}

// Converts the C progress hook; NULL selects no progress/no cancel.
[[nodiscard]] eer_status convert_progress(
    const eer_progress *progress,
    embed::RetrievalProgress &out,
    string &message) noexcept {
    out = embed::RetrievalProgress{};
    if (progress == nullptr) { return EER_STATUS_OK; }
    if (progress->struct_size != 0u &&
        progress->struct_size < offsetof(eer_progress, cancel_flag)) {
        message = format("eer_progress::struct_size {} is smaller than the base size {}",
                         progress->struct_size, offsetof(eer_progress, cancel_flag));
        return EER_STATUS_INVALID_ARGUMENT;
    }
    out.on_progress = progress->on_progress;
    out.user_data = progress->user_data;
    if (progress->struct_size == 0u ||
        progress->struct_size >= offsetof(eer_progress, cancel_flag) + sizeof(uint32_t *)) {
        out.cancel_flag = progress->cancel_flag;
    }
    return EER_STATUS_OK;
}

// Converts the C query options; NULL selects all defaults.
[[nodiscard]] eer_status convert_query_options(
    const eer_query_options *options,
    embed::RetrievalQueryOptions &out,
    string &message) noexcept {
    out = embed::RetrievalQueryOptions{};
    if (options == nullptr) { return EER_STATUS_OK; }
    if (options->struct_size != 0u && options->struct_size < kQueryOptionsBaseSize) {
        message = format("eer_query_options::struct_size {} is smaller than the supported size {}",
                         options->struct_size, kQueryOptionsBaseSize);
        return EER_STATUS_INVALID_ARGUMENT;
    }
    out.dense_weight = options->dense_weight;
    out.sparse_weight = options->sparse_weight;
    out.filter_json = string{text_or_empty(options->filter_json)};
    out.max_seq_len = options->max_seq_len;
    out.diversity = options->diversity;
    out.report_truncation = (options->flags & 1u) != 0u;
    out.instruction_prefix = string{text_or_empty(options->instruction_prefix)};
    if (options->struct_size == 0u || options->struct_size >= kQueryOptionsColbertSize) {
        out.want_colbert = options->want_colbert != 0u;
    }
    return EER_STATUS_OK;
}

// Fills a load result payload from the engine result (shared by the plain and
// the _ex entry points, including the cancelled-partial state).
[[nodiscard]] eer_status fill_load_result(
    const embed::RetrievalLoadResult &loaded,
    eer_load_result *result) noexcept {
    result->root = allocate_string(loaded.root);
    result->extensions = allocate_string(join_extensions(
        span<const string>{loaded.extensions.data(), loaded.extensions.size()}));
    result->added = loaded.added;
    result->cached = loaded.cached;
    result->skipped = loaded.skipped;
    result->failed = loaded.failed;
    result->total_documents = loaded.total_documents;
    if (!loaded.status.ok()) {
        return fail_result(result, from_error_code(loaded.status.code),
                           loaded.status.message, loaded.status.sub_code);
    }
    if (result->root == nullptr || result->extensions == nullptr) {
        eer_load_result_free(result);
        return fail(result->message, EER_STATUS_OPERATIONAL, "out of memory", &result->sub_code);
    }
    return EER_STATUS_OK;
}

// Fills a query result payload from the engine result.
[[nodiscard]] eer_status fill_query_result(
    const embed::RetrievalQueryResult &queried,
    eer_query_result *result) noexcept {
    result->truncated = queried.truncated ? 1u : 0u;
    if (!queried.status.ok()) {
        return fail_result(result, from_error_code(queried.status.code),
                           queried.status.message, queried.status.sub_code);
    }
    if (queried.hits.empty()) { return EER_STATUS_OK; }
    auto *hits = allocate_with_allocator<eer_hit>(queried.hits.size());
    if (hits == nullptr) {
        return fail(result->message, EER_STATUS_OPERATIONAL,
                    "out of memory while building the hit list", &result->sub_code);
    }
    std::memset(hits, 0, queried.hits.size() * sizeof(eer_hit));
    result->hits = hits;
    result->hit_count = queried.hits.size();
    for (size_t i = 0u; i < queried.hits.size(); i++) {
        hits[i].score = queried.hits[i].score;
        hits[i].chunk_index = queried.hits[i].chunk_index;
        hits[i].byte_offset = queried.hits[i].byte_offset;
        hits[i].byte_length = queried.hits[i].byte_length;
        hits[i].dense_score = queried.hits[i].dense_score;
        hits[i].sparse_score = queried.hits[i].sparse_score;
        hits[i].colbert_score = queried.hits[i].colbert_score;
        hits[i].path = queried.hits[i].path.empty() ? nullptr
                                                    : allocate_string(queried.hits[i].path);
        hits[i].doc_id = allocate_string(queried.hits[i].doc_id);
        if (hits[i].doc_id == nullptr || queried.hits[i].path.empty() != (hits[i].path == nullptr)) {
            eer_query_result_free(result);
            return fail(result->message, EER_STATUS_OPERATIONAL,
                        "out of memory while building the hit list", &result->sub_code);
        }
    }
    return EER_STATUS_OK;
}

// Fills a document result payload from the engine result.
[[nodiscard]] eer_status fill_document_result(
    const embed::RetrievalDocumentResult &document,
    eer_document_result *result) noexcept {
    result->doc_id = allocate_string(document.doc_id);
    result->chunks_added = document.chunks_added;
    result->chunks_removed = document.chunks_removed;
    result->chunks_unchanged = document.chunks_unchanged;
    result->total_documents = document.total_documents;
    result->total_docs = document.total_docs;
    if (result->doc_id == nullptr) {
        return fail_result(result, EER_STATUS_OPERATIONAL, "out of memory",
                           embed::RetrievalSubCode::none);
    }
    if (!document.status.ok()) {
        return fail_result(result, from_error_code(document.status.code),
                           document.status.message, document.status.sub_code);
    }
    return EER_STATUS_OK;
}

}// namespace

// --- introspection ----------------------------------------------------------

const char *eer_status_string(eer_status status) {
    switch (status) {
        case EER_STATUS_OK: return "ok";
        case EER_STATUS_INVALID_ARGUMENT: return "invalid_argument";
        case EER_STATUS_NOT_FOUND: return "not_found";
        case EER_STATUS_OPERATIONAL: return "operational";
        case EER_STATUS_INVALID_HANDLE: return "invalid_handle";
    }
    return "unknown";
}

const char *eer_sub_code_string(int32_t sub_code) {
    switch (sub_code) {
        case EER_SUB_NONE: return "none";
        case EER_SUB_DEVICE: return "device";
        case EER_SUB_MODEL: return "model";
        case EER_SUB_IO: return "io";
        case EER_SUB_ENCODE: return "encode";
        case EER_SUB_INDEX: return "index";
        case EER_SUB_CANCELLED: return "cancelled";
    }
    return "unknown";
}

const char *eer_version(void) {
    return kVersionString;
}

const char *eer_default_extensions(void) {
    return EER_DEFAULT_EXTENSIONS;
}

// --- lifecycle --------------------------------------------------------------

eer_status eer_context_create(const eer_create_info *info,
                              eer_context **out_context,
                              eer_error *out_error) {
    if (out_context == nullptr) { return EER_STATUS_INVALID_ARGUMENT; }
    *out_context = nullptr;
    if (out_error != nullptr) { *out_error = eer_error{}; }

    const auto report = [out_error](eer_status code, string_view text,
                                    int32_t sub = EER_SUB_NONE) {
        if (out_error != nullptr) {
            out_error->message = allocate_string(text);
            out_error->sub_code = sub;
        }
        return code;
    };

    if (info == nullptr) {
        return report(EER_STATUS_INVALID_ARGUMENT, "eer_create_info must not be null");
    }
    if (info->struct_size != 0u && info->struct_size < sizeof(eer_create_info)) {
        return report(EER_STATUS_INVALID_ARGUMENT,
                      format("eer_create_info::struct_size {} is smaller than the supported size {}",
                             info->struct_size, sizeof(eer_create_info)));
    }

    auto context = luisa::make_unique<eer_context>();
    context->encode_fn = info->encode;
    context->encode_user_data = info->encode_user_data;
    if (info->struct_size == 0u ||
        info->struct_size >= offsetof(eer_create_info, sparse_encode) + sizeof(eer_sparse_encode_fn)) {
        context->sparse_encode_fn = info->sparse_encode;
        context->sparse_encode_user_data = info->sparse_encode_user_data;
    }
    const auto want_disk_cache = (info->flags & EER_CREATE_DISABLE_DISK_CACHE) == 0u;
    const auto want_warmup = (info->flags & EER_CREATE_SKIP_WARMUP) == 0u;
    context->want_colbert = (info->flags & EER_CREATE_WANT_COLBERT) != 0u;
    const auto worker_count = info->num_workers == 0u ? default_worker_count() : info->num_workers;

    if (context->encode_fn == nullptr) {
        // Engine mode: a GPU backend session over a packed model.
        const auto backend = text_or_empty(info->backend).empty() ? string_view{kDefaultBackend} : text_or_empty(info->backend);
        luisa::filesystem::path model_path;
        luisa::filesystem::path tokenizer_path;
        string message;
        if (auto status = resolve_regular_file(text_or_empty(info->model_path), model_path, message);
            status != EER_STATUS_OK) {
            return report(status, message, EER_SUB_MODEL);
        }
        if (auto status = resolve_regular_file(text_or_empty(info->tokenizer_path), tokenizer_path, message);
            status != EER_STATUS_OK) {
            return report(status, message, EER_SUB_MODEL);
        }

        const auto program_path = text_or_empty(info->program_path);
        context->context = luisa::make_unique<Context>(program_path);
        LUISA_INFO("example_embed_retrieval_lib: selecting backend '{}'", backend);
        auto device = context->context->create_device(backend);
        if (!device) {
            string installed;
            for (const auto &name : context->context->installed_backends()) {
                if (!installed.empty()) { installed += ", "; }
                installed += name;
            }
            return report(EER_STATUS_OPERATIONAL,
                          format("backend '{}' is unavailable; installed backends: {}",
                                 backend, installed.empty() ? string{"<none>"} : installed),
                          EER_SUB_DEVICE);
        }
        context->backend_name = string{backend};
        context->device = luisa::make_unique<Device>(std::move(device));
        context->fiber_scheduler = luisa::make_unique<luisa::fiber::scheduler>();
        context->session = embed::Session::create(
            *context->device, model_path, tokenizer_path, worker_count);
        if (context->session == nullptr) {
            return report(EER_STATUS_OPERATIONAL, "Session::create failed", EER_SUB_DEVICE);
        }
        // Fully initialize and drain the backend execution path once, exactly
        // like the executable does before accepting commands.
        if (want_warmup) {
            auto warmup = context->session->encode(string_view{kWarmupText});
            if (warmup.dense.empty()) {
                return report(EER_STATUS_OPERATIONAL,
                              "session warm-up returned an empty dense embedding",
                              EER_SUB_DEVICE);
            }
        }
    }

    // Encoder selection is structural, not data driven (retrieval plan phase H2):
    // engine mode always builds the hybrid service (dense + BGE-M3 sparse lexical
    // weights); callback mode builds the hybrid service too when a sparse host
    // encoder is supplied (1.2), and the dense-only one otherwise.
    context->service = context->encode_fn == nullptr ? make_hybrid_service(context.get()) :
                         context->sparse_encode_fn != nullptr ? make_hybrid_callback_service(context.get()) :
                                                                make_dense_service(context.get());
    if (context->want_colbert) {
        context->service->set_store_colbert(true);
    }

    if (want_disk_cache) {
        // A null device selects the host-only cache path, which is what
        // callback mode (no device) uses.
        context->disk_cache = luisa::make_unique<embed::DiskEmbedCache>(context->device.get());
        context->service->set_embed_cache(context->disk_cache.get());
        context->cache_enabled = true;
    }

    *out_context = context.release();
    return EER_STATUS_OK;
}

void eer_context_destroy(eer_context *context) {
    if (!context_is_live(context)) { return; }
    context->magic = 0u;
    // Wrapped in a unique_ptr so the member teardown order above is honored by
    // the normal destructor path (and so the engine's own RAII does the work).
    luisa::unique_ptr<eer_context>{context};
}

const char *eer_context_backend(const eer_context *context) {
    if (!context_is_live(context) || context->backend_name.empty()) { return nullptr; }
    return context->backend_name.c_str();
}

uint32_t eer_context_disk_cache_enabled(const eer_context *context) {
    if (!context_is_live(context)) { return 0u; }
    return context->cache_enabled ? 1u : 0u;
}

// --- load / unload / update ---------------------------------------------------

eer_status eer_load_dir(eer_context *context,
                        const char *root,
                        const char *extensions,
                        eer_load_result *result) {
    return eer_load_dir_ex(context, root, extensions, nullptr, result, nullptr);
}

eer_status eer_load_dir_ex(eer_context *context,
                           const char *root,
                           const char *extensions,
                           const eer_chunk_options *chunk_options,
                           eer_load_result *result,
                           const eer_progress *progress) {
    if (result == nullptr) { return EER_STATUS_INVALID_ARGUMENT; }
    *result = eer_load_result{};
    if (!context_is_live(context)) {
        return fail(result->message, EER_STATUS_INVALID_HANDLE,
                    "context handle is null or was already destroyed", &result->sub_code);
    }
    if (root == nullptr) {
        return fail(result->message, EER_STATUS_INVALID_ARGUMENT,
                    "load root must not be null", &result->sub_code);
    }
    embed::ChunkOptions options;
    embed::RetrievalProgress prog;
    string message;
    if (auto status = convert_chunk_options(chunk_options, options, message);
        status != EER_STATUS_OK) {
        return fail(result->message, status, message, &result->sub_code);
    }
    if (auto status = convert_progress(progress, prog, message); status != EER_STATUS_OK) {
        return fail(result->message, status, message, &result->sub_code);
    }
    luisa::filesystem::path directory;
    if (auto status = embed::RetrievalIndex::path_from_utf8(text_or_empty(root), directory);
        !status.ok()) {
        result->sub_code = to_c_sub(status.sub_code);
        return fail(result->message, from_error_code(status.code), status.message);
    }
    luisa::vector<string> parsed_extensions;
    const auto has_rule = extensions != nullptr && extensions[0] != '\0';
    if (has_rule) {
        auto status = embed::RetrievalIndex::parse_extension_list(
            text_or_empty(extensions), parsed_extensions);
        if (!status.ok()) {
            result->sub_code = to_c_sub(status.sub_code);
            return fail(result->message, from_error_code(status.code), status.message);
        }
    } else {
        parsed_extensions = embed::RetrievalIndex::default_extensions();
    }
    auto loaded = context->service->load(
        directory,
        span<const string>{parsed_extensions.data(), parsed_extensions.size()},
        options, progress == nullptr ? nullptr : &prog);
    return fill_load_result(loaded, result);
}

eer_status eer_unload_dir(eer_context *context,
                          const char *root,
                          eer_unload_result *result) {
    if (result == nullptr) { return EER_STATUS_INVALID_ARGUMENT; }
    *result = eer_unload_result{};
    if (!context_is_live(context)) {
        return fail(result->message, EER_STATUS_INVALID_HANDLE,
                    "context handle is null or was already destroyed", &result->sub_code);
    }
    if (root == nullptr) {
        return fail(result->message, EER_STATUS_INVALID_ARGUMENT,
                    "unload root must not be null", &result->sub_code);
    }
    luisa::filesystem::path directory;
    if (auto status = embed::RetrievalIndex::path_from_utf8(text_or_empty(root), directory);
        !status.ok()) {
        result->sub_code = to_c_sub(status.sub_code);
        return fail(result->message, from_error_code(status.code), status.message);
    }
    auto unloaded = context->service->unload(directory);
    if (!unloaded.status.ok()) {
        return fail_result(result, from_error_code(unloaded.status.code),
                           unloaded.status.message, unloaded.status.sub_code);
    }
    result->root = allocate_string(unloaded.root);
    result->removed_documents = unloaded.removed_documents;
    result->total_documents = unloaded.total_documents;
    if (result->root == nullptr) {
        eer_unload_result_free(result);
        return fail(result->message, EER_STATUS_OPERATIONAL, "out of memory", &result->sub_code);
    }
    return EER_STATUS_OK;
}

eer_status eer_update_path(eer_context *context,
                           const char *path,
                           eer_update_result *result) {
    if (result == nullptr) { return EER_STATUS_INVALID_ARGUMENT; }
    *result = eer_update_result{};
    if (!context_is_live(context)) {
        return fail(result->message, EER_STATUS_INVALID_HANDLE,
                    "context handle is null or was already destroyed", &result->sub_code);
    }
    if (path == nullptr) {
        return fail(result->message, EER_STATUS_INVALID_ARGUMENT,
                    "update path must not be null", &result->sub_code);
    }
    luisa::filesystem::path target;
    if (auto status = embed::RetrievalIndex::path_from_utf8(text_or_empty(path), target);
        !status.ok()) {
        result->sub_code = to_c_sub(status.sub_code);
        return fail(result->message, from_error_code(status.code), status.message);
    }
    auto updated = context->service->update(target);
    if (!updated.status.ok()) {
        return fail_result(result, from_error_code(updated.status.code),
                           updated.status.message, updated.status.sub_code);
    }
    result->path = allocate_string(updated.path);
    result->added = updated.added;
    result->updated = updated.updated;
    result->removed = updated.removed;
    result->unchanged = updated.unchanged;
    result->cached = updated.cached;
    result->failed = updated.failed;
    result->total_documents = updated.total_documents;
    if (result->path == nullptr) {
        eer_update_result_free(result);
        return fail(result->message, EER_STATUS_OPERATIONAL, "out of memory", &result->sub_code);
    }
    return EER_STATUS_OK;
}

// --- in-memory documents ------------------------------------------------------

eer_status eer_add_document(eer_context *context,
                            const char *doc_id,
                            const char *utf8_text,
                            size_t text_length,
                            const char *metadata_json,
                            const eer_chunk_options *chunk_options,
                            eer_document_result *result,
                            const eer_progress *progress) {
    if (result == nullptr) { return EER_STATUS_INVALID_ARGUMENT; }
    *result = eer_document_result{};
    if (!context_is_live(context)) {
        return fail(result->message, EER_STATUS_INVALID_HANDLE,
                    "context handle is null or was already destroyed", &result->sub_code);
    }
    if (doc_id == nullptr) {
        return fail(result->message, EER_STATUS_INVALID_ARGUMENT,
                    "doc_id must not be null", &result->sub_code);
    }
    if (utf8_text == nullptr && text_length != 0u) {
        return fail(result->message, EER_STATUS_INVALID_ARGUMENT,
                    "text pointer is null but the length is not zero", &result->sub_code);
    }
    embed::ChunkOptions options;
    embed::RetrievalProgress prog;
    string message;
    if (auto status = convert_chunk_options(chunk_options, options, message);
        status != EER_STATUS_OK) {
        return fail(result->message, status, message, &result->sub_code);
    }
    if (auto status = convert_progress(progress, prog, message); status != EER_STATUS_OK) {
        return fail(result->message, status, message, &result->sub_code);
    }
    const auto text = text_length == 0u ? string_view{} : string_view{utf8_text, text_length};
    auto document = context->service->add_document(
        string{text_or_empty(doc_id)}, text, text_or_empty(metadata_json), options,
        progress == nullptr ? nullptr : &prog);
    return fill_document_result(document, result);
}

eer_status eer_update_document(eer_context *context,
                               const char *doc_id,
                               const char *utf8_text,
                               size_t text_length,
                               const char *metadata_json,
                               const eer_chunk_options *chunk_options,
                               eer_document_result *result,
                               const eer_progress *progress) {
    return eer_add_document(context, doc_id, utf8_text, text_length, metadata_json,
                            chunk_options, result, progress);
}

eer_status eer_remove_document(eer_context *context,
                               const char *doc_id,
                               eer_document_result *result) {
    if (result == nullptr) { return EER_STATUS_INVALID_ARGUMENT; }
    *result = eer_document_result{};
    if (!context_is_live(context)) {
        return fail(result->message, EER_STATUS_INVALID_HANDLE,
                    "context handle is null or was already destroyed", &result->sub_code);
    }
    if (doc_id == nullptr) {
        return fail(result->message, EER_STATUS_INVALID_ARGUMENT,
                    "doc_id must not be null", &result->sub_code);
    }
    auto document = context->service->remove_document(text_or_empty(doc_id));
    return fill_document_result(document, result);
}

// --- query --------------------------------------------------------------------

eer_status eer_query(eer_context *context,
                     const char *utf8_text,
                     size_t text_length,
                     uint32_t top_k,
                     eer_query_result *result) {
    return eer_query_ex(context, utf8_text, text_length, top_k, nullptr, result);
}

eer_status eer_query_ex(eer_context *context,
                        const char *utf8_text,
                        size_t text_length,
                        uint32_t top_k,
                        const eer_query_options *options,
                        eer_query_result *result) {
    if (result == nullptr) { return EER_STATUS_INVALID_ARGUMENT; }
    *result = eer_query_result{};
    if (!context_is_live(context)) {
        return fail(result->message, EER_STATUS_INVALID_HANDLE,
                    "context handle is null or was already destroyed", &result->sub_code);
    }
    if (utf8_text == nullptr && text_length != 0u) {
        return fail(result->message, EER_STATUS_INVALID_ARGUMENT,
                    "query text pointer is null but the length is not zero", &result->sub_code);
    }
    embed::RetrievalQueryOptions query_options;
    string message;
    if (auto status = convert_query_options(options, query_options, message);
        status != EER_STATUS_OK) {
        return fail(result->message, status, message, &result->sub_code);
    }
    const auto text = text_length == 0u ? string_view{} : string_view{utf8_text, text_length};
    const auto requested = top_k == 0u ? uint32_t{EER_DEFAULT_TOP_K} : top_k;
    auto queried = context->service->query(text, requested, &query_options);
    return fill_query_result(queried, result);
}

// --- encode -------------------------------------------------------------------

eer_status eer_encode_text(eer_context *context,
                           const char *utf8_text,
                           size_t text_length,
                           float *embedding,
                           size_t embedding_capacity,
                           size_t *embedding_dimension,
                           eer_error *out_error) {
    if (embedding_dimension == nullptr) { return EER_STATUS_INVALID_ARGUMENT; }
    *embedding_dimension = 0u;
    if (out_error != nullptr) { *out_error = eer_error{}; }
    const auto report = [out_error](eer_status code, string_view text,
                                    int32_t sub = EER_SUB_NONE) {
        if (out_error != nullptr) {
            out_error->message = allocate_string(text);
            out_error->sub_code = sub;
        }
        return code;
    };
    if (!context_is_live(context)) {
        return report(EER_STATUS_INVALID_HANDLE, "context handle is null or was already destroyed");
    }
    if (utf8_text == nullptr && text_length != 0u) {
        return report(EER_STATUS_INVALID_ARGUMENT,
                      "text pointer is null but the length is not zero");
    }
    const auto text = text_length == 0u ? string_view{} : string_view{utf8_text, text_length};
    auto dense = encode_text(context, text);
    if (dense.empty()) {
        return report(EER_STATUS_OPERATIONAL,
                      "the encoder returned an empty dense embedding",
                      EER_SUB_ENCODE);
    }
    *embedding_dimension = dense.size();
    if (embedding == nullptr || embedding_capacity == 0u) {
        // Dimension query: nothing to write.
        return EER_STATUS_OK;
    }
    if (embedding_capacity < dense.size()) {
        return report(EER_STATUS_INVALID_ARGUMENT,
                      format("embedding buffer holds {} float(s) but the encoder returns {}",
                             embedding_capacity, dense.size()));
    }
    std::memcpy(embedding, dense.data(), dense.size() * sizeof(float));
    return EER_STATUS_OK;
}

eer_status eer_encode_batch(eer_context *context,
                            const eer_text *texts,
                            size_t count,
                            float *out_vectors,
                            size_t capacity,
                            size_t *out_dimension,
                            eer_error *out_error) {
    if (out_dimension == nullptr) { return EER_STATUS_INVALID_ARGUMENT; }
    *out_dimension = 0u;
    if (out_error != nullptr) { *out_error = eer_error{}; }
    const auto report = [out_error](eer_status code, string_view text,
                                    int32_t sub = EER_SUB_NONE) {
        if (out_error != nullptr) {
            out_error->message = allocate_string(text);
            out_error->sub_code = sub;
        }
        return code;
    };
    if (!context_is_live(context)) {
        return report(EER_STATUS_INVALID_HANDLE, "context handle is null or was already destroyed");
    }
    if (count == 0u || texts == nullptr) {
        return report(EER_STATUS_INVALID_ARGUMENT, "texts must not be empty");
    }
    for (size_t i = 0u; i < count; i++) {
        if (texts[i].utf8 == nullptr && texts[i].length != 0u) {
            return report(EER_STATUS_INVALID_ARGUMENT,
                          format("texts[{}] pointer is null but the length is not zero", i));
        }
    }
    luisa::vector<span<const float>> rows;
    rows.reserve(count);
    if (context->encode_fn == nullptr) {
        // Engine mode: one scheduler round trip through encode_batch.
        luisa::vector<string_view> views;
        views.reserve(count);
        for (size_t i = 0u; i < count; i++) {
            views.push_back(texts[i].length == 0u ? string_view{} :
                                                    string_view{texts[i].utf8, texts[i].length});
        }
        auto embeddings = context->session->encode_batch(views);
        if (embeddings.size() != count) {
            return report(EER_STATUS_OPERATIONAL, "the batch encode did not complete",
                          EER_SUB_ENCODE);
        }
        for (auto &embedding : embeddings) {
            rows.push_back(span<const float>{embedding.dense.data(), embedding.dense.size()});
        }
        // Keep the embeddings alive until the copy below: stash them.
        // (rows reference their dense storage.)
        const auto dimension = rows.front().size();
        for (size_t i = 0u; i < count; i++) {
            if (rows[i].size() != dimension) {
                return report(EER_STATUS_OPERATIONAL,
                              format("text {} produced dimension {}, expected {}",
                                     i, rows[i].size(), dimension),
                              EER_SUB_ENCODE);
            }
        }
        *out_dimension = dimension;
        if (out_vectors == nullptr || capacity == 0u) { return EER_STATUS_OK; }
        if (capacity < count * dimension) {
            return report(EER_STATUS_INVALID_ARGUMENT,
                          format("output buffer holds {} float(s) but the batch needs {}",
                                 capacity, count * dimension));
        }
        for (size_t i = 0u; i < count; i++) {
            std::memcpy(out_vectors + i * dimension, rows[i].data(), dimension * sizeof(float));
        }
        return EER_STATUS_OK;
    }
    // Callback mode: encode per text, then one uniform-dimension check.
    luisa::vector<luisa::vector<float>> embeddings;
    embeddings.reserve(count);
    for (size_t i = 0u; i < count; i++) {
        const auto text = texts[i].length == 0u ? string_view{} :
                                                  string_view{texts[i].utf8, texts[i].length};
        embeddings.push_back(encode_with_host_callback(context, text));
        if (embeddings.back().empty()) {
            return report(EER_STATUS_OPERATIONAL,
                          format("the encoder returned an empty dense embedding for text {}", i),
                          EER_SUB_ENCODE);
        }
    }
    const auto dimension = embeddings.front().size();
    for (size_t i = 0u; i < count; i++) {
        if (embeddings[i].size() != dimension) {
            return report(EER_STATUS_OPERATIONAL,
                          format("text {} produced dimension {}, expected {}",
                                 i, embeddings[i].size(), dimension),
                          EER_SUB_ENCODE);
        }
    }
    *out_dimension = dimension;
    if (out_vectors == nullptr || capacity == 0u) { return EER_STATUS_OK; }
    if (capacity < count * dimension) {
        return report(EER_STATUS_INVALID_ARGUMENT,
                      format("output buffer holds {} float(s) but the batch needs {}",
                             capacity, count * dimension));
    }
    for (size_t i = 0u; i < count; i++) {
        std::memcpy(out_vectors + i * dimension, embeddings[i].data(), dimension * sizeof(float));
    }
    return EER_STATUS_OK;
}

eer_status eer_encode_colbert(eer_context *context,
                              const char *utf8_text,
                              size_t text_length,
                              float *out,
                              size_t capacity,
                              size_t *out_values,
                              uint32_t *out_rows,
                              eer_error *out_error) {
    if (out_values == nullptr || out_rows == nullptr) { return EER_STATUS_INVALID_ARGUMENT; }
    *out_values = 0u;
    *out_rows = 0u;
    if (out_error != nullptr) { *out_error = eer_error{}; }
    const auto report = [out_error](eer_status code, string_view text,
                                    int32_t sub = EER_SUB_NONE) {
        if (out_error != nullptr) {
            out_error->message = allocate_string(text);
            out_error->sub_code = sub;
        }
        return code;
    };
    if (!context_is_live(context)) {
        return report(EER_STATUS_INVALID_HANDLE, "context handle is null or was already destroyed");
    }
    if (utf8_text == nullptr && text_length != 0u) {
        return report(EER_STATUS_INVALID_ARGUMENT,
                      "text pointer is null but the length is not zero");
    }
    if (context->encode_fn != nullptr) {
        return report(EER_STATUS_OPERATIONAL,
                      "ColBERT encoding requires engine mode (no host ColBERT callback exists)",
                      EER_SUB_ENCODE);
    }
    const auto text = text_length == 0u ? string_view{} : string_view{utf8_text, text_length};
    embed::EncodeOptions options;
    options.want_colbert = true;
    auto encoded = context->session->encode(text, options);
    if (encoded.dense.empty() || encoded.colbert.empty()) {
        return report(EER_STATUS_OPERATIONAL,
                      "the encoder returned an empty ColBERT matrix",
                      EER_SUB_ENCODE);
    }
    const auto dimension = encoded.dense.size();
    const auto rows = encoded.colbert.size() / dimension;
    *out_values = encoded.colbert.size();
    *out_rows = static_cast<uint32_t>(rows);
    if (out == nullptr || capacity == 0u) { return EER_STATUS_OK; }
    if (capacity < encoded.colbert.size()) {
        return report(EER_STATUS_INVALID_ARGUMENT,
                      format("output buffer holds {} float(s) but the ColBERT matrix needs {}",
                             capacity, encoded.colbert.size()));
    }
    std::memcpy(out, encoded.colbert.data(), encoded.colbert.size() * sizeof(float));
    return EER_STATUS_OK;
}

// --- snapshots ----------------------------------------------------------------

eer_status eer_index_save(eer_context *context,
                          const char *path,
                          eer_error *out_error) {
    if (out_error != nullptr) { *out_error = eer_error{}; }
    const auto report = [out_error](eer_status code, string_view text,
                                    int32_t sub = EER_SUB_NONE) {
        if (out_error != nullptr) {
            out_error->message = allocate_string(text);
            out_error->sub_code = sub;
        }
        return code;
    };
    if (!context_is_live(context)) {
        return report(EER_STATUS_INVALID_HANDLE, "context handle is null or was already destroyed");
    }
    if (path == nullptr || path[0] == '\0') {
        return report(EER_STATUS_INVALID_ARGUMENT, "snapshot path must not be empty");
    }
    luisa::filesystem::path target;
    if (auto status = embed::RetrievalIndex::path_from_utf8(text_or_empty(path), target);
        !status.ok()) {
        return report(from_error_code(status.code), status.message, to_c_sub(status.sub_code));
    }
    auto status = context->service->save(target);
    if (!status.ok()) {
        return report(from_error_code(status.code), status.message, to_c_sub(status.sub_code));
    }
    return EER_STATUS_OK;
}

eer_status eer_index_load(eer_context *context,
                          const char *path,
                          eer_error *out_error) {
    if (out_error != nullptr) { *out_error = eer_error{}; }
    const auto report = [out_error](eer_status code, string_view text,
                                    int32_t sub = EER_SUB_NONE) {
        if (out_error != nullptr) {
            out_error->message = allocate_string(text);
            out_error->sub_code = sub;
        }
        return code;
    };
    if (!context_is_live(context)) {
        return report(EER_STATUS_INVALID_HANDLE, "context handle is null or was already destroyed");
    }
    if (path == nullptr || path[0] == '\0') {
        return report(EER_STATUS_INVALID_ARGUMENT, "snapshot path must not be empty");
    }
    luisa::filesystem::path source;
    if (auto status = embed::RetrievalIndex::path_from_utf8(text_or_empty(path), source);
        !status.ok()) {
        return report(from_error_code(status.code), status.message, to_c_sub(status.sub_code));
    }
    auto status = context->service->load_snapshot(source);
    if (!status.ok()) {
        return report(from_error_code(status.code), status.message, to_c_sub(status.sub_code));
    }
    return EER_STATUS_OK;
}

// --- serving surface ------------------------------------------------------------

eer_status eer_stats(eer_context *context, eer_stats_data *out_stats) {
    if (out_stats == nullptr) { return EER_STATUS_INVALID_ARGUMENT; }
    // struct_size-gated writes: a caller whose struct ends before a field (a
    // future 1.3 caller, or any caller that passes a truncated size) is never
    // written past. struct_size == 0 means "the full current layout".
    const auto struct_size = out_stats->struct_size;
    *out_stats = eer_stats_data{};
    out_stats->struct_size = struct_size;
    if (!context_is_live(context)) { return EER_STATUS_INVALID_HANDLE; }
    const auto limit = struct_size == 0u ? sizeof(eer_stats_data) : struct_size;
    const auto fits = [limit](size_t offset, size_t size) noexcept {
        return offset + size <= limit;
    };
    const auto stats = context->service->stats();
    if (fits(offsetof(eer_stats_data, document_count), sizeof(uint64_t))) {
        out_stats->document_count = stats.document_count;
    }
    if (fits(offsetof(eer_stats_data, chunk_count), sizeof(uint64_t))) {
        out_stats->chunk_count = stats.chunk_count;
    }
    if (fits(offsetof(eer_stats_data, dimension), sizeof(uint32_t))) {
        out_stats->dimension = stats.dimension;
    }
    if (fits(offsetof(eer_stats_data, hybrid), sizeof(uint32_t))) {
        out_stats->hybrid = stats.hybrid ? 1u : 0u;
    }
    if (fits(offsetof(eer_stats_data, cache_hit_ratio), sizeof(double))) {
        out_stats->cache_hit_ratio = stats.cache_hit_ratio;
    }
    if (fits(offsetof(eer_stats_data, mean_encode_ms), sizeof(double))) {
        out_stats->mean_encode_ms = stats.mean_encode_ms;
    }
    if (fits(offsetof(eer_stats_data, encode_count), sizeof(uint64_t))) {
        out_stats->encode_count = stats.encode_count;
    }
    if (fits(offsetof(eer_stats_data, cache_hits), sizeof(uint64_t))) {
        out_stats->cache_hits = stats.cache_hits;
    }
    if (fits(offsetof(eer_stats_data, cache_misses), sizeof(uint64_t))) {
        out_stats->cache_misses = stats.cache_misses;
    }
    return EER_STATUS_OK;
}

void eer_stats_data_free(eer_stats_data *stats) {
    if (stats == nullptr) { return; }
    *stats = eer_stats_data{};
}

eer_status eer_context_ping(eer_context *context, eer_error *out_error) {
    if (out_error != nullptr) { *out_error = eer_error{}; }
    const auto report = [out_error](eer_status code, string_view text,
                                    int32_t sub = EER_SUB_NONE) {
        if (out_error != nullptr) {
            out_error->message = allocate_string(text);
            out_error->sub_code = sub;
        }
        return code;
    };
    if (!context_is_live(context)) {
        return report(EER_STATUS_INVALID_HANDLE, "context handle is null or was already destroyed");
    }
    auto dense = encode_text(context, string_view{kPingText});
    if (dense.empty()) {
        return report(EER_STATUS_OPERATIONAL,
                      "the ping encode returned an empty dense embedding",
                      context->encode_fn != nullptr ? EER_SUB_ENCODE : EER_SUB_DEVICE);
    }
    return EER_STATUS_OK;
}

eer_status eer_context_device_memory(eer_context *context,
                                     uint64_t *out_used_bytes,
                                     uint64_t *out_total_bytes) {
    if (out_used_bytes == nullptr || out_total_bytes == nullptr) {
        return EER_STATUS_INVALID_ARGUMENT;
    }
    *out_used_bytes = 0u;
    *out_total_bytes = 0u;
    if (!context_is_live(context)) { return EER_STATUS_INVALID_HANDLE; }
    // LuisaCompute does not expose a device memory-stats API: report the
    // honest "not available" pair (see the header comment).
    return EER_STATUS_OK;
}

eer_status eer_context_park(eer_context *context, eer_error *out_error) {
    if (out_error != nullptr) { *out_error = eer_error{}; }
    const auto report = [out_error](eer_status code, string_view text) {
        if (out_error != nullptr) { out_error->message = allocate_string(text); }
        return code;
    };
    if (!context_is_live(context)) {
        return report(EER_STATUS_INVALID_HANDLE, "context handle is null or was already destroyed");
    }
    return report(EER_STATUS_OPERATIONAL,
                  "park/resume is not supported in this build; the index stays resident");
}

eer_status eer_context_resume(eer_context *context, eer_error *out_error) {
    return eer_context_park(context, out_error);
}

// --- memory management ----------------------------------------------------------

// The payloads above are allocated with the engine allocator
// (allocate_with_allocator), so they are released here with the matching
// deallocate_with_allocator -- never with C free(), which may use a different
// CRT/heap across the shared-library boundary.
void eer_error_free(eer_error *error) {
    if (error == nullptr) { return; }
    deallocate_with_allocator(error->message);
    *error = eer_error{};
}

void eer_load_result_free(eer_load_result *result) {
    if (result == nullptr) { return; }
    deallocate_with_allocator(result->message);
    deallocate_with_allocator(result->root);
    deallocate_with_allocator(result->extensions);
    *result = eer_load_result{};
}

void eer_unload_result_free(eer_unload_result *result) {
    if (result == nullptr) { return; }
    deallocate_with_allocator(result->message);
    deallocate_with_allocator(result->root);
    *result = eer_unload_result{};
}

void eer_update_result_free(eer_update_result *result) {
    if (result == nullptr) { return; }
    deallocate_with_allocator(result->message);
    deallocate_with_allocator(result->path);
    *result = eer_update_result{};
}

void eer_query_result_free(eer_query_result *result) {
    if (result == nullptr) { return; }
    if (result->hits != nullptr) {
        for (size_t i = 0u; i < result->hit_count; i++) {
            deallocate_with_allocator(result->hits[i].path);
            deallocate_with_allocator(result->hits[i].doc_id);
        }
        deallocate_with_allocator(result->hits);
    }
    deallocate_with_allocator(result->message);
    *result = eer_query_result{};
}

void eer_document_result_free(eer_document_result *result) {
    if (result == nullptr) { return; }
    deallocate_with_allocator(result->message);
    deallocate_with_allocator(result->doc_id);
    *result = eer_document_result{};
}
