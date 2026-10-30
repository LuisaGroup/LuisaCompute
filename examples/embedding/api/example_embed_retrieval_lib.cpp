// C ABI implementation for the bge-m3 whole-file retrieval engine
// (example_embed_retrieval_lib.h; retrieval plan phases 2-4). This translation
// unit is the only place where the C boundary meets LuisaCompute / engine
// types: every entry point validates its arguments, converts UTF-8 into the
// engine's path/string types, runs one engine step, and copies the result back
// into caller-owned C structs. The engine sources are untouched; the steps
// mirror example_embed_retrieval.cpp (load/unload/query/update) and add a
// host-encoder mode that runs without a GPU.

#include "example_embed_retrieval_lib.h"

#include <cstdlib>
#include <cstring>
#include <system_error>

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
    luisa::unique_ptr<luisa::embed::Session> session;
    luisa::unique_ptr<luisa::embed::DiskEmbedCache> disk_cache;
    luisa::unique_ptr<luisa::embed::RetrievalService> service;
    eer_encode_fn encode_fn{nullptr};
    void *encode_user_data{nullptr};
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
// Scheduler worker count used when the caller passes 0.
constexpr uint32_t kDefaultWorkerCount = 2u;
// Backend used when the caller does not name one (same default as the console).
constexpr char kDefaultBackend[] = "vk";
// The executable performs exactly this warm-up encode; keep the two in sync.
constexpr char kWarmupText[] = "retrieval session warmup";

#define EER_STRINGIFY_INNER(x) #x
#define EER_STRINGIFY(x) EER_STRINGIFY_INNER(x)
constexpr char kVersionString[] =
    EER_STRINGIFY(EER_VERSION_MAJOR) "." EER_STRINGIFY(EER_VERSION_MINOR) "." EER_STRINGIFY(EER_VERSION_PATCH);

// --- small helpers ----------------------------------------------------------

// Allocates a NUL-terminated copy of `text`; nullptr means out of memory, which
// callers report as a failed (but non-fatal) result field.
[[nodiscard]] char *allocate_string(string_view text) noexcept {
    auto *buffer = static_cast<char *>(std::malloc(text.size() + 1u));
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

// Fills a result's message slot and returns the status, so a step can write
// `return fail(result->message, code, "...")`.
[[nodiscard]] eer_status fail(char *&message, eer_status code, string_view text) noexcept {
    message = allocate_string(text);
    return code;
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

// The single encoder the index sees: the host callback in callback mode, the
// model session otherwise.
[[nodiscard]] luisa::vector<float> encode_text(eer_context *context, string_view text) {
    if (context->encode_fn != nullptr) { return encode_with_host_callback(context, text); }
    auto embedding = context->session->encode(text);
    return std::move(embedding.dense);
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
    if (out_error != nullptr) { out_error->message = nullptr; }

    const auto report = [out_error](eer_status code, string_view text) {
        if (out_error != nullptr) { out_error->message = allocate_string(text); }
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
    const auto want_disk_cache = (info->flags & EER_CREATE_DISABLE_DISK_CACHE) == 0u;
    const auto want_warmup = (info->flags & EER_CREATE_SKIP_WARMUP) == 0u;
    const auto worker_count = info->num_workers == 0u ? kDefaultWorkerCount : info->num_workers;

    if (context->encode_fn == nullptr) {
        // Engine mode: a GPU backend session over a packed model.
        const auto backend = text_or_empty(info->backend).empty() ? string_view{kDefaultBackend} : text_or_empty(info->backend);
        luisa::filesystem::path model_path;
        luisa::filesystem::path tokenizer_path;
        string message;
        if (auto status = resolve_regular_file(text_or_empty(info->model_path), model_path, message);
            status != EER_STATUS_OK) {
            return report(status, message);
        }
        if (auto status = resolve_regular_file(text_or_empty(info->tokenizer_path), tokenizer_path, message);
            status != EER_STATUS_OK) {
            return report(status, message);
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
                                 backend, installed.empty() ? string{"<none>"} : installed));
        }
        context->backend_name = string{backend};
        context->device = luisa::make_unique<Device>(std::move(device));
        context->session = embed::Session::create(
            *context->device, model_path, tokenizer_path, worker_count);
        if (context->session == nullptr) {
            return report(EER_STATUS_OPERATIONAL, "Session::create failed");
        }
        // Fully initialize and drain the backend execution path once, exactly
        // like the executable does before accepting commands.
        if (want_warmup) {
            auto warmup = context->session->encode(string_view{kWarmupText});
            if (warmup.dense.empty()) {
                return report(EER_STATUS_OPERATIONAL,
                              "session warm-up returned an empty dense embedding");
            }
        }
    }

    context->service = luisa::make_unique<embed::RetrievalService>(
        [context = context.get()](string_view text) {
            return encode_text(context, text);
        });

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

// --- load / unload / update / query -----------------------------------------

eer_status eer_load_dir(eer_context *context,
                        const char *root,
                        const char *extensions,
                        eer_load_result *result) {
    if (result == nullptr) { return EER_STATUS_INVALID_ARGUMENT; }
    *result = eer_load_result{};
    if (!context_is_live(context)) {
        return fail(result->message, EER_STATUS_INVALID_HANDLE,
                    "context handle is null or was already destroyed");
    }
    if (root == nullptr) {
        return fail(result->message, EER_STATUS_INVALID_ARGUMENT,
                    "load root must not be null");
    }
    luisa::filesystem::path directory;
    if (auto status = embed::RetrievalIndex::path_from_utf8(text_or_empty(root), directory);
        !status.ok()) {
        return fail(result->message, from_error_code(status.code), status.message);
    }
    luisa::vector<string> parsed_extensions;
    const auto has_rule = extensions != nullptr && extensions[0] != '\0';
    if (has_rule) {
        auto status = embed::RetrievalIndex::parse_extension_list(
            text_or_empty(extensions), parsed_extensions);
        if (!status.ok()) {
            return fail(result->message, from_error_code(status.code), status.message);
        }
    }
    auto loaded = has_rule ? context->service->load(
                                 directory,
                                 span<const string>{parsed_extensions.data(), parsed_extensions.size()}) :
                             context->service->load(directory);
    if (!loaded.status.ok()) {
        return fail(result->message, from_error_code(loaded.status.code), loaded.status.message);
    }
    result->root = allocate_string(loaded.root);
    result->extensions = allocate_string(join_extensions(
        span<const string>{loaded.extensions.data(), loaded.extensions.size()}));
    result->added = loaded.added;
    result->cached = loaded.cached;
    result->skipped = loaded.skipped;
    result->failed = loaded.failed;
    result->total_documents = loaded.total_documents;
    if (result->root == nullptr || result->extensions == nullptr) {
        eer_load_result_free(result);
        return fail(result->message, EER_STATUS_OPERATIONAL, "out of memory");
    }
    return EER_STATUS_OK;
}

eer_status eer_unload_dir(eer_context *context,
                          const char *root,
                          eer_unload_result *result) {
    if (result == nullptr) { return EER_STATUS_INVALID_ARGUMENT; }
    *result = eer_unload_result{};
    if (!context_is_live(context)) {
        return fail(result->message, EER_STATUS_INVALID_HANDLE,
                    "context handle is null or was already destroyed");
    }
    if (root == nullptr) {
        return fail(result->message, EER_STATUS_INVALID_ARGUMENT,
                    "unload root must not be null");
    }
    luisa::filesystem::path directory;
    if (auto status = embed::RetrievalIndex::path_from_utf8(text_or_empty(root), directory);
        !status.ok()) {
        return fail(result->message, from_error_code(status.code), status.message);
    }
    auto unloaded = context->service->unload(directory);
    if (!unloaded.status.ok()) {
        return fail(result->message, from_error_code(unloaded.status.code), unloaded.status.message);
    }
    result->root = allocate_string(unloaded.root);
    result->removed_documents = unloaded.removed_documents;
    result->total_documents = unloaded.total_documents;
    if (result->root == nullptr) {
        eer_unload_result_free(result);
        return fail(result->message, EER_STATUS_OPERATIONAL, "out of memory");
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
                    "context handle is null or was already destroyed");
    }
    if (path == nullptr) {
        return fail(result->message, EER_STATUS_INVALID_ARGUMENT,
                    "update path must not be null");
    }
    luisa::filesystem::path target;
    if (auto status = embed::RetrievalIndex::path_from_utf8(text_or_empty(path), target);
        !status.ok()) {
        return fail(result->message, from_error_code(status.code), status.message);
    }
    auto updated = context->service->update(target);
    if (!updated.status.ok()) {
        return fail(result->message, from_error_code(updated.status.code), updated.status.message);
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
        return fail(result->message, EER_STATUS_OPERATIONAL, "out of memory");
    }
    return EER_STATUS_OK;
}

eer_status eer_query(eer_context *context,
                     const char *utf8_text,
                     size_t text_length,
                     uint32_t top_k,
                     eer_query_result *result) {
    if (result == nullptr) { return EER_STATUS_INVALID_ARGUMENT; }
    *result = eer_query_result{};
    if (!context_is_live(context)) {
        return fail(result->message, EER_STATUS_INVALID_HANDLE,
                    "context handle is null or was already destroyed");
    }
    if (utf8_text == nullptr && text_length != 0u) {
        return fail(result->message, EER_STATUS_INVALID_ARGUMENT,
                    "query text pointer is null but the length is not zero");
    }
    const auto text = text_length == 0u ? string_view{} : string_view{utf8_text, text_length};
    const auto requested = top_k == 0u ? uint32_t{EER_DEFAULT_TOP_K} : top_k;
    auto queried = context->service->query(text, requested);
    if (!queried.status.ok()) {
        return fail(result->message, from_error_code(queried.status.code), queried.status.message);
    }
    if (queried.hits.empty()) { return EER_STATUS_OK; }
    auto *hits = static_cast<eer_hit *>(std::calloc(queried.hits.size(), sizeof(eer_hit)));
    if (hits == nullptr) {
        return fail(result->message, EER_STATUS_OPERATIONAL,
                    "out of memory while building the hit list");
    }
    result->hits = hits;
    result->hit_count = queried.hits.size();
    for (size_t i = 0u; i < queried.hits.size(); i++) {
        hits[i].score = queried.hits[i].score;
        hits[i].path = allocate_string(queried.hits[i].path);
        if (hits[i].path == nullptr) {
            eer_query_result_free(result);
            return fail(result->message, EER_STATUS_OPERATIONAL,
                        "out of memory while building the hit list");
        }
    }
    return EER_STATUS_OK;
}

// --- encode -----------------------------------------------------------------

eer_status eer_encode_text(eer_context *context,
                           const char *utf8_text,
                           size_t text_length,
                           float *embedding,
                           size_t embedding_capacity,
                           size_t *embedding_dimension,
                           eer_error *out_error) {
    if (embedding_dimension == nullptr) { return EER_STATUS_INVALID_ARGUMENT; }
    *embedding_dimension = 0u;
    if (out_error != nullptr) { out_error->message = nullptr; }
    const auto report = [out_error](eer_status code, string_view text) {
        if (out_error != nullptr) { out_error->message = allocate_string(text); }
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
                      "the encoder returned an empty dense embedding");
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

// --- memory management ------------------------------------------------------

void eer_error_free(eer_error *error) {
    if (error == nullptr) { return; }
    std::free(error->message);
    *error = eer_error{};
}

void eer_load_result_free(eer_load_result *result) {
    if (result == nullptr) { return; }
    std::free(result->message);
    std::free(result->root);
    std::free(result->extensions);
    *result = eer_load_result{};
}

void eer_unload_result_free(eer_unload_result *result) {
    if (result == nullptr) { return; }
    std::free(result->message);
    std::free(result->root);
    *result = eer_unload_result{};
}

void eer_update_result_free(eer_update_result *result) {
    if (result == nullptr) { return; }
    std::free(result->message);
    std::free(result->path);
    *result = eer_update_result{};
}

void eer_query_result_free(eer_query_result *result) {
    if (result == nullptr) { return; }
    if (result->hits != nullptr) {
        for (size_t i = 0u; i < result->hit_count; i++) {
            std::free(result->hits[i].path);
        }
        std::free(result->hits);
    }
    std::free(result->message);
    *result = eer_query_result{};
}
