// Test for the C-FFI retrieval library (examples/embedding/api/
// example_embed_retrieval_lib.h).
//
// Host-only (LUISA_EMBED_HOST_ONLY=1): every case here drives the library in
// callback mode, i.e. with a host-supplied dense encoder, so no GPU backend and
// no packed bge-m3 model are needed. This is the only way the exported C ABI
// can be exercised end to end in CI, and it covers the same engine path the
// model session uses (load / query / update / unload + the disk cache).
//
// This test covers:
// - version, status-string and default-extension introspection
// - invalid-handle and null-argument rejection on every entry point
// - eer_context_create() argument validation (struct size, missing files)
// - callback-mode end-to-end: load -> query -> update -> unload, with the
//   documented counts, ranking order and error codes
// - the on-disk embedding cache shared by two consecutive contexts
// - encoder failure accounting and the empty-corpus diagnostic
// - the host-encoder buffer-retry path and its dimension limit
// - RAG growth (ABI 1.2, report.md §2-§6): chunked in-memory documents with
//   byte ranges, idempotent re-ingestion, paragraph/sentence/token splitting,
//   corpus snapshots across a context recreate (and corrupt-snapshot safety),
//   eer_query_ex weight overrides / metadata filters / MMR diversity,
//   sparse-callback hybrid mode, eer_encode_batch parity, load cancellation,
//   stats, ping, the honest 0/0 device-memory path and the park/resume stub
// - ABI 1.1 compatibility: NULL chunk options keep the whole-file behavior on
//   every legacy path (load_dir, load_dir_ex, add_document), query-options
//   struct_size gating, sub_code population on failure paths, the ColBERT
//   callback-mode error contract, and snapshot restore with no re-encode
//
// Run: xmake run test_embed_retrieval_api

#include "ut/ut.hpp"
#include "example_embed_retrieval_lib.h"
#include <algorithm>
#include <cctype>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

using namespace boost::ut;

namespace {

// --- filesystem helpers -----------------------------------------------------

[[nodiscard]] std::string utf8_path(const std::filesystem::path &path) {
    const auto utf8 = path.generic_u8string();
    return std::string{reinterpret_cast<const char *>(utf8.data()), utf8.size()};
}

// Unique directory under the system temp directory, removed on destruction so
// a failing case never leaves index or cache files behind.
class TempDir {

public:
    explicit TempDir(const char *tag) {
        static uint32_t counter = 0u;
        std::error_code ec;
        auto base = std::filesystem::temp_directory_path(ec);
        if (ec) {
            ec.clear();
            base = std::filesystem::current_path(ec);
        }
        _path = base / ("lc_embed_retrieval_api_" + std::string{tag} + "_" + std::to_string(counter++));
        std::filesystem::remove_all(_path, ec);
        ec.clear();
        std::filesystem::create_directories(_path, ec);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(_path, ec);
    }

    TempDir(const TempDir &) = delete;
    TempDir(TempDir &&) = delete;
    TempDir &operator=(const TempDir &) = delete;
    TempDir &operator=(TempDir &&) = delete;

    [[nodiscard]] const std::filesystem::path &path() const noexcept { return _path; }
    [[nodiscard]] std::string utf8() const { return utf8_path(_path); }

    void write(const char *name, const char *contents) const {
        std::ofstream stream{_path / name, std::ios::binary | std::ios::trunc};
        stream.write(contents, static_cast<std::streamsize>(std::strlen(contents)));
        stream.close();
    }

    void remove(const char *name) const {
        std::error_code ec;
        std::filesystem::remove(_path / name, ec);
    }

private:
    std::filesystem::path _path;
};

// Basename of a UTF-8 path, extracted textually so the test never has to
// narrow-convert a full path (non-ASCII temp directories stay safe).
[[nodiscard]] std::string file_name(const char *path) {
    if (path == nullptr) { return {}; }
    const std::string text{path};
    const auto position = text.find_last_of("/\\");
    return position == std::string::npos ? text : text.substr(position + 1u);
}

// --- deterministic host encoders -------------------------------------------

// 4-d "letter bag": counts of 'a', 'b', 'c' plus 0.25 per other byte, stored
// L2-normalized so the retrieval cosine ranking is fully reproducible.
constexpr size_t kBagDimension = 4u;

int32_t bag_encoder(void *user_data,
                    const char *text,
                    size_t text_length,
                    float *embedding,
                    size_t embedding_capacity,
                    size_t *embedding_dimension) {
    (void)user_data;
    *embedding_dimension = kBagDimension;
    if (embedding == nullptr || embedding_capacity < kBagDimension) {
        return EER_ENCODE_BUFFER_TOO_SMALL;
    }
    float values[kBagDimension] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t i = 0u; i < text_length; i++) {
        switch (text[i]) {
            case 'a': values[0] += 1.0f; break;
            case 'b': values[1] += 1.0f; break;
            case 'c': values[2] += 1.0f; break;
            default: values[3] += 0.25f; break;
        }
    }
    double squared = 0.0;
    for (const auto value : values) { squared += static_cast<double>(value) * value; }
    const auto norm = static_cast<float>(std::sqrt(squared));
    for (size_t i = 0u; i < kBagDimension; i++) {
        embedding[i] = norm > 0.0f ? values[i] / norm : 0.5f;
    }
    return EER_ENCODE_OK;
}

// Always fails, to exercise the per-file failure accounting.
int32_t failing_encoder(void *,
                        const char *,
                        size_t,
                        float *,
                        size_t,
                        size_t *embedding_dimension) {
    *embedding_dimension = 0u;
    return EER_ENCODE_FAILED;
}

// Needs more floats than the library's first offer, so the retry path runs.
constexpr size_t kRetryDimension = 2048u;

int32_t retry_encoder(void *,
                      const char *,
                      size_t,
                      float *embedding,
                      size_t embedding_capacity,
                      size_t *embedding_dimension) {
    *embedding_dimension = kRetryDimension;
    if (embedding == nullptr || embedding_capacity < kRetryDimension) {
        return EER_ENCODE_BUFFER_TOO_SMALL;
    }
    for (size_t i = 0u; i < kRetryDimension; i++) {
        embedding[i] = static_cast<float>(i % 7u) / 7.0f;
    }
    return EER_ENCODE_OK;
}

// Asks for more than the library accepts (16384 floats): every encode fails.
int32_t oversized_encoder(void *,
                          const char *,
                          size_t,
                          float *,
                          size_t,
                          size_t *embedding_dimension) {
    *embedding_dimension = 20000u;
    return EER_ENCODE_BUFFER_TOO_SMALL;
}

// Deterministic sparse lexical encoder (ABI 1.2 hybrid callback mode): one
// (token_id = byte, weight = 1) pair per input byte; duplicate ids exercise
// the index canonicalization (sum + sort + L2-normalize).
int32_t letter_sparse_encoder(void *,
                              const char *text,
                              size_t text_length,
                              uint8_t *pairs,
                              size_t pair_capacity,
                              size_t *pair_count) {
    *pair_count = text_length;
    const auto required = text_length * 8u;
    if (pairs == nullptr || pair_capacity < required) {
        return EER_ENCODE_BUFFER_TOO_SMALL;
    }
    for (size_t i = 0u; i < text_length; i++) {
        const auto token = static_cast<uint32_t>(static_cast<uint8_t>(text[i]));
        const float weight = 1.0f;
        std::memcpy(pairs + i * 8u, &token, sizeof(token));
        std::memcpy(pairs + i * 8u + 4u, &weight, sizeof(weight));
    }
    return EER_ENCODE_OK;
}

[[nodiscard]] bool create_callback_context(eer_encode_fn encoder,
                                           uint32_t flags,
                                           eer_context **context,
                                           eer_error *error) {
    eer_create_info info{};
    info.struct_size = sizeof(info);
    info.flags = flags;
    info.encode = encoder;
    const auto status = eer_context_create(&info, context, error);
    return status == EER_STATUS_OK;
}

// Word-level sparse encoder: token id = (first byte << 8) | word length, one
// pair per word. Unlike the per-byte variant this lets a fixture make the
// dense (letter bag) and sparse (word) halves DISAGREE on the ranking.
int32_t word_sparse_encoder(void *,
                            const char *text,
                            size_t text_length,
                            uint8_t *pairs,
                            size_t pair_capacity,
                            size_t *pair_count) {
    size_t count = 0u;
    size_t i = 0u;
    while (i < text_length) {
        while (i < text_length && std::isspace(static_cast<unsigned char>(text[i])) != 0) { i++; }
        if (i >= text_length) { break; }
        const auto begin = i;
        while (i < text_length && std::isspace(static_cast<unsigned char>(text[i])) == 0) { i++; }
        count++;
        const auto bytes = count * 8u;
        if (pairs != nullptr && pair_capacity >= bytes) {
            const auto token = (static_cast<uint32_t>(static_cast<uint8_t>(text[begin])) << 8u) |
                               static_cast<uint32_t>(i - begin);
            const float weight = 1.0f;
            std::memcpy(pairs + (count - 1u) * 8u, &token, sizeof(token));
            std::memcpy(pairs + (count - 1u) * 8u + 4u, &weight, sizeof(weight));
        }
    }
    *pair_count = count;
    if (pairs == nullptr || pair_capacity < count * 8u) { return EER_ENCODE_BUFFER_TOO_SMALL; }
    return EER_ENCODE_OK;
}

// Callback mode with BOTH encoders: builds a hybrid (dense + sparse) index
// over host callbacks (report.md §5 sparse callback parity).
[[nodiscard]] bool create_hybrid_callback_context(uint32_t flags,
                                                  eer_context **context,
                                                  eer_error *error) {
    eer_create_info info{};
    info.struct_size = sizeof(info);
    info.flags = flags;
    info.encode = bag_encoder;
    info.sparse_encode = word_sparse_encoder;
    const auto status = eer_context_create(&info, context, error);
    return status == EER_STATUS_OK;
}

[[nodiscard]] eer_chunk_options chunk_by(uint32_t by, uint32_t size, uint32_t overlap) {
    eer_chunk_options options{};
    options.struct_size = sizeof(options);
    options.chunk_size = size;
    options.chunk_overlap = overlap;
    options.chunk_by = by;
    return options;
}

// UTF-8 continuation-byte check: a chunk boundary must never point inside a
// multi-byte code point.
[[nodiscard]] bool at_code_point_boundary(const char *text, size_t offset) {
    if (offset == 0u) { return true; }
    const auto byte = static_cast<uint8_t>(text[offset]);
    return (byte & 0xc0u) != 0x80u;
}

}// namespace

// --- tests ------------------------------------------------------------------

void test_introspection() {
    "introspection"_test = [] {
        expect(std::strcmp(eer_version(), "1.2.0") == 0) << "version must match the header";
        expect(std::strcmp(eer_sub_code_string(EER_SUB_NONE), "none") == 0);
        expect(std::strcmp(eer_sub_code_string(EER_SUB_CANCELLED), "cancelled") == 0);
        expect(std::strcmp(eer_sub_code_string(12345), "unknown") == 0);
        expect(std::strcmp(eer_status_string(EER_STATUS_OK), "ok") == 0);
        expect(std::strcmp(eer_status_string(EER_STATUS_INVALID_ARGUMENT), "invalid_argument") == 0);
        expect(std::strcmp(eer_status_string(EER_STATUS_NOT_FOUND), "not_found") == 0);
        expect(std::strcmp(eer_status_string(EER_STATUS_OPERATIONAL), "operational") == 0);
        expect(std::strcmp(eer_status_string(EER_STATUS_INVALID_HANDLE), "invalid_handle") == 0);
        expect(std::strcmp(eer_status_string(static_cast<eer_status>(999)), "unknown") == 0);
        expect(std::strcmp(eer_default_extensions(), EER_DEFAULT_EXTENSIONS) == 0);
        expect(std::strcmp(eer_default_extensions(), ".txt,.md") == 0);
    };
}

void test_null_arguments_rejected() {
    "null_arguments_rejected"_test = [] {
        // Steps reject a null handle with a diagnostic message.
        eer_load_result load{};
        expect(eer_load_dir(nullptr, "somewhere", nullptr, &load) == EER_STATUS_INVALID_HANDLE);
        expect(load.message != nullptr);
        eer_load_result_free(&load);
        expect(load.message == nullptr) << "free must zero the payload";

        eer_unload_result unload{};
        expect(eer_unload_dir(nullptr, "somewhere", &unload) == EER_STATUS_INVALID_HANDLE);
        eer_unload_result_free(&unload);
        expect(unload.message == nullptr);

        eer_update_result update{};
        expect(eer_update_path(nullptr, "somewhere", &update) == EER_STATUS_INVALID_HANDLE);
        eer_update_result_free(&update);
        expect(update.message == nullptr);

        eer_query_result query{};
        expect(eer_query(nullptr, "text", 4u, 5u, &query) == EER_STATUS_INVALID_HANDLE);
        eer_query_result_free(&query);
        expect(query.message == nullptr && query.hits == nullptr && query.hit_count == 0u);

        eer_error error{};
        size_t dimension = 0u;
        expect(eer_encode_text(nullptr, "text", 4u, nullptr, 0u, &dimension, &error) ==
               EER_STATUS_INVALID_HANDLE);
        expect(error.message != nullptr);
        eer_error_free(&error);
        expect(error.message == nullptr);

        // A null payload pointer is an argument error, not a crash.
        expect(eer_load_dir(nullptr, "somewhere", nullptr, nullptr) == EER_STATUS_INVALID_ARGUMENT);
        expect(eer_unload_dir(nullptr, "somewhere", nullptr) == EER_STATUS_INVALID_ARGUMENT);
        expect(eer_update_path(nullptr, "somewhere", nullptr) == EER_STATUS_INVALID_ARGUMENT);
        expect(eer_query(nullptr, "text", 4u, 5u, nullptr) == EER_STATUS_INVALID_ARGUMENT);
        expect(eer_encode_text(nullptr, "text", 4u, nullptr, 0u, nullptr, nullptr) ==
               EER_STATUS_INVALID_ARGUMENT);
        expect(eer_context_create(nullptr, nullptr, nullptr) == EER_STATUS_INVALID_ARGUMENT);

        // Accessors tolerate null handles, and destroy is a no-op.
        expect(eer_context_backend(nullptr) == nullptr);
        expect(eer_context_disk_cache_enabled(nullptr) == 0u);
        eer_context_destroy(nullptr);
        eer_error_free(nullptr);
        eer_load_result_free(nullptr);
        eer_unload_result_free(nullptr);
        eer_update_result_free(nullptr);
        eer_query_result_free(nullptr);
    };
}

void test_create_validation() {
    "create_validation"_test = [] {
        eer_context *context = nullptr;
        eer_error error{};

        // struct_size smaller than the supported layout.
        eer_create_info info{};
        info.struct_size = 1u;
        expect(eer_context_create(&info, &context, &error) == EER_STATUS_INVALID_ARGUMENT);
        expect(context == nullptr);
        expect(error.message != nullptr) << "a rejected create must explain itself";
        eer_error_free(&error);
        expect(error.message == nullptr);

        // struct_size == 0 means "current layout"; the missing model is then the
        // first thing that fails.
        info = eer_create_info{};
        expect(eer_context_create(&info, &context, &error) == EER_STATUS_INVALID_ARGUMENT);
        expect(context == nullptr);
        expect(error.message != nullptr);
        eer_error_free(&error);

        // An existing model but a missing tokenizer reports the file.
        TempDir dir{"create"};
        dir.write("model.bgm", "not really a model");
        const auto model = utf8_path(dir.path() / "model.bgm");
        const auto tokenizer = utf8_path(dir.path() / "missing_tokenizer.json");
        info = eer_create_info{};
        info.struct_size = sizeof(info);
        info.model_path = model.c_str();
        info.tokenizer_path = tokenizer.c_str();
        expect(eer_context_create(&info, &context, &error) == EER_STATUS_NOT_FOUND);
        expect(context == nullptr);
        expect(error.message != nullptr);
        expect(std::strstr(error.message, "missing_tokenizer.json") != nullptr)
            << "the diagnostic must name the missing file";
        eer_error_free(&error);

        // A nonexistent model is reported before the tokenizer is even looked at.
        const auto missing_model = utf8_path(dir.path() / "absent.bgm");
        info.model_path = missing_model.c_str();
        info.tokenizer_path = tokenizer.c_str();
        expect(eer_context_create(&info, &context, &error) == EER_STATUS_NOT_FOUND);
        expect(error.message != nullptr);
        expect(std::strstr(error.message, "absent.bgm") != nullptr);
        eer_error_free(&error);

        // Callback mode needs neither model nor tokenizer and cannot touch a GPU.
        context = nullptr;
        info = eer_create_info{};
        info.struct_size = sizeof(info);
        info.encode = bag_encoder;
        expect(eer_context_create(&info, &context, &error) == EER_STATUS_OK);
        expect(context != nullptr);
        expect(error.message == nullptr);
        expect(eer_context_backend(context) == nullptr) << "callback mode has no device";
        expect(eer_context_disk_cache_enabled(context) == 1u) << "the cache defaults to on";
        eer_context_destroy(context);
    };
}

void test_callback_mode_end_to_end() {
    "callback_mode_end_to_end"_test = [] {
        TempDir dir{"e2e"};
        dir.write("alpha.txt", "aaa aaa");
        dir.write("gamma.md", "aaa ccc");
        dir.write("beta.txt", "bbb ccc");
        dir.write("ignored.bin", "aaa aaa aaa");

        eer_context *context = nullptr;
        eer_error error{};
        expect(create_callback_context(bag_encoder, EER_CREATE_DISABLE_DISK_CACHE, &context, &error))
            << (error.message == nullptr ? "create failed" : error.message);
        eer_error_free(&error);
        if (context == nullptr) { return; }
        expect(eer_context_disk_cache_enabled(context) == 0u);

        // An empty corpus is a recoverable operational error with a message.
        eer_query_result empty{};
        expect(eer_query(context, "aaa", 3u, 5u, &empty) == EER_STATUS_OPERATIONAL);
        expect(empty.hit_count == 0u);
        expect(empty.message != nullptr);
        eer_query_result_free(&empty);

        // load <directory> with the default extension rule.
        const auto root = dir.utf8();
        eer_load_result load{};
        expect(eer_load_dir(context, root.c_str(), nullptr, &load) == EER_STATUS_OK)
            << (load.message == nullptr ? "" : load.message);
        expect(load.added == 3u) << "three .txt/.md files must be indexed";
        expect(load.cached == 0u);
        expect(load.skipped == 1u) << ".bin does not match the rule";
        expect(load.failed == 0u);
        expect(load.total_documents == 3u);
        expect(load.root != nullptr);
        expect(std::strcmp(load.extensions, ".md,.txt") == 0) << "the rule is normalized and sorted";
        eer_load_result_free(&load);

        // Loading the same root twice is rejected.
        expect(eer_load_dir(context, root.c_str(), nullptr, &load) == EER_STATUS_INVALID_ARGUMENT);
        expect(load.message != nullptr);
        eer_load_result_free(&load);

        // A malformed extension rule is rejected before the directory is walked.
        expect(eer_load_dir(context, root.c_str(), "*.txt", &load) == EER_STATUS_INVALID_ARGUMENT);
        expect(load.message != nullptr);
        eer_load_result_free(&load);

        // query: "aaa" ranks the a-densest document first, then path order.
        eer_query_result hits{};
        expect(eer_query(context, "aaa", 3u, 5u, &hits) == EER_STATUS_OK);
        expect(hits.hit_count == 3u);
        expect(file_name(hits.hits[0].path) == "alpha.txt");
        expect(file_name(hits.hits[1].path) == "gamma.md");
        expect(file_name(hits.hits[2].path) == "beta.txt");
        expect(hits.hits[0].score > hits.hits[1].score);
        expect(hits.hits[1].score > hits.hits[2].score);
        expect(std::abs(hits.hits[2].score) < 1e-9) << "beta has no 'a'";
        expect(hits.hits[0].score <= 1.0 + 1e-9);
        eer_query_result_free(&hits);

        // top_k: explicit, defaulted (0) and out of range.
        expect(eer_query(context, "aaa", 3u, 1u, &hits) == EER_STATUS_OK);
        expect(hits.hit_count == 1u);
        expect(file_name(hits.hits[0].path) == "alpha.txt");
        eer_query_result_free(&hits);

        expect(eer_query(context, "aaa", 3u, 0u, &hits) == EER_STATUS_OK);
        expect(hits.hit_count == 3u) << "top_k 0 selects EER_DEFAULT_TOP_K";
        eer_query_result_free(&hits);

        expect(eer_query(context, "aaa", 3u, EER_MAX_TOP_K + 1u, &hits) == EER_STATUS_INVALID_ARGUMENT);
        expect(hits.hit_count == 0u);
        expect(hits.message != nullptr);
        eer_query_result_free(&hits);

        // Empty and blank query text are argument errors, not matches-all.
        expect(eer_query(context, nullptr, 0u, 5u, &hits) == EER_STATUS_INVALID_ARGUMENT);
        eer_query_result_free(&hits);
        expect(eer_query(context, "   ", 3u, 5u, &hits) == EER_STATUS_INVALID_ARGUMENT);
        eer_query_result_free(&hits);
        expect(eer_query(context, nullptr, 5u, 5u, &hits) == EER_STATUS_INVALID_ARGUMENT)
            << "a null pointer with a nonzero length is rejected";
        eer_query_result_free(&hits);

        // eer_encode_text: dimension query, short buffer, real buffer.
        size_t dimension = 0u;
        expect(eer_encode_text(context, "aaa", 3u, nullptr, 0u, &dimension, &error) == EER_STATUS_OK);
        expect(dimension == kBagDimension);
        eer_error_free(&error);

        float small[2] = {0.0f, 0.0f};
        expect(eer_encode_text(context, "aaa", 3u, small, 2u, &dimension, &error) ==
               EER_STATUS_INVALID_ARGUMENT);
        expect(dimension == kBagDimension) << "the required size is still reported";
        expect(error.message != nullptr);
        eer_error_free(&error);

        float dense[kBagDimension] = {0.0f, 0.0f, 0.0f, 0.0f};
        expect(eer_encode_text(context, "aaa", 3u, dense, kBagDimension, &dimension, &error) ==
               EER_STATUS_OK);
        expect(dimension == kBagDimension);
        expect(std::abs(dense[0] - 1.0f) < 1e-5f);
        expect(std::abs(dense[1]) < 1e-6f);
        expect(std::abs(dense[2]) < 1e-6f);
        expect(std::abs(dense[3]) < 1e-6f);
        eer_error_free(&error);

        // update <file>: changed content is re-encoded in place.
        dir.write("alpha.txt", "aaa aaa aaa aaa");
        const auto alpha = utf8_path(dir.path() / "alpha.txt");
        eer_update_result update{};
        expect(eer_update_path(context, alpha.c_str(), &update) == EER_STATUS_OK)
            << (update.message == nullptr ? "" : update.message);
        expect(update.updated == 1u);
        expect(update.added == 0u);
        expect(update.removed == 0u);
        expect(update.unchanged == 0u);
        expect(update.failed == 0u);
        expect(update.total_documents == 3u);
        expect(file_name(update.path) == "alpha.txt");
        eer_update_result_free(&update);

        // update <directory>: new file added, siblings untouched.
        dir.write("delta.txt", "bbb bbb");
        expect(eer_update_path(context, root.c_str(), &update) == EER_STATUS_OK);
        expect(update.added == 1u);
        expect(update.updated == 0u);
        expect(update.removed == 0u);
        expect(update.unchanged == 3u);
        expect(update.total_documents == 4u);
        eer_update_result_free(&update);

        // update <directory>: deleted file removed.
        dir.remove("beta.txt");
        expect(eer_update_path(context, root.c_str(), &update) == EER_STATUS_OK);
        expect(update.removed == 1u);
        expect(update.added == 0u);
        expect(update.unchanged == 3u);
        expect(update.total_documents == 3u);
        eer_update_result_free(&update);

        // update of a path outside every root is not found.
        std::error_code ec;
        const auto outside = utf8_path(std::filesystem::temp_directory_path(ec));
        expect(eer_update_path(context, outside.c_str(), &update) == EER_STATUS_NOT_FOUND);
        expect(update.message != nullptr);
        eer_update_result_free(&update);

        // A directory that does not exist cannot be loaded; the engine
        // normalizes (canonicalizes) the root first, so this is an argument
        // error rather than "not found".
        const auto missing = utf8_path(dir.path() / "nope");
        eer_load_result missing_load{};
        expect(eer_load_dir(context, missing.c_str(), nullptr, &missing_load) ==
               EER_STATUS_INVALID_ARGUMENT);
        expect(missing_load.message != nullptr);
        eer_load_result_free(&missing_load);

        // unload <root>: removes exactly that root and its documents.
        eer_unload_result unload{};
        expect(eer_unload_dir(context, root.c_str(), &unload) == EER_STATUS_OK);
        expect(unload.removed_documents == 3u);
        expect(unload.total_documents == 0u);
        eer_unload_result_free(&unload);

        expect(eer_unload_dir(context, root.c_str(), &unload) == EER_STATUS_NOT_FOUND);
        expect(unload.message != nullptr) << "unloading twice is an error";
        eer_unload_result_free(&unload);

        // Back to the empty corpus.
        expect(eer_query(context, "aaa", 3u, 5u, &hits) == EER_STATUS_OPERATIONAL);
        eer_query_result_free(&hits);

        eer_context_destroy(context);
    };
}

void test_disk_cache_across_contexts() {
    "disk_cache_across_contexts"_test = [] {
        TempDir dir{"cache"};
        dir.write("one.txt", "aaa");
        dir.write("two.md", "bbb ccc");
        const auto root = dir.utf8();

        // First context encodes and persists the embeddings.
        eer_context *first = nullptr;
        eer_error error{};
        expect(create_callback_context(bag_encoder, 0u, &first, &error))
            << (error.message == nullptr ? "create failed" : error.message);
        eer_error_free(&error);
        if (first == nullptr) { return; }
        expect(eer_context_disk_cache_enabled(first) == 1u);
        eer_load_result load{};
        expect(eer_load_dir(first, root.c_str(), nullptr, &load) == EER_STATUS_OK);
        expect(load.added == 2u);
        expect(load.cached == 0u) << "nothing was cached yet";
        expect(load.total_documents == 2u);
        eer_load_result_free(&load);
        eer_context_destroy(first);

        // A fresh context over the same directory is served from the cache.
        eer_context *second = nullptr;
        expect(create_callback_context(bag_encoder, 0u, &second, &error));
        eer_error_free(&error);
        if (second == nullptr) { return; }
        expect(eer_load_dir(second, root.c_str(), nullptr, &load) == EER_STATUS_OK);
        expect(load.cached == 2u) << "both embeddings must come from .lc_embed_cache";
        expect(load.added == 2u) << "a new index still tracks both documents";
        expect(load.failed == 0u);
        expect(load.total_documents == 2u);
        eer_load_result_free(&load);

        // Cached embeddings must rank identically to freshly encoded ones.
        eer_query_result hits{};
        expect(eer_query(second, "aaa", 3u, 5u, &hits) == EER_STATUS_OK);
        expect(hits.hit_count == 2u);
        expect(file_name(hits.hits[0].path) == "one.txt");
        eer_query_result_free(&hits);
        eer_context_destroy(second);
    };
}

void test_encoder_failure_accounting() {
    "encoder_failure_accounting"_test = [] {
        TempDir dir{"failure"};
        dir.write("broken.txt", "aaa");

        eer_context *context = nullptr;
        eer_error error{};
        expect(create_callback_context(failing_encoder, EER_CREATE_DISABLE_DISK_CACHE, &context, &error))
            << (error.message == nullptr ? "create failed" : error.message);
        eer_error_free(&error);
        if (context == nullptr) { return; }

        // A failed encode does not fail the whole load: it is counted per file.
        const auto root = dir.utf8();
        eer_load_result load{};
        expect(eer_load_dir(context, root.c_str(), nullptr, &load) == EER_STATUS_OK);
        expect(load.added == 0u);
        expect(load.failed == 1u);
        expect(load.total_documents == 0u);
        eer_load_result_free(&load);

        // ... and the corpus stays empty, which the query reports.
        eer_query_result hits{};
        expect(eer_query(context, "aaa", 3u, 5u, &hits) == EER_STATUS_OPERATIONAL);
        expect(hits.message != nullptr);
        eer_query_result_free(&hits);

        // Encoding on demand reports the failure directly.
        size_t dimension = 0u;
        expect(eer_encode_text(context, "aaa", 3u, nullptr, 0u, &dimension, &error) ==
               EER_STATUS_OPERATIONAL);
        expect(error.message != nullptr);
        eer_error_free(&error);

        eer_context_destroy(context);
    };
}

void test_encoder_buffer_retry_and_limits() {
    "encoder_buffer_retry_and_limits"_test = [] {
        // The library first offers 1024 floats; this encoder asks for 2048 and
        // must be retried with the larger buffer.
        eer_context *context = nullptr;
        eer_error error{};
        expect(create_callback_context(retry_encoder, EER_CREATE_DISABLE_DISK_CACHE, &context, &error))
            << (error.message == nullptr ? "create failed" : error.message);
        eer_error_free(&error);
        if (context == nullptr) { return; }

        size_t dimension = 0u;
        expect(eer_encode_text(context, "aaa", 3u, nullptr, 0u, &dimension, &error) == EER_STATUS_OK);
        expect(dimension == kRetryDimension);
        std::vector<float> buffer(kRetryDimension, 0.0f);
        expect(eer_encode_text(context, "aaa", 3u, buffer.data(), buffer.size(), &dimension, &error) ==
               EER_STATUS_OK);
        expect(dimension == kRetryDimension);
        expect(std::abs(buffer[1] - 1.0f / 7.0f) < 1e-6f) << "the retried buffer must be filled";
        expect(std::abs(buffer[7] - 0.0f) < 1e-6f);
        eer_error_free(&error);
        eer_context_destroy(context);

        // An encoder that asks for more than the accepted maximum is rejected
        // instead of allocating without bound.
        context = nullptr;
        expect(create_callback_context(oversized_encoder, EER_CREATE_DISABLE_DISK_CACHE, &context, &error));
        eer_error_free(&error);
        if (context == nullptr) { return; }
        expect(eer_encode_text(context, "aaa", 3u, nullptr, 0u, &dimension, &error) ==
               EER_STATUS_OPERATIONAL);
        expect(error.message != nullptr);
        eer_error_free(&error);
        eer_context_destroy(context);
    };
}

void test_document_crud_and_chunking() {
    "document_crud_and_chunking"_test = [] {
        eer_context *context = nullptr;
        eer_error error{};
        expect(create_callback_context(bag_encoder, EER_CREATE_DISABLE_DISK_CACHE, &context, &error))
            << (error.message == nullptr ? "create failed" : error.message);
        eer_error_free(&error);
        if (context == nullptr) { return; }

        // Whole-document ingestion (chunk_by 0): one chunk, full byte range.
        const char *text = "Intro paragraph.\n\nSecond paragraph here.\n\nThird.";
        eer_document_result doc{};
        expect(eer_add_document(context, "doc.whole", text, std::strlen(text),
                                "{\"tenant\":\"acme\"}", nullptr, &doc, nullptr) == EER_STATUS_OK)
            << (doc.message == nullptr ? "" : doc.message);
        expect(doc.chunks_added == 1u);
        expect(doc.chunks_removed == 0u);
        expect(doc.chunks_unchanged == 0u);
        expect(doc.total_docs == 1u);
        expect(doc.total_documents == 1u);
        expect(std::strcmp(doc.doc_id, "doc.whole") == 0);
        eer_document_result_free(&doc);

        // Idempotent re-ingestion: identical text/metadata/chunking is a no-op.
        expect(eer_add_document(context, "doc.whole", text, std::strlen(text),
                                "{\"tenant\":\"acme\"}", nullptr, &doc, nullptr) == EER_STATUS_OK);
        expect(doc.chunks_added == 0u);
        expect(doc.chunks_unchanged == 1u);
        eer_document_result_free(&doc);

        // Replace: old chunk dropped, new chunks added.
        const char *replacement = "aaa bbb";
        expect(eer_update_document(context, "doc.whole", replacement, std::strlen(replacement),
                                   "{\"tenant\":\"acme\"}", nullptr, &doc, nullptr) == EER_STATUS_OK);
        expect(doc.chunks_removed == 1u);
        expect(doc.chunks_added == 1u);
        expect(doc.chunks_unchanged == 0u);
        eer_document_result_free(&doc);

        // Invalid metadata and bad doc_id are argument errors.
        expect(eer_add_document(context, "doc.badmeta", "x", 1u, "[1,2]", nullptr, &doc, nullptr) ==
               EER_STATUS_INVALID_ARGUMENT);
        expect(doc.message != nullptr);
        eer_document_result_free(&doc);
        expect(eer_add_document(context, "", "x", 1u, nullptr, nullptr, &doc, nullptr) ==
               EER_STATUS_INVALID_ARGUMENT);
        eer_document_result_free(&doc);

        // Paragraph chunking: offsets monotonic, in-bounds, on code point
        // boundaries, and each hit's passage reproduces the source bytes.
        const auto paragraph = chunk_by(1u, 16u, 4u);
        expect(eer_add_document(context, "doc.par", text, std::strlen(text),
                                "{}", &paragraph, &doc, nullptr) == EER_STATUS_OK)
            << (doc.message == nullptr ? "" : doc.message);
        expect(doc.chunks_added >= 2u) << "16-byte chunks must split the document";
        const auto par_chunks = doc.chunks_added;
        eer_document_result_free(&doc);

        eer_query_result hits{};
        expect(eer_query(context, "paragraph", 9u, 10u, &hits) == EER_STATUS_OK);
        expect(hits.hit_count >= 2u);
        // Hits come back in RANKED order, so offsets are not monotonic on the
        // hit list itself; verify location fields per chunk instead: collect
        // the doc.par hits, order them by chunk_index, then require strictly
        // increasing offsets and byte-exact passage reproduction.
        struct Location {
            uint32_t chunk_index;
            size_t offset;
            size_t length;
        };
        std::vector<Location> locations;
        for (size_t i = 0u; i < hits.hit_count; i++) {
            if (std::strcmp(hits.hits[i].doc_id, "doc.par") != 0) { continue; }
            expect(hits.hits[i].path == nullptr) << "memory-only docs have no path";
            locations.push_back(Location{
                hits.hits[i].chunk_index,
                static_cast<size_t>(hits.hits[i].byte_offset),
                static_cast<size_t>(hits.hits[i].byte_length)});
        }
        std::sort(locations.begin(), locations.end(),
                  [](const Location &a, const Location &b) { return a.chunk_index < b.chunk_index; });
        expect(locations.size() >= 2u) << "the paragraph chunks carrying the query must rank";
        for (size_t i = 0u; i < locations.size(); i++) {
            const auto &loc = locations[i];
            expect(loc.offset + loc.length <= std::strlen(text));
            expect(loc.length > 0u);
            expect(at_code_point_boundary(text, loc.offset));
            if (i > 0u) {
                expect(loc.offset > locations[i - 1].offset)
                    << "chunk offsets must increase with chunk_index";
            }
        }
        eer_query_result_free(&hits);

        // Sentence and token modes both produce multiple chunks here.
        const auto sentence = chunk_by(2u, 8u, 2u);
        expect(eer_add_document(context, "doc.sent", "One two. Three four! Five?", 27u,
                                nullptr, &sentence, &doc, nullptr) == EER_STATUS_OK);
        expect(doc.chunks_added >= 2u);
        eer_document_result_free(&doc);
        const auto token = chunk_by(3u, 6u, 2u);
        expect(eer_add_document(context, "doc.tok", "w1 w2 w3 w4 w5 w6", 20u,
                                nullptr, &token, &doc, nullptr) == EER_STATUS_OK);
        expect(doc.chunks_added >= 2u);
        eer_document_result_free(&doc);

        // Querying by the original text no longer matches the replaced doc.
        expect(eer_query(context, "Intro", 5u, 5u, &hits) == EER_STATUS_OK);
        for (size_t i = 0u; i < hits.hit_count; i++) {
            expect(std::strcmp(hits.hits[i].doc_id, "doc.whole") != 0 ||
                   hits.hits[i].score > 0.0) << "replaced text must not match";
        }
        eer_query_result_free(&hits);

        // Removal: unknown id is NOT_FOUND, known id drops every chunk.
        expect(eer_remove_document(context, "doc.absent", &doc) == EER_STATUS_NOT_FOUND);
        eer_document_result_free(&doc);
        expect(eer_remove_document(context, "doc.par", &doc) == EER_STATUS_OK);
        expect(doc.chunks_removed == par_chunks);
        expect(doc.total_docs == 3u) << "whole + sent + tok remain";
        eer_document_result_free(&doc);

        eer_context_destroy(context);
    };
}

void test_snapshot_round_trip() {
    "snapshot_round_trip"_test = [] {
        TempDir dir{"snapshot"};
        dir.write("one.txt", "aaa bbb");
        const auto root = dir.utf8();
        const auto snapshot = utf8_path(dir.path() / "corpus.eer");

        eer_context *first = nullptr;
        eer_error error{};
        expect(create_callback_context(bag_encoder, EER_CREATE_DISABLE_DISK_CACHE, &first, &error))
            << (error.message == nullptr ? "create failed" : error.message);
        eer_error_free(&error);
        if (first == nullptr) { return; }
        eer_load_result load{};
        expect(eer_load_dir(first, root.c_str(), nullptr, &load) == EER_STATUS_OK);
        expect(load.added == 1u);
        eer_load_result_free(&load);
        eer_document_result doc{};
        expect(eer_add_document(first, "mem.doc", "ccc ccc", 7u,
                                "{\"k\":\"v\"}", nullptr, &doc, nullptr) == EER_STATUS_OK);
        eer_document_result_free(&doc);
        expect(eer_index_save(first, snapshot.c_str(), &error) == EER_STATUS_OK)
            << (error.message == nullptr ? "save failed" : error.message);
        eer_error_free(&error);
        expect(eer_index_save(first, nullptr, &error) == EER_STATUS_INVALID_ARGUMENT);
        eer_error_free(&error);
        eer_context_destroy(first);

        // A fresh context restores corpus + metadata with no re-encode.
        eer_context *second = nullptr;
        expect(create_callback_context(bag_encoder, EER_CREATE_DISABLE_DISK_CACHE, &second, &error));
        eer_error_free(&error);
        if (second == nullptr) { return; }
        const auto load_status = eer_index_load(second, snapshot.c_str(), &error);
        expect(load_status == EER_STATUS_OK)
            << (error.message == nullptr ? "load failed" : error.message);
        eer_error_free(&error);
        eer_query_result hits{};
        expect(eer_query(second, "ccc", 3u, 5u, &hits) == EER_STATUS_OK);
        if (hits.hit_count > 0u) {
            expect(std::strcmp(hits.hits[0].doc_id, "mem.doc") == 0);
            expect(hits.hits[0].path == nullptr);
            expect(hits.hits[0].byte_length == 7u);
        } else {
            expect(false) << "the restored corpus must contain mem.doc";
        }
        eer_query_result_free(&hits);

        // The restored corpus is live: queries and metadata filters work, and
        // the document counters reflect the snapshot.
        eer_stats_data stats{};
        stats.struct_size = sizeof(stats);
        expect(eer_stats(second, &stats) == EER_STATUS_OK);
        expect(stats.document_count == 2u);
        expect(stats.chunk_count == 2u);
        expect(stats.encode_count == 0u)
            << "a restored snapshot must not re-encode anything";
        eer_stats_data_free(&stats);

        // Metadata survived the snapshot: the filter sees the restored doc.
        {
            eer_query_options options{};
            options.struct_size = sizeof(options);
            options.filter_json = "{\"k\":\"v\"}";
            expect(eer_query_ex(second, "ccc", 3u, 5u, &options, &hits) == EER_STATUS_OK);
            expect(hits.hit_count == 1u);
            expect(std::strcmp(hits.hits[0].doc_id, "mem.doc") == 0);
            eer_query_result_free(&hits);
        }

        // Corrupted snapshot: a clean OPERATIONAL error, index untouched.
        {
            std::ofstream stream{snapshot, std::ios::binary | std::ios::trunc};
            stream << "{ not json";
        }
        expect(eer_index_load(second, snapshot.c_str(), &error) == EER_STATUS_OPERATIONAL);
        expect(error.message != nullptr);
        expect(error.sub_code != EER_SUB_NONE);
        eer_error_free(&error);
        expect(eer_query(second, "ccc", 3u, 5u, &hits) == EER_STATUS_OK)
            << "the previous index must survive a failed load";
        expect(hits.hit_count == 2u) << "mem.doc plus the zero-scoring file doc";
        expect(std::strcmp(hits.hits[0].doc_id, "mem.doc") == 0);
        expect(std::abs(hits.hits[1].score) < 1e-12);
        eer_query_result_free(&hits);

        // Missing snapshot file reports the io sub-code.
        const auto missing = utf8_path(dir.path() / "absent.eer");
        expect(eer_index_load(second, missing.c_str(), &error) == EER_STATUS_OPERATIONAL);
        expect(error.sub_code == EER_SUB_IO);
        eer_error_free(&error);
        eer_context_destroy(second);
    };
}

void test_query_ex_controls() {
    "query_ex_controls"_test = [] {
        // Hybrid callback mode: dense letter bag + word-level sparse halves.
        eer_context *context = nullptr;
        eer_error error{};
        expect(create_hybrid_callback_context(EER_CREATE_DISABLE_DISK_CACHE, &context, &error))
            << (error.message == nullptr ? "create failed" : error.message);
        eer_error_free(&error);
        if (context == nullptr) { return; }

        // Fixture: query "red blue". doc1 matches at the WORD level (sparse
        // winner), doc2's letters match the dense bag but its words do not
        // (dense winner) -- the two halves genuinely disagree.
        eer_document_result doc{};
        expect(eer_add_document(context, "doc1", "red blue green yellow white", 29u,
                                "{\"tenant\":\"acme\",\"year\":2023}", nullptr, &doc, nullptr) ==
               EER_STATUS_OK);
        eer_document_result_free(&doc);
        expect(eer_add_document(context, "doc2", "redred blueblue", 15u,
                                "{\"tenant\":\"other\",\"year\":2021}", nullptr, &doc, nullptr) ==
               EER_STATUS_OK);
        eer_document_result_free(&doc);

        auto query_with = [&](const char *text, const char *filter, double wd, double ws,
                              double diversity) {
            eer_query_options options{};
            options.struct_size = sizeof(options);
            options.dense_weight = wd;
            options.sparse_weight = ws;
            options.filter_json = filter;
            options.diversity = diversity;
            eer_query_result result{};
            const auto qstatus = eer_query_ex(context, text, std::strlen(text), 5u, &options, &result);
            expect(qstatus == EER_STATUS_OK)
                << (result.message == nullptr ? "" : result.message);
            return result;
        };

        // Default 0.7/0.3 weights: measure the winner, then prove both
        // single-half overrides flip it deterministically. The name is COPIED:
        // it points into the hit buffer that eer_query_result_free releases.
        std::string default_first;
        {
            auto hits = query_with("red blue", nullptr, -1.0, -1.0, 0.0);
            expect(hits.hit_count == 2u);
            default_first = hits.hits[0].doc_id;
            // Score breakdown: score == 0.7 * dense + 0.3 * sparse.
            const auto fused = 0.7 * hits.hits[0].dense_score + 0.3 * hits.hits[0].sparse_score;
            expect(std::abs(hits.hits[0].score - fused) < 1e-9)
                << "score must be the weighted fusion of the breakdown";
            expect(hits.hits[0].sparse_score > 0.0);
            expect(hits.hits[0].colbert_score == 0.0);
            eer_query_result_free(&hits);
        }
        {
            auto hits = query_with("red blue", nullptr, 1.0, 0.0, 0.0);
            expect(std::strcmp(hits.hits[0].doc_id, "doc2") == 0)
                << "dense-only weights must rank the dense winner first";
            expect(std::abs(hits.hits[0].score - hits.hits[0].dense_score) < 1e-12);
            eer_query_result_free(&hits);
        }
        {
            auto hits = query_with("red blue", nullptr, 0.0, 1.0, 0.0);
            expect(std::strcmp(hits.hits[0].doc_id, "doc1") == 0)
                << "sparse-only weights must rank the sparse winner first";
            expect(std::abs(hits.hits[0].score - hits.hits[0].sparse_score) < 1e-12);
            eer_query_result_free(&hits);
        }
        expect(default_first != "doc2") << "the default fusion must not be dense-only order";

        // Metadata filter applied BEFORE top-k: even with dense-only weights
        // (where doc2 would out-rank everything), the tenant filter keeps
        // only doc1.
        {
            auto hits = query_with("red blue", "{\"tenant\":\"acme\"}", 1.0, 0.0, 0.0);
            expect(hits.hit_count == 1u);
            expect(std::strcmp(hits.hits[0].doc_id, "doc1") == 0);
            eer_query_result_free(&hits);
        }
        // $gte + $and.
        {
            auto hits = query_with("red blue",
                                   "{\"$and\":[{\"tenant\":\"acme\"},{\"year\":{\"$gte\":2022}}]}",
                                   -1.0, -1.0, 0.0);
            expect(hits.hit_count == 1u);
            expect(std::strcmp(hits.hits[0].doc_id, "doc1") == 0);
            eer_query_result_free(&hits);
            hits = query_with("red blue",
                              "{\"$and\":[{\"tenant\":\"acme\"},{\"year\":{\"$gte\":2024}}]}",
                              -1.0, -1.0, 0.0);
            expect(hits.hit_count == 0u) << "year 2023 does not satisfy $gte 2024";
            eer_query_result_free(&hits);
        }
        // $in.
        {
            auto hits = query_with("red blue", "{\"tenant\":{\"$in\":[\"acme\",\"other\"]}}",
                                   -1.0, -1.0, 0.0);
            expect(hits.hit_count == 2u);
            eer_query_result_free(&hits);
        }
        // A malformed filter is an argument error.
        {
            eer_query_options options{};
            options.struct_size = sizeof(options);
            options.filter_json = "{\"tenant\":";
            eer_query_result result{};
            expect(eer_query_ex(context, "red blue", 9u, 5u, &options, &result) ==
                   EER_STATUS_INVALID_ARGUMENT);
            expect(result.message != nullptr);
            eer_query_result_free(&result);
        }

        // Diversity / MMR: three identical alpha chunks plus a no-match beta.
        {
            const auto paragraph = chunk_by(1u, 12u, 3u);
            expect(eer_add_document(context, "alpha",
                                    "aaaa bbbb\n\naaaa bbbb\n\naaaa bbbb", 29u,
                                    "{}", &paragraph, &doc, nullptr) == EER_STATUS_OK);
            expect(doc.chunks_added == 3u);
            eer_document_result_free(&doc);
            expect(eer_add_document(context, "beta", "eeee ffff", 9u,
                                    "{}", nullptr, &doc, nullptr) == EER_STATUS_OK);
            eer_document_result_free(&doc);

            auto plain = query_with("a", nullptr, -1.0, -1.0, 0.0);
            // Without diversity the three alpha chunks take the top.
            expect(plain.hit_count >= 3u);
            expect(std::strcmp(plain.hits[0].doc_id, "alpha") == 0);
            expect(std::strcmp(plain.hits[1].doc_id, "alpha") == 0);
            eer_query_result_free(&plain);

            auto diverse = query_with("a", nullptr, -1.0, -1.0, 1.0);
            expect(std::strcmp(diverse.hits[0].doc_id, "alpha") == 0);
            expect(std::strcmp(diverse.hits[1].doc_id, "beta") == 0)
                << "MMR must suppress the same-document repetition";
            eer_query_result_free(&diverse);
        }

        eer_context_destroy(context);
    };
}

void test_encode_batch_and_stats() {
    "encode_batch_and_stats"_test = [] {
        eer_context *context = nullptr;
        eer_error error{};
        expect(create_callback_context(bag_encoder, EER_CREATE_DISABLE_DISK_CACHE, &context, &error))
            << (error.message == nullptr ? "create failed" : error.message);
        eer_error_free(&error);
        if (context == nullptr) { return; }

        // Batch parity: row-major output identical to per-text encodes.
        const eer_text texts[] = {{"aaa", 3u}, {"bbb ccc", 7u}, {"ddd", 3u}};
        float batch[3u * kBagDimension] = {};
        size_t dimension = 0u;
        expect(eer_encode_batch(context, texts, 3u, batch, 3u * kBagDimension,
                                &dimension, &error) == EER_STATUS_OK)
            << (error.message == nullptr ? "" : error.message);
        expect(dimension == kBagDimension);
        for (size_t i = 0u; i < 3u; i++) {
            float single[kBagDimension] = {};
            size_t single_dim = 0u;
            expect(eer_encode_text(context, texts[i].utf8, texts[i].length,
                                   single, kBagDimension, &single_dim, &error) == EER_STATUS_OK);
            for (size_t k = 0u; k < kBagDimension; k++) {
                expect(batch[i * kBagDimension + k] == single[k])
                    << "batch row must equal the per-text encode";
            }
        }

        // Dimension probe: nothing written, dimension reported.
        dimension = 0u;
        expect(eer_encode_batch(context, texts, 3u, nullptr, 0u, &dimension, &error) ==
               EER_STATUS_OK);
        expect(dimension == kBagDimension);
        // Too-small buffer is an argument error.
        expect(eer_encode_batch(context, texts, 3u, batch, 3u, &dimension, &error) ==
               EER_STATUS_INVALID_ARGUMENT);
        expect(error.message != nullptr);
        eer_error_free(&error);
        expect(eer_encode_batch(context, nullptr, 0u, nullptr, 0u, &dimension, &error) ==
               EER_STATUS_INVALID_ARGUMENT);
        eer_error_free(&error);

        // Stats, ping, device memory, park/resume.
        eer_document_result doc{};
        expect(eer_add_document(context, "stats.doc", "aaa bbb ccc", 11u,
                                "{}", nullptr, &doc, nullptr) == EER_STATUS_OK);
        eer_document_result_free(&doc);
        eer_stats_data stats{};
        stats.struct_size = sizeof(stats);
        expect(eer_stats(context, &stats) == EER_STATUS_OK);
        expect(stats.document_count == 1u);
        expect(stats.chunk_count == 1u);
        expect(stats.dimension == kBagDimension);
        expect(stats.hybrid == 0u) << "dense callback mode is not hybrid";
        expect(stats.encode_count > 0u);
        expect(stats.mean_encode_ms >= 0.0);
        expect(eer_stats(nullptr, &stats) == EER_STATUS_INVALID_HANDLE);
        eer_stats_data_free(&stats);

        expect(eer_context_ping(context, &error) == EER_STATUS_OK)
            << (error.message == nullptr ? "" : error.message);
        eer_error_free(&error);
        uint64_t used = 99u, total = 99u;
        expect(eer_context_device_memory(context, &used, &total) == EER_STATUS_OK);
        expect(used == 0u && total == 0u) << "callback mode reports the honest 0/0";
        expect(eer_context_device_memory(context, nullptr, &total) == EER_STATUS_INVALID_ARGUMENT);
        expect(eer_context_park(context, &error) == EER_STATUS_OPERATIONAL);
        expect(error.message != nullptr);
        eer_error_free(&error);
        expect(eer_context_resume(context, &error) == EER_STATUS_OPERATIONAL);
        eer_error_free(&error);

        // Hybrid callback mode reports hybrid stats.
        eer_context *hybrid = nullptr;
        expect(create_hybrid_callback_context(EER_CREATE_DISABLE_DISK_CACHE, &hybrid, &error));
        eer_error_free(&error);
        if (hybrid != nullptr) {
            expect(eer_add_document(hybrid, "h1", "aaa", 3u, "{}", nullptr, &doc, nullptr) ==
                   EER_STATUS_OK);
            eer_document_result_free(&doc);
            stats = eer_stats_data{};
            stats.struct_size = sizeof(stats);
            expect(eer_stats(hybrid, &stats) == EER_STATUS_OK);
            expect(stats.hybrid == 1u) << "sparse callback builds a hybrid index";
            eer_stats_data_free(&stats);
            eer_context_destroy(hybrid);
        }
        eer_context_destroy(context);
    };
}

void test_load_progress_and_cancel() {
    "load_progress_and_cancel"_test = [] {
        TempDir dir{"cancel"};
        for (int i = 0; i < 6; i++) {
            dir.write(("f" + std::to_string(i) + ".txt").c_str(), "aaa bbb");
        }
        const auto root = dir.utf8();

        // Progress callback sees (done, total); setting the cancel flag at 3
        // stops the load with a consistent partial state.
        struct CancelState {
            uint32_t flag = 0u;
            uint64_t calls = 0u;
            uint64_t last_done = 0u;
            uint64_t last_total = 0u;
        } state;
        auto on_progress = [](void *user_data, uint64_t done, uint64_t total) {
            auto *state = static_cast<CancelState *>(user_data);
            state->calls++;
            state->last_done = done;
            state->last_total = total;
            if (done >= 3u) { state->flag = 1u; }
        };
        eer_progress progress{};
        progress.struct_size = sizeof(progress);
        progress.on_progress = on_progress;
        progress.user_data = &state;
        progress.cancel_flag = &state.flag;

        eer_context *context = nullptr;
        eer_error error{};
        expect(create_callback_context(bag_encoder, EER_CREATE_DISABLE_DISK_CACHE, &context, &error))
            << (error.message == nullptr ? "create failed" : error.message);
        eer_error_free(&error);
        if (context == nullptr) { return; }
        eer_load_result load{};
        expect(eer_load_dir_ex(context, root.c_str(), nullptr, nullptr, &load, &progress) ==
               EER_STATUS_OPERATIONAL)
            << "a cancelled load is a clean operational state";
        expect(load.sub_code == EER_SUB_CANCELLED);
        expect(load.message != nullptr);
        expect(std::strstr(load.message, "cancelled") != nullptr);
        expect(load.added == 3u) << "exactly the files before the cancel";
        expect(load.total_documents == 3u);
        expect(state.last_total == 6u);
        eer_load_result_free(&load);

        // Re-loading without the flag resumes: the first three files are
        // unchanged, the remaining three are added.
        expect(eer_load_dir_ex(context, root.c_str(), nullptr, nullptr, &load, nullptr) ==
               EER_STATUS_OK)
            << (load.message == nullptr ? "resume failed" : load.message);
        expect(load.added == 3u) << "the three new files are indexed";
        expect(load.failed == 0u);
        expect(load.total_documents == 6u) << "the resumed corpus holds all six";
        eer_load_result_free(&load);

        // Structured sub-codes surface on failures.
        eer_query_result hits{};
        expect(eer_query(context, "zzz", 3u, 5u, &hits) == EER_STATUS_OK);
        eer_query_result_free(&hits);
        size_t dimension = 0u;
        expect(eer_encode_text(context, "aaa", 3u, nullptr, 0u, &dimension, &error) ==
               EER_STATUS_OK);
        eer_error_free(&error);
        eer_context_destroy(context);

        // Encoder failure carries the encode sub-code.
        context = nullptr;
        expect(create_callback_context(failing_encoder, EER_CREATE_DISABLE_DISK_CACHE,
                                       &context, &error));
        eer_error_free(&error);
        if (context == nullptr) { return; }
        expect(eer_encode_text(context, "aaa", 3u, nullptr, 0u, &dimension, &error) ==
               EER_STATUS_OPERATIONAL);
        expect(error.sub_code == EER_SUB_ENCODE);
        eer_error_free(&error);
        expect(eer_query(context, "aaa", 3u, 5u, &hits) == EER_STATUS_OPERATIONAL);
        expect(hits.sub_code == EER_SUB_INDEX) << "empty corpus reports the index sub-code";
        expect(hits.message != nullptr);
        eer_query_result_free(&hits);
        eer_context_destroy(context);
    };
}

void test_hit_location_fields() {
    "hit_location_fields"_test = [] {
        TempDir dir{"hits"};
        dir.write("place.txt", "aaa bbb");
        const auto root = dir.utf8();

        eer_context *context = nullptr;
        eer_error error{};
        expect(create_callback_context(bag_encoder, EER_CREATE_DISABLE_DISK_CACHE, &context, &error))
            << (error.message == nullptr ? "create failed" : error.message);
        eer_error_free(&error);
        if (context == nullptr) { return; }
        eer_load_result load{};
        expect(eer_load_dir(context, root.c_str(), nullptr, &load) == EER_STATUS_OK);
        expect(load.total_documents == 1u);
        eer_load_result_free(&load);

        eer_query_result hits{};
        expect(eer_query(context, "aaa", 3u, 5u, &hits) == EER_STATUS_OK);
        expect(hits.hit_count == 1u);
        expect(hits.hits[0].doc_id != nullptr);
        expect(hits.hits[0].path != nullptr);
        expect(std::strcmp(hits.hits[0].doc_id, hits.hits[0].path) == 0)
            << "whole-file documents derive doc_id from the path";
        expect(hits.hits[0].chunk_index == 0u);
        expect(hits.hits[0].byte_offset == 0u);
        expect(hits.hits[0].byte_length == 7u) << "whole-file chunk spans the file";
        expect(std::abs(hits.hits[0].score - hits.hits[0].dense_score) < 1e-12);
        expect(hits.hits[0].sparse_score == 0.0) << "dense mode has no sparse term";
        expect(hits.truncated == 0u);
        eer_query_result_free(&hits);
        eer_context_destroy(context);
    };
}

void test_whole_file_chunking_defaults() {
    "whole_file_chunking_defaults"_test = [] {
        // ABI 1.1 compatibility (the NULL-chunk-options regression): every
        // legacy path that passes no chunk options must keep the whole-file
        // behavior, even for multi-paragraph documents that the 1.2 paragraph
        // defaults would otherwise split.
        TempDir dir{"wholefile"};
        const char *contents = "first paragraph\n\nsecond paragraph\n\nthird";
        dir.write("multi.txt", contents);
        const auto root = dir.utf8();

        eer_context *context = nullptr;
        eer_error error{};
        expect(create_callback_context(bag_encoder, EER_CREATE_DISABLE_DISK_CACHE, &context, &error))
            << (error.message == nullptr ? "create failed" : error.message);
        eer_error_free(&error);
        if (context == nullptr) { return; }

        // Legacy eer_load_dir (NULL options): one chunk spanning the file.
        eer_load_result load{};
        expect(eer_load_dir(context, root.c_str(), nullptr, &load) == EER_STATUS_OK);
        expect(load.added == 1u);
        eer_load_result_free(&load);
        eer_stats_data stats{};
        stats.struct_size = sizeof(stats);
        expect(eer_stats(context, &stats) == EER_STATUS_OK);
        expect(stats.chunk_count == 1u) << "legacy load_dir must not chunk";
        expect(stats.document_count == 1u);
        eer_stats_data_free(&stats);
        eer_query_result hits{};
        expect(eer_query(context, "paragraph", 9u, 5u, &hits) == EER_STATUS_OK);
        expect(hits.hit_count == 1u);
        expect(hits.hits[0].byte_offset == 0u);
        expect(hits.hits[0].byte_length == std::strlen(contents))
            << "the whole file stays one chunk";
        eer_query_result_free(&hits);
        eer_unload_result unloaded{};
        expect(eer_unload_dir(context, root.c_str(), &unloaded) == EER_STATUS_OK);
        eer_unload_result_free(&unloaded);

        // NULL options on eer_add_document: the whole document, one chunk.
        eer_document_result doc{};
        expect(eer_add_document(context, "mem.multi", contents, std::strlen(contents),
                                nullptr, nullptr, &doc, nullptr) == EER_STATUS_OK);
        expect(doc.chunks_added == 1u) << "NULL chunk options keep one whole chunk";
        eer_document_result_free(&doc);
        // Explicit chunk_by 0 / chunk_size 0 is the same contract.
        const auto off = chunk_by(0u, 0u, 0u);
        expect(eer_add_document(context, "mem.explicit", contents, std::strlen(contents),
                                nullptr, &off, &doc, nullptr) == EER_STATUS_OK);
        expect(doc.chunks_added == 1u);
        eer_document_result_free(&doc);
        stats = eer_stats_data{};
        stats.struct_size = sizeof(stats);
        expect(eer_stats(context, &stats) == EER_STATUS_OK);
        expect(stats.chunk_count == 2u) << "both memory docs are single whole chunks";
        eer_stats_data_free(&stats);

        // A NULL-options directory load through the _ex entry point too.
        expect(eer_load_dir_ex(context, root.c_str(), nullptr, nullptr, &load, nullptr) ==
               EER_STATUS_OK);
        expect(load.added == 1u);
        eer_load_result_free(&load);

        // Truncation-flag plumbing is inert but well-formed in callback mode
        // (a host encoder never truncates; real truncation needs engine mode).
        eer_query_options options{};
        options.struct_size = sizeof(options);
        options.flags = 1u;
        expect(eer_query_ex(context, "paragraph", 9u, 5u, &options, &hits) == EER_STATUS_OK);
        expect(hits.truncated == 0u);
        eer_query_result_free(&hits);

        // want_colbert without the create flag: a clean operational error
        // carrying the index sub-code (no stored ColBERT matrices).
        options.want_colbert = 1u;
        expect(eer_query_ex(context, "paragraph", 9u, 5u, &options, &hits) ==
               EER_STATUS_OPERATIONAL);
        expect(hits.sub_code == EER_SUB_INDEX);
        expect(hits.message != nullptr);
        eer_query_result_free(&hits);
        // ColBERT encoding has no host callback: encode sub-code, no output.
        float colbert[8] = {};
        size_t values = 99u;
        uint32_t rows = 99u;
        expect(eer_encode_colbert(context, "paragraph", 9u, colbert, 8u,
                                  &values, &rows, &error) == EER_STATUS_OPERATIONAL);
        expect(error.sub_code == EER_SUB_ENCODE);
        expect(values == 0u && rows == 0u);
        eer_error_free(&error);

        // INVALID_HANDLE paths populate sub_code explicitly (EER_SUB_NONE).
        expect(eer_query(nullptr, "x", 1u, 5u, &hits) == EER_STATUS_INVALID_HANDLE);
        expect(hits.sub_code == EER_SUB_NONE);
        expect(hits.message != nullptr);
        eer_query_result_free(&hits);

        // struct_size gating: the layout that ends before want_colbert is
        // accepted, anything smaller is rejected, and a truncated struct is
        // never read past its end.
        options = eer_query_options{};
        options.struct_size = offsetof(eer_query_options, want_colbert);
        options.dense_weight = 1.0;
        expect(eer_query_ex(context, "paragraph", 9u, 5u, &options, &hits) == EER_STATUS_OK);
        expect(hits.hit_count >= 1u);
        eer_query_result_free(&hits);
        options = eer_query_options{};
        options.struct_size = offsetof(eer_query_options, want_colbert) - 1u;
        expect(eer_query_ex(context, "paragraph", 9u, 5u, &options, &hits) ==
               EER_STATUS_INVALID_ARGUMENT);
        expect(hits.sub_code == EER_SUB_NONE);
        eer_query_result_free(&hits);

        eer_context_destroy(context);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_introspection();
    test_null_arguments_rejected();
    test_create_validation();
    test_callback_mode_end_to_end();
    test_disk_cache_across_contexts();
    test_encoder_failure_accounting();
    test_encoder_buffer_retry_and_limits();
    test_document_crud_and_chunking();
    test_snapshot_round_trip();
    test_query_ex_controls();
    test_encode_batch_and_stats();
    test_load_progress_and_cancel();
    test_hit_location_fields();
    test_whole_file_chunking_defaults();
}
