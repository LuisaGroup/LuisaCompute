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
//
// Run: xmake run test_embed_retrieval_api

#include "ut/ut.hpp"

#include "example_embed_retrieval_lib.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
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

}// namespace

// --- tests ------------------------------------------------------------------

void test_introspection() {
    "introspection"_test = [] {
        expect(std::strcmp(eer_version(), "1.0.0") == 0) << "version must match the header";
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
}
