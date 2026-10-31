// Test for embedding/embed/retrieval_index.h — host-only whole-file retrieval.
// This test covers extension filtering, recursive discovery, deterministic dense
// ranking, root ownership, exact-content incremental updates, deletion handling,
// and per-file encode-failure rollback. It also covers phase H1 hybrid retrieval:
// 0.7/0.3 dense+sparse fusion, sparse canonicalization (sort/merge/normalize),
// empty and malformed sparse vectors, and the hybrid disk-cache sidecar
// (<md5>.sparse.bin) round trip, invalidation, and dense-mode compatibility.
// No model, device, or backend is used.
// Run: xmake run test_embed_retrieval_index
#include "ut/ut.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/mathematics.h>
#include <luisa/core/stl/filesystem.h>

#include "embedding/embed/retrieval_disk_cache.h"
#include "embedding/embed/retrieval_index.h"

using namespace luisa;
using namespace luisa::embed;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<uint64_t> sequence{0u};
        const auto ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::error_code ec;
        const auto base = luisa::filesystem::temp_directory_path(ec);
        if (ec) {
            LUISA_ERROR("cannot locate temporary directory");
        }
        _path = base / luisa::format("lc_retrieval_index_{}_{}", ticks, sequence.fetch_add(1u));
        luisa::filesystem::create_directories(_path, ec);
        if (ec) {
            LUISA_ERROR("cannot create temporary test directory");
        }
    }
    ~TempDirectory() noexcept {
        std::error_code ec;
        luisa::filesystem::remove_all(_path, ec);
    }
    [[nodiscard]] const luisa::filesystem::path &path() const noexcept { return _path; }

private:
    luisa::filesystem::path _path;
};

void make_directories(const luisa::filesystem::path &path) {
    std::error_code ec;
    luisa::filesystem::create_directories(path, ec);
    if (ec) {
        LUISA_ERROR("cannot create test directory '{}'", path.string());
    }
}

void write_file(const luisa::filesystem::path &path, luisa::string_view contents) {
    make_directories(path.parent_path());
    std::ofstream stream{path, std::ios::binary | std::ios::trunc};
    if (!stream.is_open()) {
        LUISA_ERROR("cannot create test file '{}'", path.string());
    }
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!stream) {
        LUISA_ERROR("cannot write test file '{}'", path.string());
    }
}

void remove_file(const luisa::filesystem::path &path) {
    std::error_code ec;
    const auto removed = luisa::filesystem::remove(path, ec);
    expect(removed && !ec) << "test file deletion should succeed";
}

[[nodiscard]] luisa::string utf8_path(const luisa::filesystem::path &path) {
    std::error_code ec;
    const auto canonical = luisa::filesystem::canonical(path, ec);
    const auto &selected = ec ? path : canonical;
    const auto utf8 = selected.generic_u8string();
    return {reinterpret_cast<const char *>(utf8.data()), utf8.size()};
}

struct FakeEncoder {
    size_t calls{};
    luisa::vector<luisa::string> inputs;
    luisa::string fail_on;

    [[nodiscard]] RetrievalEncodeCallback callback() {
        return [this](luisa::string_view text) -> luisa::vector<float> {
            calls++;
            inputs.emplace_back(text);
            if (!fail_on.empty() && text.find(fail_on) != luisa::string_view::npos) {
                return {};
            }
            if (text.find("beta") != luisa::string_view::npos) {
                return {0.0f, 1.0f, 0.0f};
            }
            if (text.find("gamma") != luisa::string_view::npos) {
                return {0.0f, 0.0f, 1.0f};
            }
            return {1.0f, 0.0f, 0.0f};
        };
    }
};

// Hybrid fake encoder (phase H1): deterministic dense + sparse outputs keyed by
// a keyword in the text, so the tests can predict the fused score exactly.
//
//   "query"    -> dense {1, 0, 0}           sparse {(10, 0.6), (20, 0.8)} (unit norm)
//   "alpha"    -> dense {1, 0, 0}           sparse {(10, 1)}
//   "beta"     -> dense {1, 0, 0}           sparse {(20, 1)}
//   "gamma"    -> dense {0, 1, 0}           sparse {(20, 1)}
//   "delta"    -> dense {0.5, 0.5, 0}       sparse {} (no lexical evidence)
//   "dupquery" -> dense {1, 0, 0}           sparse {(5, 2), (5, 2), (9, 0)}
//   "dupdoc"   -> dense {1, 0, 0}           sparse {(5, 1), (5, 3), (9, 0)}
//   otherwise  -> dense {0, 0, 1}           sparse {}
//
// With `bad_weights` set (1 = negative, 2 = NaN, 3 = +Inf) any text containing
// "bad" returns that malformed sparse weight instead.
struct FakeHybridEncoder {
    size_t calls{};
    luisa::vector<luisa::string> inputs;
    int bad_weights{};

    [[nodiscard]] RetrievalEmbedding embedding_for(luisa::string_view text) const {
        const auto contains = [text](luisa::string_view keyword) {
            return text.find(keyword) != luisa::string_view::npos;
        };
        RetrievalEmbedding embedding;
        if (bad_weights != 0 && contains("bad")) {
            const auto weight = bad_weights == 1   ? -1.0f
                                : bad_weights == 2 ? std::numeric_limits<float>::quiet_NaN()
                                                   : std::numeric_limits<float>::infinity();
            embedding.dense = {1.0f, 0.0f, 0.0f};
            embedding.sparse = {{42u, 1.0f}, {7u, weight}};
            return embedding;
        }
        if (contains("dupquery")) {
            embedding.dense = {1.0f, 0.0f, 0.0f};
            embedding.sparse = {{5u, 2.0f}, {9u, 0.0f}, {5u, 2.0f}};
            return embedding;
        }
        if (contains("dupdoc")) {
            embedding.dense = {1.0f, 0.0f, 0.0f};
            embedding.sparse = {{5u, 1.0f}, {9u, 0.0f}, {5u, 3.0f}};
            return embedding;
        }
        if (contains("query")) {
            embedding.dense = {1.0f, 0.0f, 0.0f};
            embedding.sparse = {{10u, 0.6f}, {20u, 0.8f}};
            return embedding;
        }
        if (contains("alpha")) {
            embedding.dense = {1.0f, 0.0f, 0.0f};
            embedding.sparse = {{10u, 1.0f}};
            return embedding;
        }
        if (contains("beta")) {
            embedding.dense = {1.0f, 0.0f, 0.0f};
            embedding.sparse = {{20u, 1.0f}};
            return embedding;
        }
        if (contains("gamma")) {
            embedding.dense = {0.0f, 1.0f, 0.0f};
            embedding.sparse = {{20u, 1.0f}};
            return embedding;
        }
        if (contains("delta")) {
            embedding.dense = {0.5f, 0.5f, 0.0f};
            return embedding;// empty sparse vector: valid, no lexical evidence
        }
        embedding.dense = {0.0f, 0.0f, 1.0f};
        return embedding;
    }

    [[nodiscard]] RetrievalHybridEncodeCallback callback() {
        return [this](luisa::string_view text) -> RetrievalEmbedding {
            calls++;
            inputs.emplace_back(text);
            return embedding_for(text);
        };
    }

    // Dense half only, for the dense-vs-hybrid comparisons and for the
    // dense-only cache metadata tests.
    [[nodiscard]] RetrievalEncodeCallback dense_callback() {
        return [this](luisa::string_view text) -> luisa::vector<float> {
            calls++;
            inputs.emplace_back(text);
            return embedding_for(text).dense;
        };
    }
};

[[nodiscard]] luisa::string read_text_file(const luisa::filesystem::path &path) {
    std::ifstream stream{path, std::ios::binary | std::ios::ate};
    if (!stream.is_open()) {
        expect(false) << "cache artifact should be readable";
        return {};
    }
    const auto end = stream.tellg();
    if (end < 0) {
        return {};
    }
    luisa::string text;
    text.resize(static_cast<size_t>(end));
    stream.seekg(0, std::ios::beg);
    if (!text.empty()) {
        stream.read(text.data(), static_cast<std::streamsize>(text.size()));
    }
    return text;
}

// Lists the dense (`<md5>.bin`, 36 chars) or sparse (`<md5>.sparse.bin`, 43
// chars) cache artifacts, sorted by file name for deterministic comparison.
[[nodiscard]] luisa::vector<luisa::filesystem::path> list_cache_files(
    const luisa::filesystem::path &cache_dir, bool sparse) {
    luisa::vector<luisa::filesystem::path> files;
    std::error_code ec;
    for (luisa::filesystem::directory_iterator it{cache_dir}, end; it != end; it.increment(ec)) {
        if (ec) {
            break;
        }
        const auto path = it->path();
        const auto name = path.filename().string();
        const bool is_sparse = name.ends_with(".sparse.bin");
        if (is_sparse != sparse) {
            continue;
        }
        if (name.size() == (is_sparse ? 43u : 36u)) {
            files.push_back(path);
        }
    }
    luisa::sort(files.begin(), files.end(), [](const auto &a, const auto &b) {
        return a.filename().string() < b.filename().string();
    });
    return files;
}

[[nodiscard]] size_t file_bytes(const luisa::filesystem::path &path) {
    std::error_code ec;
    const auto bytes = luisa::filesystem::file_size(path, ec);
    expect(!ec) << "cache artifact size should be readable";
    return ec ? 0u : static_cast<size_t>(bytes);
}

void reg_extension_discovery() {
    "retrieval_index_extension_discovery"_test = [] {
        TempDirectory temporary;
        const auto root = temporary.path() / "root with spaces";
        write_file(root / "a.TXT", "alpha héllo 世界");
        write_file(root / "b.md", "beta");
        write_file(root / "c.rst", "gamma");
        write_file(root / "ignore.bin", "alpha");
        write_file(root / "nested" / "d.TxT", "gamma");

        FakeEncoder encoder;
        RetrievalIndex index{encoder.callback()};
        auto loaded = index.load(root);
        expect(loaded.status.ok());
        expect(loaded.added == 3u);
        expect(loaded.skipped == 2u);
        expect(loaded.failed == 0u);
        expect(loaded.total_documents == 3u);
        expect(encoder.calls == 3u);

        auto unloaded = index.unload(root);
        expect(unloaded.status.ok());
        expect(unloaded.removed_documents == 3u);
        const luisa::vector<luisa::string> custom{".RST"};
        loaded = index.load(root, luisa::span<const luisa::string>{custom.data(), custom.size()});
        expect(loaded.status.ok());
        expect(loaded.added == 1u);
        expect(loaded.skipped == 4u);
        expect(loaded.extensions.size() == 1u && loaded.extensions[0] == ".rst");

        luisa::vector<luisa::string> parsed;
        auto status = RetrievalIndex::parse_extension_list("TXT,.Md", parsed);
        expect(status.ok());
        expect(parsed.size() == 2u && parsed[0] == ".md" && parsed[1] == ".txt");
        status = RetrievalIndex::parse_extension_list(".txt,", parsed);
        expect(!status.ok() && status.code == RetrievalErrorCode::invalid_argument);
        status = RetrievalIndex::parse_extension_list(".txt,..md", parsed);
        expect(!status.ok());
        status = RetrievalIndex::parse_extension_list(".txt,.m/d", parsed);
        expect(!status.ok());
    };
}

void reg_empty_corpus_and_validation() {
    "retrieval_index_empty_corpus_and_validation"_test = [] {
        TempDirectory temporary;
        FakeEncoder encoder;
        RetrievalIndex index{encoder.callback()};
        auto result = index.query("alpha");
        expect(!result.status.ok());
        expect(result.status.code == RetrievalErrorCode::operational);
        expect(encoder.calls == 0u);
        expect(!index.query("   \t").status.ok());
        expect(!index.query("alpha", 0u).status.ok());
        expect(!index.query("alpha", kRetrievalMaxTopK + 1u).status.ok());

        auto loaded = index.load(temporary.path());
        expect(loaded.status.ok() && loaded.added == 0u);
        expect(index.root_count() == 1u && index.document_count() == 0u);
        result = index.query("alpha");
        expect(!result.status.ok() && encoder.calls == 0u);

        write_file(temporary.path() / "invalid.txt", luisa::string_view{"\xc0\xaf", 2u});
        auto updated = index.update(temporary.path());
        expect(updated.status.ok());
        expect(updated.failed == 1u && updated.total_documents == 0u);

        std::error_code ec;
        luisa::filesystem::remove_all(temporary.path(), ec);
        expect(!ec);
        updated = index.update(temporary.path());
        expect(updated.status.ok() && updated.failed == 0u && updated.total_documents == 0u);
    };
}

void reg_deterministic_ranking() {
    "retrieval_index_deterministic_ranking"_test = [] {
        TempDirectory temporary;
        const auto a = temporary.path() / "a.txt";
        const auto b = temporary.path() / "b.txt";
        const auto c = temporary.path() / "c.txt";
        write_file(a, "alpha");
        write_file(b, "beta");
        write_file(c, "alpha");
        FakeEncoder encoder;
        RetrievalIndex index{encoder.callback()};
        expect(index.load(temporary.path()).status.ok());

        const auto result = index.query("alpha", 3u);
        expect(result.status.ok());
        expect(result.hits.size() == 3u);
        if (result.hits.size() == 3u) {
            expect(result.hits[0].path == utf8_path(a));
            expect(result.hits[1].path == utf8_path(c));
            expect(result.hits[2].path == utf8_path(b));
            expect(luisa::abs(result.hits[0].score - 1.0) < 1e-12);
            expect(luisa::abs(result.hits[1].score - 1.0) < 1e-12);
            expect(luisa::abs(result.hits[2].score) < 1e-12);
        }
        expect(index.query("alpha", 2u).hits.size() == 2u);
    };
}

void reg_root_ownership_and_unload() {
    "retrieval_index_root_ownership_and_unload"_test = [] {
        TempDirectory temporary;
        const auto docs = temporary.path() / "docs";
        const auto docs2 = temporary.path() / "docs2";
        write_file(docs / "nested" / "a.txt", "alpha");
        write_file(docs2 / "b.txt", "beta");
        FakeEncoder encoder;
        RetrievalIndex index{encoder.callback()};
        expect(index.load(docs).status.ok());
        expect(index.load(docs2).status.ok()) << "path-prefix siblings are not overlapping roots";
        expect(!index.load(docs).status.ok()) << "duplicate root must fail";
        expect(!index.load(docs / "nested").status.ok()) << "nested root must fail";
        expect(!index.load(temporary.path()).status.ok()) << "parent of loaded roots must fail";
        expect(index.document_count() == 2u);

        auto result = index.unload(docs / "nested");
        expect(!result.status.ok() && result.status.code == RetrievalErrorCode::not_found);
        result = index.unload(docs);
        expect(result.status.ok());
        expect(result.removed_documents == 1u && result.total_documents == 1u);
        expect(index.root_count() == 1u);
        const auto query = index.query("beta", 5u);
        expect(query.status.ok() && query.hits.size() == 1u);
        if (query.hits.size() == 1u) {
            expect(query.hits[0].path == utf8_path(docs2 / "b.txt"));
        }
    };
}

void reg_single_file_update() {
    "retrieval_index_single_file_update"_test = [] {
        TempDirectory temporary;
        const auto root = temporary.path() / "root";
        const auto a = root / "a.txt";
        const auto b = root / "b.txt";
        write_file(a, "alpha");
        write_file(b, "gamma");
        write_file(root / "ignore.bin", "alpha");
        FakeEncoder encoder;
        RetrievalIndex index{encoder.callback()};
        expect(index.load(root).status.ok());
        const auto after_load = encoder.calls;
        expect(after_load == 2u);

        auto updated = index.update(a);
        expect(updated.status.ok() && updated.unchanged == 1u);
        expect(encoder.calls == after_load) << "byte-identical files must not be re-encoded";

        write_file(a, "beta");
        updated = index.update(a);
        expect(updated.status.ok() && updated.updated == 1u);
        expect(encoder.calls == after_load + 1u);
        const auto query = index.query("beta", 1u);
        expect(query.status.ok() && query.hits.size() == 1u);
        if (query.hits.size() == 1u) {
            expect(query.hits[0].path == utf8_path(a));
            expect(luisa::abs(query.hits[0].score - 1.0) < 1e-12);
        }

        const auto before_new = encoder.calls;
        const auto c = root / "new file.txt";
        write_file(c, "alpha");
        updated = index.update(c);
        expect(updated.status.ok() && updated.added == 1u);
        expect(encoder.calls == before_new + 1u);
        remove_file(b);
        updated = index.update(b);
        expect(updated.status.ok() && updated.removed == 1u);
        expect(updated.total_documents == 2u);
        expect(!index.update(root / "missing.txt").status.ok());
        expect(!index.update(root / "ignore.bin").status.ok());
        expect(!index.update(temporary.path() / "outside.txt").status.ok());
    };
}

void reg_subdirectory_update() {
    "retrieval_index_subdirectory_update"_test = [] {
        TempDirectory temporary;
        const auto root = temporary.path() / "root";
        const auto left = root / "left";
        const auto right = root / "right";
        const auto changed = left / "changed.txt";
        write_file(changed, "alpha");
        write_file(left / "deleted.txt", "gamma");
        write_file(right / "sibling.txt", "beta");
        FakeEncoder encoder;
        RetrievalIndex index{encoder.callback()};
        expect(index.load(root).status.ok());
        const auto after_load = encoder.calls;

        write_file(changed, "beta");
        write_file(left / "added.txt", "alpha");
        remove_file(left / "deleted.txt");
        auto updated = index.update(left);
        expect(updated.status.ok());
        expect(updated.added == 1u && updated.updated == 1u);
        expect(updated.removed == 1u && updated.unchanged == 0u);
        expect(updated.failed == 0u && updated.total_documents == 3u);
        expect(encoder.calls == after_load + 2u) << "sibling documents must not be re-encoded";

        const auto before_unchanged = encoder.calls;
        updated = index.update(left);
        expect(updated.status.ok() && updated.unchanged == 2u);
        expect(encoder.calls == before_unchanged);
        updated = index.update(right);
        expect(updated.status.ok() && updated.unchanged == 1u);
        expect(encoder.calls == before_unchanged);

        std::error_code ec;
        luisa::filesystem::remove_all(left, ec);
        expect(!ec);
        updated = index.update(left);
        expect(updated.status.ok() && updated.removed == 2u && updated.failed == 0u);
        expect(updated.total_documents == 1u);
        expect(encoder.calls == before_unchanged);
    };
}

void reg_failed_reencode_preserves_document() {
    "retrieval_index_failed_reencode_preserves_document"_test = [] {
        TempDirectory temporary;
        const auto file = temporary.path() / "a.txt";
        write_file(file, "alpha");
        FakeEncoder encoder;
        RetrievalIndex index{encoder.callback()};
        expect(index.load(temporary.path()).status.ok());

        encoder.fail_on = "beta";
        write_file(file, "beta");
        auto updated = index.update(file);
        expect(updated.status.ok());
        expect(updated.failed == 1u && updated.updated == 0u);
        auto query = index.query("alpha", 1u);
        expect(query.status.ok() && query.hits.size() == 1u);
        if (query.hits.size() == 1u) {
            expect(query.hits[0].path == utf8_path(file));
            expect(luisa::abs(query.hits[0].score - 1.0) < 1e-12);
        }

        encoder.fail_on.clear();
        updated = index.update(file);
        expect(updated.status.ok() && updated.updated == 1u);
        query = index.query("beta", 1u);
        expect(query.status.ok() && query.hits.size() == 1u);
        if (query.hits.size() == 1u) {
            expect(luisa::abs(query.hits[0].score - 1.0) < 1e-12);
        }
    };
}

void reg_disk_cache_roundtrip() {
    "retrieval_index_disk_cache_roundtrip"_test = [] {
        TempDirectory temporary;
        const auto root = temporary.path() / "root";
        write_file(root / "a.txt", "alpha");
        write_file(root / "nested" / "b.md", "beta");
        write_file(root / "c.txt", "gamma");

        // First import encodes everything and persists the cache artifacts.
        DiskEmbedCache cache{nullptr};// host-only fallback reads (no device)
        FakeEncoder encoder;
        {
            RetrievalIndex index{encoder.callback()};
            index.set_embed_cache(&cache);
            const auto loaded = index.load(root);
            expect(loaded.status.ok());
            expect(loaded.added == 3u && loaded.cached == 0u && loaded.failed == 0u);
            expect(encoder.calls == 3u);
        }
        std::error_code ec;
        expect(luisa::filesystem::is_regular_file(root / ".lc_embed_cache.json", ec) && !ec);
        const auto cache_dir = root / ".lc_embed_cache";
        size_t binaries = 0u;
        for (luisa::filesystem::directory_iterator it{cache_dir}, end; it != end; it.increment(ec)) {
            expect(!ec);
            const auto name = it->path().filename().string();
            expect(name.size() == 36u && name.ends_with(".bin"));
            binaries++;
        }
        expect(binaries == 3u);

        // Second import of the same directory: every file is served from the
        // disk cache and the encoder is never invoked.
        {
            DiskEmbedCache second_cache{nullptr};
            FakeEncoder second_encoder;
            RetrievalIndex index{second_encoder.callback()};
            index.set_embed_cache(&second_cache);
            const auto loaded = index.load(root);
            expect(loaded.status.ok());
            expect(loaded.added == 3u && loaded.cached == 3u && loaded.failed == 0u);
            expect(second_encoder.calls == 0u) << "unchanged files must not be re-encoded";
            const auto query = index.query("alpha", 3u);
            expect(query.status.ok() && query.hits.size() == 3u);
            if (!query.hits.empty()) {
                expect(query.hits.front().path == utf8_path(root / "a.txt"));
                expect(luisa::abs(query.hits.front().score - 1.0) < 1e-12);
            }
        }

        // A changed file re-encodes; unchanged siblings still hit the cache.
        write_file(root / "c.txt", "gamma v2");
        {
            DiskEmbedCache third_cache{nullptr};
            FakeEncoder third_encoder;
            RetrievalIndex index{third_encoder.callback()};
            index.set_embed_cache(&third_cache);
            const auto loaded = index.load(root);
            expect(loaded.status.ok());
            expect(loaded.added == 3u && loaded.cached == 2u && loaded.failed == 0u);
            expect(third_encoder.calls == 1u);
        }

        // The update path keeps the cache consistent: a file modified after
        // import is re-encoded once and its cache entry is refreshed, so the
        // next cold load serves the whole directory from disk again.
        {
            DiskEmbedCache fourth_cache{nullptr};
            FakeEncoder fourth_encoder;
            RetrievalIndex index{fourth_encoder.callback()};
            index.set_embed_cache(&fourth_cache);
            expect(index.load(root).status.ok());
            expect(fourth_encoder.calls == 0u);
            write_file(root / "c.txt", "gamma v4");
            const auto updated = index.update(root);
            expect(updated.status.ok());
            expect(updated.updated == 1u && updated.unchanged == 2u);
            expect(updated.cached == 0u && updated.failed == 0u);
            expect(fourth_encoder.calls == 1u);
        }
        {
            DiskEmbedCache fifth_cache{nullptr};
            FakeEncoder fifth_encoder;
            RetrievalIndex index{fifth_encoder.callback()};
            index.set_embed_cache(&fifth_cache);
            const auto loaded = index.load(root);
            expect(loaded.status.ok());
            expect(loaded.added == 3u && loaded.cached == 3u && loaded.failed == 0u);
            expect(fifth_encoder.calls == 0u);
        }
    };
}

void reg_disk_cache_invalidation() {
    "retrieval_index_disk_cache_invalidation"_test = [] {
        TempDirectory temporary;
        const auto root = temporary.path() / "root";
        const auto file = root / "a.txt";
        write_file(file, "alpha");
        write_file(root / "b.txt", "beta");

        DiskEmbedCache cache{nullptr};
        FakeEncoder encoder;
        {
            RetrievalIndex index{encoder.callback()};
            index.set_embed_cache(&cache);
            expect(index.load(root).status.ok());
            expect(encoder.calls == 2u);
        }

        // Deleting the binary cache files turns every hit into a re-encode.
        luisa::vector<luisa::filesystem::path> binaries;
        std::error_code ec;
        for (luisa::filesystem::directory_iterator it{root / ".lc_embed_cache"}, end;
             it != end; it.increment(ec)) {
            expect(!ec);
            binaries.push_back(it->path());
        }
        expect(binaries.size() == 2u);
        for (const auto &binary : binaries) {
            expect(luisa::filesystem::remove(binary, ec) && !ec);
        }
        {
            DiskEmbedCache second_cache{nullptr};
            FakeEncoder second_encoder;
            RetrievalIndex index{second_encoder.callback()};
            index.set_embed_cache(&second_cache);
            const auto loaded = index.load(root);
            expect(loaded.status.ok() && loaded.cached == 0u && loaded.added == 2u);
            expect(second_encoder.calls == 2u);
        }

        // A mismatched embedding dimension invalidates the entry: once the
        // index dimension is established by a fresh encode, cache entries with
        // a different geometry must not poison the index.
        {
            DiskEmbedCache third_cache{nullptr};
            size_t calls = 0u;
            RetrievalIndex index{[&calls](luisa::string_view text) {
                calls++;
                const float marker = text.find("beta") != luisa::string_view::npos ? 1.0f : 0.0f;
                return luisa::vector<float>{1.0f, marker};// 2-dim vs cached 3-dim
            }};
            index.set_embed_cache(&third_cache);
            write_file(file, "alpha v2");// forces the first encode at 2-dim
            const auto loaded = index.load(root);
            expect(loaded.status.ok() && loaded.cached == 0u && loaded.added == 2u);
            expect(calls == 2u) << "stale-dimension cache entries must be re-encoded";
            const auto query = index.query("beta", 2u);
            expect(query.status.ok() && query.hits.size() == 2u);
            if (!query.hits.empty()) {
                expect(query.hits.front().path == utf8_path(root / "b.txt"));
            }
        }

        // Corrupt meta JSON is ignored gracefully (full re-encode, no crash).
        write_file(root / ".lc_embed_cache.json", "{ not json");
        {
            DiskEmbedCache fourth_cache{nullptr};
            FakeEncoder fourth_encoder;
            RetrievalIndex index{fourth_encoder.callback()};
            index.set_embed_cache(&fourth_cache);
            const auto loaded = index.load(root);
            expect(loaded.status.ok() && loaded.cached == 0u && loaded.added == 2u);
            expect(fourth_encoder.calls == 2u);
        }
    };
}

void reg_embedding_validation() {
    "retrieval_index_embedding_validation"_test = [] {
        TempDirectory temporary;
        write_file(temporary.path() / "a.txt", "alpha");
        size_t calls = 0u;
        RetrievalIndex index{[&calls](luisa::string_view) {
            calls++;
            return luisa::vector<float>{1.0f, std::numeric_limits<float>::quiet_NaN()};
        }};
        auto loaded = index.load(temporary.path());
        expect(loaded.status.ok() && loaded.failed == 1u && loaded.added == 0u);
        expect(index.document_count() == 0u && calls == 1u);

        TempDirectory dimensional;
        write_file(dimensional.path() / "a.txt", "alpha");
        write_file(dimensional.path() / "b.txt", "beta");
        RetrievalIndex dimensional_index{[](luisa::string_view text) {
            if (text == "beta") {
                return luisa::vector<float>{1.0f, 0.0f, 0.0f};
            }
            return luisa::vector<float>{1.0f, 0.0f};
        }};
        loaded = dimensional_index.load(dimensional.path());
        expect(loaded.status.ok() && loaded.added == 1u && loaded.failed == 1u);
        expect(loaded.total_documents == 1u);

        luisa::filesystem::path converted;
        auto status = RetrievalIndex::path_from_utf8(luisa::string_view{"\xc0\xaf", 2u}, converted);
        expect(!status.ok() && status.code == RetrievalErrorCode::invalid_argument);
    };
}

// ---------------------------------------------------------------------------
// Phase H1: hybrid dense + sparse retrieval
// ---------------------------------------------------------------------------

void reg_hybrid_fusion_ranking() {
    "retrieval_index_hybrid_fusion_ranking"_test = [] {
        TempDirectory temporary;
        const auto root = temporary.path() / "root";
        write_file(root / "alpha.txt", "alpha");
        write_file(root / "beta.txt", "beta");
        write_file(root / "gamma.txt", "gamma");
        write_file(root / "delta.txt", "delta");
        write_file(root / "dup_a.txt", "alpha");// ties with alpha.txt and dup_b.txt
        write_file(root / "dup_b.txt", "alpha");

        FakeHybridEncoder encoder;
        RetrievalIndex index{encoder.callback()};
        expect(index.hybrid()) << "the hybrid callback must select hybrid mode";
        const auto loaded = index.load(root);
        expect(loaded.status.ok());
        expect(loaded.added == 6u && loaded.failed == 0u);
        expect(encoder.calls == 6u);

        // Query dense {1,0,0} / sparse {(10, 0.6), (20, 0.8)}.
        // Predicted fused scores: beta 0.7*1 + 0.3*0.8 = 0.94,
        // alpha 0.7*1 + 0.3*0.6 = 0.88, delta 0.7*0.5 = 0.35,
        // gamma 0.7*0 + 0.3*0.8 = 0.24.
        const auto result = index.query("query", 6u);
        expect(result.status.ok());
        expect(result.hits.size() == 6u);
        if (result.hits.size() == 6u) {
            const auto alpha = utf8_path(root / "alpha.txt");
            const auto beta = utf8_path(root / "beta.txt");
            // Sparse evidence moves beta above alpha; the three alpha documents
            // keep their deterministic path tie-break.
            expect(result.hits[0].path == beta);
            expect(result.hits[1].path == alpha);
            expect(result.hits[2].path == utf8_path(root / "dup_a.txt"));
            expect(result.hits[3].path == utf8_path(root / "dup_b.txt"));
            expect(result.hits[4].path == utf8_path(root / "delta.txt"));
            expect(result.hits[5].path == utf8_path(root / "gamma.txt"));
            expect(luisa::abs(result.hits[0].score - 0.94) < 1e-6);
            expect(luisa::abs(result.hits[1].score - 0.88) < 1e-6);
            expect(luisa::abs(result.hits[2].score - 0.88) < 1e-6);
            expect(luisa::abs(result.hits[3].score - 0.88) < 1e-6);
            // Dense-only document: the sparse term contributes exactly zero.
            expect(luisa::abs(result.hits[4].score -
                              kRetrievalDenseWeight * 0.5) < 1e-6);
            // Sparse-only document: it has no dense overlap with the query.
            expect(luisa::abs(result.hits[5].score -
                              kRetrievalSparseWeight * 0.8) < 1e-6);
        }
        const auto top1 = index.query("query", 1u);
        expect(top1.status.ok() && top1.hits.size() == 1u);
        if (!top1.hits.empty()) {
            expect(top1.hits.front().path == utf8_path(root / "beta.txt"));
        }

        // The same encoder restricted to its dense half must reproduce the old
        // dense-only ranking (alpha first: the 1.0 tie breaks on the path).
        FakeHybridEncoder dense_probe;
        RetrievalIndex dense_index{dense_probe.dense_callback()};
        expect(!dense_index.hybrid());
        expect(dense_index.load(root).status.ok());
        const auto dense_result = dense_index.query("query", 6u);
        expect(dense_result.status.ok() && dense_result.hits.size() == 6u);
        if (dense_result.hits.size() == 6u) {
            expect(dense_result.hits[0].path == utf8_path(root / "alpha.txt"));
            expect(dense_result.hits[1].path == utf8_path(root / "beta.txt"));
            expect(luisa::abs(dense_result.hits[0].score - 1.0) < 1e-12);
            expect(luisa::abs(dense_result.hits[1].score - 1.0) < 1e-12);
            expect(luisa::abs(dense_result.hits[4].score - 0.5) < 1e-12);
            expect(luisa::abs(dense_result.hits[5].score) < 1e-12);
            expect(dense_result.hits[0].path != result.hits[0].path)
                << "hybrid fusion must change the ranking of this corpus";
        }
    };
}

void reg_hybrid_sparse_canonicalization() {
    "retrieval_index_hybrid_sparse_canonicalization"_test = [] {
        // Sorting, duplicate merging (double sum), zero dropping, normalization.
        luisa::vector<std::pair<uint32_t, float>> input{
            {7u, 1.0f}, {5u, 1.0f}, {7u, 1.0f}, {11u, 0.0f}};
        luisa::vector<std::pair<uint32_t, float>> canonical;
        auto status = RetrievalIndex::canonicalize_sparse(
            luisa::span<const std::pair<uint32_t, float>>{input.data(), input.size()}, canonical);
        expect(status.ok());
        expect(canonical.size() == 2u);
        if (canonical.size() == 2u) {
            expect(canonical[0].first == 5u && canonical[1].first == 7u)
                << "canonical vectors are sorted by token id";
            // {(5,1),(7,2)} / sqrt(5)
            expect(luisa::abs(canonical[0].second - 0.4472136f) < 1e-6f);
            expect(luisa::abs(canonical[1].second - 0.8944272f) < 1e-6f);
        }
        luisa::vector<std::pair<uint32_t, float>> again;
        status = RetrievalIndex::canonicalize_sparse(
            luisa::span<const std::pair<uint32_t, float>>{canonical.data(), canonical.size()}, again);
        expect(status.ok() && again.size() == canonical.size());
        for (size_t i = 0u; i < again.size() && i < canonical.size(); i++) {
            expect(again[i].first == canonical[i].first) << "canonicalization is idempotent";
            expect(luisa::abs(again[i].second - canonical[i].second) < 1e-6f);
        }

        // Empty and all-zero vectors are valid with norm zero.
        luisa::vector<std::pair<uint32_t, float>> nothing;
        status = RetrievalIndex::canonicalize_sparse(
            luisa::span<const std::pair<uint32_t, float>>{}, nothing);
        expect(status.ok() && nothing.empty());
        luisa::vector<std::pair<uint32_t, float>> zeros{{1u, 0.0f}, {2u, 0.0f}};
        luisa::vector<std::pair<uint32_t, float>> zero_canonical;
        status = RetrievalIndex::canonicalize_sparse(
            luisa::span<const std::pair<uint32_t, float>>{zeros.data(), zeros.size()},
            zero_canonical);
        expect(status.ok() && zero_canonical.empty());

        // Malformed weights are recoverable errors.
        const auto rejects = [&canonical](luisa::vector<std::pair<uint32_t, float>> raw) {
            const auto rejected = RetrievalIndex::canonicalize_sparse(
                luisa::span<const std::pair<uint32_t, float>>{raw.data(), raw.size()},
                canonical);
            return !rejected.ok() && rejected.code == RetrievalErrorCode::operational;
        };
        expect(rejects({{1u, -0.5f}})) << "negative weights are rejected";
        expect(rejects({{1u, std::numeric_limits<float>::quiet_NaN()}}))
            << "NaN weights are rejected";
        expect(rejects({{1u, std::numeric_limits<float>::infinity()}}))
            << "infinite weights are rejected";
        expect(rejects({{1u, -std::numeric_limits<float>::infinity()}}))
            << "negative infinite weights are rejected";
        const auto huge = std::numeric_limits<float>::max();
        expect(rejects({{1u, huge}, {1u, huge}}))
            << "duplicate sums that overflow the weight range are rejected";

        // Duplicate tokens collapse end to end: {(5,1),(5,3)} and {(5,2),(5,2)}
        // are the same canonical vector, so the fusion hits its 1.0 ceiling.
        TempDirectory temporary;
        const auto root = temporary.path() / "root";
        write_file(root / "dupdoc.txt", "dupdoc");
        write_file(root / "other.txt", "gamma");
        FakeHybridEncoder encoder;
        RetrievalIndex index{encoder.callback()};
        expect(index.load(root).status.ok());
        const auto result = index.query("dupquery", 1u);
        expect(result.status.ok() && result.hits.size() == 1u);
        if (!result.hits.empty()) {
            expect(result.hits.front().path == utf8_path(root / "dupdoc.txt"));
            expect(luisa::abs(result.hits.front().score - 1.0) < 1e-6);
        }
    };
}

void reg_hybrid_sparse_failures_and_rollback() {
    "retrieval_index_hybrid_sparse_failures"_test = [] {
        // Malformed sparse weights are recoverable: the offending file fails and
        // every sibling still indexes.
        TempDirectory temporary;
        const auto root = temporary.path() / "root";
        const auto file = root / "good.txt";
        write_file(file, "alpha");
        write_file(root / "bad.txt", "bad negative");
        FakeHybridEncoder encoder;
        encoder.bad_weights = 1;// negative weight
        RetrievalIndex index{encoder.callback()};
        const auto loaded = index.load(root);
        expect(loaded.status.ok());
        expect(loaded.added == 1u && loaded.failed == 1u && loaded.total_documents == 1u);
        expect(index.document_count() == 1u);

        // A failed re-encode must keep the previously committed document whole
        // (dense and sparse): the transaction is per file.
        auto query = index.query("query", 2u);
        expect(query.status.ok() && query.hits.size() == 1u);
        if (!query.hits.empty()) {
            expect(luisa::abs(query.hits.front().score - 0.88) < 1e-6);
        }
        write_file(file, "bad negative v2");
        auto updated = index.update(file);
        expect(updated.status.ok());
        expect(updated.failed == 1u && updated.updated == 0u && updated.added == 0u);
        expect(updated.total_documents == 1u);
        query = index.query("query", 2u);
        expect(query.status.ok() && query.hits.size() == 1u);
        if (!query.hits.empty()) {
            expect(query.hits.front().path == utf8_path(file));
            expect(luisa::abs(query.hits.front().score - 0.88) < 1e-6)
                << "a failed update must preserve the old dense + sparse embedding";
        }

        // NaN and +Inf weights fail the same way and leave the document intact.
        const int bad_modes[] = {2, 3};
        for (const auto mode : bad_modes) {
            encoder.bad_weights = mode;
            write_file(file, mode == 2 ? "bad nan" : "bad inf");
            updated = index.update(file);
            expect(updated.status.ok());
            expect(updated.failed == 1u && updated.updated == 0u);
            expect(index.document_count() == 1u);
            query = index.query("query", 2u);
            expect(query.status.ok() && query.hits.size() == 1u);
            if (!query.hits.empty()) {
                expect(luisa::abs(query.hits.front().score - 0.88) < 1e-6);
            }
        }

        // Recovery: a clean encoder commits the new content.
        encoder.bad_weights = 0;
        write_file(file, "beta");
        updated = index.update(file);
        expect(updated.status.ok() && updated.updated == 1u && updated.failed == 0u);
        query = index.query("beta", 1u);
        expect(query.status.ok() && query.hits.size() == 1u);
        if (!query.hits.empty()) {
            expect(query.hits.front().path == utf8_path(file));
            expect(luisa::abs(query.hits.front().score - 1.0) < 1e-6);
        }
    };
}

void reg_hybrid_disk_cache_roundtrip() {
    "retrieval_index_hybrid_disk_cache_roundtrip"_test = [] {
        TempDirectory temporary;
        const auto root = temporary.path() / "root";
        write_file(root / "a.txt", "alpha");
        write_file(root / "nested" / "b.md", "beta");
        write_file(root / "c.txt", "gamma");
        write_file(root / "d.txt", "delta");// empty sparse vector -> zero-byte sidecar

        luisa::vector<RetrievalHit> first_hits;
        {
            DiskEmbedCache cache{nullptr};
            FakeHybridEncoder encoder;
            RetrievalIndex index{encoder.callback()};
            index.set_embed_cache(&cache);
            const auto loaded = index.load(root);
            expect(loaded.status.ok());
            expect(loaded.added == 4u && loaded.cached == 0u && loaded.failed == 0u);
            expect(encoder.calls == 4u);
            const auto result = index.query("query", 4u);
            expect(result.status.ok() && result.hits.size() == 4u);
            first_hits = result.hits;
        }

        const auto cache_dir = root / ".lc_embed_cache";
        const auto dense_files = list_cache_files(cache_dir, false);
        const auto sparse_files = list_cache_files(cache_dir, true);
        expect(dense_files.size() == 4u);
        expect(sparse_files.size() == 4u) << "one sparse sidecar per hybrid encode";
        size_t empty_sidecars = 0u;
        size_t sidecar_records = 0u;
        for (const auto &path : sparse_files) {
            const auto bytes = file_bytes(path);
            if (bytes == 0u) {
                empty_sidecars++;
            } else {
                expect(bytes % 8u == 0u) << "sidecars hold 8-byte records";
                sidecar_records += bytes / 8u;
            }
        }
        expect(empty_sidecars == 1u) << "the empty sparse vector has a zero-byte sidecar";
        expect(sidecar_records == 3u) << "alpha/beta/gamma carry one pair each";
        const auto meta = read_text_file(root / ".lc_embed_cache.json");
        expect(meta.find("sparse_count") != luisa::string::npos)
            << "hybrid metadata records the sidecar pair counts";

        // Second load: every file is a hybrid hit and the encoder is untouched.
        {
            DiskEmbedCache cache{nullptr};
            FakeHybridEncoder encoder;
            RetrievalIndex index{encoder.callback()};
            index.set_embed_cache(&cache);
            const auto loaded = index.load(root);
            expect(loaded.status.ok());
            expect(loaded.added == 4u && loaded.cached == 4u && loaded.failed == 0u);
            expect(encoder.calls == 0u) << "hybrid hits require zero encoder calls";
            const auto result = index.query("query", 4u);
            expect(result.status.ok() && result.hits.size() == first_hits.size());
            for (size_t i = 0u; i < result.hits.size() && i < first_hits.size(); i++) {
                expect(result.hits[i].path == first_hits[i].path)
                    << "hybrid ranking must survive the cache round trip";
                expect(luisa::abs(result.hits[i].score - first_hits[i].score) < 1e-6);
            }
        }

        // Collect the three non-empty sidecars (sorted by name) to invalidate one
        // at a time: alpha, beta and gamma.
        luisa::vector<luisa::filesystem::path> non_empty;
        for (const auto &path : sparse_files) {
            if (file_bytes(path) != 0u) {
                non_empty.push_back(path);
            }
        }
        expect(non_empty.size() == 3u);

        const auto run_partial_load = [&root, &cache_dir](size_t expected_cached,
                                                          size_t expected_calls) {
            DiskEmbedCache cache{nullptr};
            FakeHybridEncoder encoder;
            RetrievalIndex index{encoder.callback()};
            index.set_embed_cache(&cache);
            const auto loaded = index.load(root);
            expect(loaded.status.ok() && loaded.failed == 0u);
            expect(loaded.cached == expected_cached);
            expect(encoder.calls == expected_calls);
            expect(list_cache_files(cache_dir, false).size() == 4u)
                << "dense .bin artifacts must not be added or removed";
            expect(list_cache_files(cache_dir, true).size() == 4u)
                << "every hybrid entry keeps a sidecar";
            return index.query("query", 4u);
        };

        // Deleting a sparse sidecar forces a re-encode of both halves for that file.
        if (non_empty.size() == 3u) {
            std::error_code ec;
            expect(luisa::filesystem::remove(non_empty[0], ec) && !ec);
            const auto result = run_partial_load(3u, 1u);
            expect(result.status.ok() && result.hits.size() == 4u);
        }

        // A corrupt (NaN) weight in an otherwise correctly sized sidecar is a miss.
        if (non_empty.size() == 3u) {
            const luisa::string garbage(8u, static_cast<char>(0xff));
            write_file(non_empty[1], garbage);
            const auto result = run_partial_load(3u, 1u);
            expect(result.status.ok() && result.hits.size() == 4u);
        }

        // A truncated sidecar (size/count mismatch) is a miss as well.
        if (non_empty.size() == 3u) {
            write_file(non_empty[2], luisa::string_view{"\x01\x02\x03\x04", 4u});
            const auto result = run_partial_load(3u, 1u);
            expect(result.status.ok() && result.hits.size() == 4u);
        }

        // Everything was repaired along the way: the cache is fully warm again.
        {
            DiskEmbedCache cache{nullptr};
            FakeHybridEncoder encoder;
            RetrievalIndex index{encoder.callback()};
            index.set_embed_cache(&cache);
            const auto loaded = index.load(root);
            expect(loaded.status.ok() && loaded.cached == 4u && loaded.failed == 0u);
            expect(encoder.calls == 0u);
            const auto result = index.query("query", 4u);
            expect(result.status.ok() && result.hits.size() == first_hits.size());
            for (size_t i = 0u; i < result.hits.size() && i < first_hits.size(); i++) {
                expect(result.hits[i].path == first_hits[i].path);
                expect(luisa::abs(result.hits[i].score - first_hits[i].score) < 1e-6);
            }
        }
    };
}

void reg_hybrid_cache_requires_sidecar() {
    "retrieval_index_hybrid_cache_dense_only_metadata"_test = [] {
        TempDirectory temporary;
        const auto root = temporary.path() / "root";
        write_file(root / "a.txt", "alpha");
        write_file(root / "b.txt", "beta");
        const auto cache_dir = root / ".lc_embed_cache";

        // A dense-only import keeps the v1 layout: dense .bin files, no sidecars.
        {
            DiskEmbedCache cache{nullptr};
            FakeHybridEncoder encoder;
            RetrievalIndex index{encoder.dense_callback()};
            index.set_embed_cache(&cache);
            expect(index.load(root).status.ok());
            expect(encoder.calls == 2u);
        }
        const auto dense_before = list_cache_files(cache_dir, false);
        expect(dense_before.size() == 2u);
        expect(list_cache_files(cache_dir, true).empty())
            << "dense-only stores must never create sparse sidecars";
        expect(read_text_file(root / ".lc_embed_cache.json").find("sparse_count") ==
               luisa::string::npos);

        // That v1 metadata stays a dense-mode hit.
        {
            DiskEmbedCache cache{nullptr};
            FakeHybridEncoder encoder;
            RetrievalIndex index{encoder.dense_callback()};
            index.set_embed_cache(&cache);
            const auto loaded = index.load(root);
            expect(loaded.status.ok() && loaded.cached == 2u && loaded.failed == 0u);
            expect(encoder.calls == 0u);
        }

        // Hybrid mode misses on it: the dense half alone is not a hybrid hit.
        {
            DiskEmbedCache cache{nullptr};
            FakeHybridEncoder encoder;
            RetrievalIndex index{encoder.callback()};
            index.set_embed_cache(&cache);
            const auto loaded = index.load(root);
            expect(loaded.status.ok());
            expect(loaded.added == 2u && loaded.cached == 0u && loaded.failed == 0u);
            expect(encoder.calls == 2u) << "a missing sparse sidecar forces a full re-encode";
        }
        const auto dense_after = list_cache_files(cache_dir, false);
        expect(dense_after.size() == dense_before.size())
            << "re-encoding must not change the number of dense .bin files";
        for (size_t i = 0u; i < dense_after.size() && i < dense_before.size(); i++) {
            expect(dense_after[i].filename().string() == dense_before[i].filename().string())
                << "dense .bin names are the content MD5s and must be unchanged";
        }
        expect(list_cache_files(cache_dir, true).size() == 2u);

        // With sidecars present both modes hit; dense loads ignore them.
        {
            DiskEmbedCache cache{nullptr};
            FakeHybridEncoder encoder;
            RetrievalIndex index{encoder.dense_callback()};
            index.set_embed_cache(&cache);
            const auto loaded = index.load(root);
            expect(loaded.status.ok() && loaded.cached == 2u);
            expect(encoder.calls == 0u);
        }
        {
            DiskEmbedCache cache{nullptr};
            FakeHybridEncoder encoder;
            RetrievalIndex index{encoder.callback()};
            index.set_embed_cache(&cache);
            const auto loaded = index.load(root);
            expect(loaded.status.ok() && loaded.cached == 2u);
            expect(encoder.calls == 0u);
        }
    };
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    reg_extension_discovery();
    reg_empty_corpus_and_validation();
    reg_deterministic_ranking();
    reg_root_ownership_and_unload();
    reg_single_file_update();
    reg_subdirectory_update();
    reg_failed_reencode_preserves_document();
    reg_embedding_validation();
    reg_disk_cache_roundtrip();
    reg_disk_cache_invalidation();
    reg_hybrid_fusion_ranking();
    reg_hybrid_sparse_canonicalization();
    reg_hybrid_sparse_failures_and_rollback();
    reg_hybrid_disk_cache_roundtrip();
    reg_hybrid_cache_requires_sidecar();
    return 0;
}
