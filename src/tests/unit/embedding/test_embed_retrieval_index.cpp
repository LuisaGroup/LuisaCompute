// Test for embedding/embed/retrieval_index.h — host-only whole-file retrieval.
// This test covers extension filtering, recursive discovery, deterministic dense
// ranking, root ownership, exact-content incremental updates, deletion handling,
// and per-file encode-failure rollback. No model, device, or backend is used.
// Run: xmake run test_embed_retrieval_index
#include "ut/ut.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/mathematics.h>
#include <luisa/core/stl/filesystem.h>

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
    return 0;
}
