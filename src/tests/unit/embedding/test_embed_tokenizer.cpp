// Test for embedding/embed/tokenizer.h — bge-m3 tokenizer (embedding_model_plan.md
// §3.2, milestone M1 gate). Host-only test (LUISA_EMBED_HOST_ONLY=1): it creates
// no LuisaCompute device and takes no backend argument.
//
// This test covers:
// - Tokenizer::load() of the real bge-m3 tokenizer.json (Unigram pipeline)
// - exact token-id parity for ALL corpus sentences against the pinned goldens
//   (src/tests/data/embedding/{corpus.txt,golden_tokens.json}, produced by HF
//   `tokenizers` with truncation max_length=512; the last sentences exercise the
//   truncated path)
//
// Run: xmake run test_embed_tokenizer
#include "ut/ut.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "embedding/embed/tokenizer.h"

#include <yyjson.h>

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Reads a whole file; fatal on failure.
luisa::vector<char> read_file(const std::filesystem::path &p) {
    if (std::FILE *f = std::fopen(p.string().c_str(), "rb"); f != nullptr) {
        luisa::vector<char> data;
        char buf[65536];
        size_t got = 0u;
        while ((got = std::fread(buf, 1u, sizeof(buf), f)) > 0u) {
            data.insert(data.end(), buf, buf + static_cast<ptrdiff_t>(got));
        }
        std::fclose(f);
        return data;
    }
    LUISA_ERROR("cannot open test data file \"{}\"", p.string());
}

// corpus.txt lines escape backslash, \n and \r (see golden_meta.json);
// readers must unescape after reading a line.
luisa::string unescape_corpus_line(luisa::string_view line) {
    luisa::string out;
    out.reserve(line.size());
    size_t i = 0u;
    while (i < line.size()) {
        if (line[i] == '\\' && i + 1u < line.size()) {
            const auto c = line[i + 1u];
            if (c == 'n') {
                out.push_back('\n');
                i += 2u;
                continue;
            }
            if (c == 'r') {
                out.push_back('\r');
                i += 2u;
                continue;
            }
            if (c == '\\') {
                out.push_back('\\');
                i += 2u;
                continue;
            }
        }
        out.push_back(line[i]);
        i++;
    }
    return out;
}

luisa::vector<luisa::string> load_corpus(const std::filesystem::path &dir) {
    auto raw = read_file(dir / "corpus.txt");
    luisa::vector<luisa::string> lines;
    size_t start = 0u;
    auto emit_line = [&](size_t end) {
        luisa::string_view line{raw.data() + start, end - start};
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1u);// CRLF tolerance
        lines.push_back(unescape_corpus_line(line));
        start = end + 1u;
    };
    for (size_t i = 0u; i < raw.size(); i++) {
        if (raw[i] == '\n') emit_line(i);
    }
    if (start < raw.size()) emit_line(raw.size());// final line without newline
    return lines;
}

struct golden_entry {
    uint32_t index;
    luisa::vector<uint32_t> ids;
};

luisa::vector<golden_entry> load_goldens(const std::filesystem::path &dir) {
    auto path = (dir / "golden_tokens.json").string();
    yyjson_read_err err;
    yyjson_doc *doc = yyjson_read_file(path.c_str(), 0u, nullptr, &err);
    if (doc == nullptr) LUISA_ERROR("cannot read golden_tokens.json: {} (pos {})",
                                    (err.msg != nullptr ? err.msg : "?"), err.pos);
    struct guard {
        yyjson_doc *d;
        ~guard() { yyjson_doc_free(d); }
    } g{doc};
    const auto *root = yyjson_doc_get_root(doc);
    const auto *entries = yyjson_obj_get(root, "entries");
    if (!yyjson_is_arr(entries)) LUISA_ERROR("golden_tokens.json: missing entries array");
    luisa::vector<golden_entry> out;
    yyjson_val *item;
    yyjson_arr_iter iter = yyjson_arr_iter_with(entries);
    while ((item = yyjson_arr_iter_next(&iter)) != nullptr) {
        const auto *iv = yyjson_obj_get(item, "i");
        const auto *ids = yyjson_obj_get(item, "ids");
        if (!yyjson_is_uint(iv) || !yyjson_is_arr(ids))
            LUISA_ERROR("golden_tokens.json: malformed entry");
        golden_entry e;
        e.index = static_cast<uint32_t>(yyjson_get_uint(iv));
        yyjson_val *id;
        yyjson_arr_iter id_iter = yyjson_arr_iter_with(ids);
        while ((id = yyjson_arr_iter_next(&id_iter)) != nullptr) {
            if (!yyjson_is_uint(id)) LUISA_ERROR("golden_tokens.json: non-integer id");
            e.ids.push_back(static_cast<uint32_t>(yyjson_get_uint(id)));
        }
        out.push_back(std::move(e));
    }
    return out;
}

const std::filesystem::path &resolve_data_dir() {
    // Preferred: the compiled-in LUISA_EMBED_TEST_DATA_DIR macro — but note
    // the xmake registration injects it with native backslashes which the
    // compiler consumes as escape sequences inside the string literal, so
    // it must be verified before use (CMake's forward-slash form is intact).
    static const std::filesystem::path dir = [] {
        std::filesystem::path cand{luisa::string{LUISA_EMBED_TEST_DATA_DIR}};
        if (std::filesystem::exists(cand / "corpus.txt")) return cand;
        // Relative/absolute path of this source file: .../src/tests/unit/
        // embedding/test_embed_tokenizer.cpp -> .../src/tests/data/embedding
        auto p = std::filesystem::path{__FILE__};
        for (int i = 0; i < 3; i++) p = p.parent_path();
        p /= "data";
        p /= "embedding";
        if (std::filesystem::exists(p / "corpus.txt")) return p;
        // Walk up from the current directory looking for the data dir.
        p = std::filesystem::current_path();
        while (!p.empty()) {
            auto q = p / "src" / "tests" / "data" / "embedding";
            if (std::filesystem::exists(q / "corpus.txt")) return q;
            auto parent = p.parent_path();
            if (parent == p) break;
            p = parent;
        }
        LUISA_ERROR("cannot locate src/tests/data/embedding (corpus.txt) from cwd \"{}\"",
                    std::filesystem::current_path().string());
    }();
    return dir;
}

std::filesystem::path test_data_dir() { return resolve_data_dir(); }

// bge-m3 assets sit at the repo root (walk up: embedding -> data -> tests ->
// src -> repo root).
std::filesystem::path tokenizer_json_path() {
    auto root = test_data_dir();
    for (int i = 0; i < 4; i++) root = root.parent_path();
    return root / "bge-m3" / "bge-m3" / "tokenizer.json";
}

}// namespace

static const auto test_embed_tokenizer_registration = [] {
    "bge_m3_tokenizer_corpus_exact_match"_test = [] {
        const auto corpus = load_corpus(test_data_dir());
        const auto goldens = load_goldens(test_data_dir());
        expect(corpus.size() == goldens.size())
            << luisa::format("corpus has {} lines but goldens have {} entries",
                             corpus.size(), goldens.size());

        const auto tkn_path = tokenizer_json_path();
        auto tkn = luisa::embed::Tokenizer::load(tkn_path);
        expect(tkn != nullptr) << "tokenizer load must succeed";
        if (tkn == nullptr) return;

        size_t mismatch_count = 0u;
        for (const auto &g : goldens) {
            if (g.index >= corpus.size()) {
                LUISA_WARNING("golden entry {} beyond corpus size", g.index);
                mismatch_count++;
                continue;
            }
            const auto got = tkn->encode(corpus[g.index]);
            if (got != g.ids) {
                mismatch_count++;
                // report index + first divergence (position + expected/got ids)
                size_t div = 0u;
                while (div < got.size() && div < g.ids.size() && got[div] == g.ids[div]) div++;
                LUISA_WARNING("sentence {} mismatch (got {} ids, golden {} ids); first divergence "
                              "at position {}: got {}, expected {}",
                              g.index, got.size(), g.ids.size(), div,
                              div < got.size() ? got[div] : 0xFFFFFFFFu,
                              div < g.ids.size() ? g.ids[div] : 0xFFFFFFFFu);
            }
        }
        LUISA_INFO("bge-m3 tokenizer corpus gate: {} sentences, {} mismatches",
                   goldens.size(), mismatch_count);
        expect(mismatch_count == 0u) << "all corpus sentences must exact-match the golden token ids";
    };

    // Regression: destruction must not pass a CRT-heap pointer to mimalloc.
    "bge_m3_tokenizer_load_destroy"_test = [] {
        auto tkn = luisa::embed::Tokenizer::load(tokenizer_json_path());
        expect(tkn != nullptr) << "tokenizer load must succeed";
        if (tkn == nullptr) return;
        tkn.reset();
        expect(tkn == nullptr) << "reset() must release the tokenizer";
    };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}
