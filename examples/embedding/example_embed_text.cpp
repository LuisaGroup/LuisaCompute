// bge-m3 embedding engine — M6 command-line demo (embedding_model_plan.md §6,
// milestone M6 "demo"; target example_embed_text). Encodes sentences with a
// Session over a packed model.bgm + tokenizer.json on a chosen backend and
// prints per-sentence results through plain luisa::core logging (no GUI).
//
// CLI:
//   example_embed_text <backend> <model.bgm> <tokenizer.json>
//                      [--sparse] [--colbert] [sentence...]
// Sentences are the positional arguments after the three required ones; with
// none, one sentence per line is read from stdin until EOF. --sparse and
// --colbert request the retrieval heads (EncodeOptions). Exit codes: 0 success,
// 1 bad args / missing files / session load failure / wrong norm, 2 the demo
// was compiled without the Session API (B1's session.h not in the tree).
//
// The demo codes against the frozen Session contract (plan §3.1 as amended by
// the parent):
//   Session::create(Device&, bgm_package, tokenizer_json, num_workers = 2)
//   Embedding encode(string_view utf8, const EncodeOptions& = {})
//   EncodeOptions{max_seq_len=512, want_sparse, want_colbert}
//   Embedding{dense [1024], sparse [(id,weight)], colbert [S*1024]}
// When session.h is absent at compile time this file still builds: it exercises
// full argument handling + file validation and then stops at a loud
// "session unavailable" pre-check (B3/B1 wave integration, status reported).
//
// Token counts are reported from the engine Tokenizer (A1, stable API) applied
// with the same EncodeOptions::max_seq_len truncation, so the printed count is
// exactly the number of input rows the executor saw (including <s>/</s>).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string_view>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "embedding/embed/tokenizer.h"

#if __has_include("embedding/embed/session.h")
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include "embedding/embed/session.h"
#endif

using namespace luisa;

namespace {

    void print_usage() {
        LUISA_INFO("usage: example_embed_text <backend> <model.bgm> <tokenizer.json> "
                   "[--sparse] [--colbert] [sentence...]");
        LUISA_INFO("  backend          dx | vk | cuda | metal | fallback | simd | ...");
        LUISA_INFO("  model.bgm        packed weights (BGM3 v1, see tools/pack_model.cpp)");
        LUISA_INFO("  tokenizer.json   HF fast-tokenizer JSON (bge-m3 Unigram)");
        LUISA_INFO("  sentence...      texts to encode; if none, one per line from stdin");
    }

    struct ParsedArgs {
        luisa::string backend;
        std::filesystem::path bgm;
        std::filesystem::path tokenizer_json;
        luisa::vector<luisa::string> sentences;// empty -> read stdin
        bool want_sparse{false};
        bool want_colbert{false};
        bool ok{false};
    };

    ParsedArgs parse_args(int argc, char *argv[]) {
        ParsedArgs p;
        for(int i = 1; i < argc; i++) {
            const std::string_view a{argv[i]};
            if(a == "--sparse") {
                p.want_sparse = true;
            } else if(a == "--colbert") {
                p.want_colbert = true;
            } else if(!a.starts_with("-")) {
                if(p.backend.empty()) {
                    p.backend = luisa::string{a.data(), a.size()};
                } else if(p.bgm.empty()) {
                    p.bgm = std::filesystem::path{argv[i]};
                } else if(p.tokenizer_json.empty()) {
                    p.tokenizer_json = std::filesystem::path{argv[i]};
                } else {
                    p.sentences.emplace_back(luisa::string{a.data(), a.size()});
                }
            } else {
                LUISA_INFO("example_embed_text: unknown option '{}'", a);
                return p;
            }
        }
        p.ok = !p.backend.empty() && !p.bgm.empty() && !p.tokenizer_json.empty();
        return p;
    }

    // shared validation of the parsed arguments (both build modes)
    bool validate_files(const ParsedArgs &p) {
        if(!p.ok) {
            LUISA_INFO("example_embed_text: <backend> <model.bgm> <tokenizer.json> are all required");
            print_usage();
            return false;
        }
        if(!std::filesystem::exists(p.bgm)) {
            LUISA_INFO("example_embed_text: packed model '{}' does not exist", p.bgm.string());
            return false;
        }
        if(!std::filesystem::exists(p.tokenizer_json)) {
            LUISA_INFO("example_embed_text: tokenizer '{}' does not exist", p.tokenizer_json.string());
            return false;
        }
        return true;
    }

}// namespace

#if !__has_include("embedding/embed/session.h")
// ---- stub build: B1's session.h is not in the tree yet ----------------------
int main(int argc, char *argv[]) {
    const auto args = parse_args(argc, argv);
    if(!validate_files(args)) {
        return 1;
    }
    LUISA_INFO("example_embed_text: arguments accepted (backend '{}', package '{}', "
               "tokenizer '{}', {} explicit sentence(s){}, sparse={}, colbert={})",
               args.backend, args.bgm.string(), args.tokenizer_json.string(),
               args.sentences.size(),
               args.sentences.empty() ? " (stdin mode)" : "",
               args.want_sparse, args.want_colbert);
    LUISA_INFO("SESSION UNAVAILABLE: embedding/embed/session.h was not in the tree when "
               "this binary was compiled (B1 executor/session deliverable); the demo "
               "cannot encode. Rebuild example_embed_text after B1 lands.");
    return 2;
}
#else
// ---- full build: the Session API is present ---------------------------------
namespace {

    // reads all of stdin; returns one line per sentence (CR/LF stripped)
    luisa::vector<luisa::string> read_stdin_lines() {
        luisa::vector<luisa::string> lines;
        luisa::string cur;
        int c = std::fgetc(stdin);
        while(c != EOF) {
            if(c == '\n') {
                if(!cur.empty() && cur.back() == '\r') {
                    cur.pop_back();
                }
                lines.push_back(std::move(cur));
                cur.clear();
            } else {
                cur.push_back(static_cast<char>(c));
            }
            c = std::fgetc(stdin);
        }
        if(!cur.empty()) {
            lines.push_back(std::move(cur));
        }
        return lines;
    }

    // first `n` floats of `v`, comma separated, for one-line logging
    luisa::string join_head(const luisa::vector<float> &v, size_t n) {
        luisa::string s;
        for(size_t i = 0u; i < n && i < v.size(); i++) {
            if(i > 0u) {
                s += ", ";
            }
            s += luisa::format("{:.6g}", v[i]);
        }
        return s;
    }

}// namespace

int main(int argc, char *argv[]) {
    auto args = parse_args(argc, argv);
    if(!validate_files(args)) {
        return 1;
    }
    if(args.sentences.empty()) {
        LUISA_INFO("example_embed_text: no sentence arguments; reading sentences from stdin "
                   "(one per line, EOF to finish)");
        args.sentences = read_stdin_lines();
        if(args.sentences.empty()) {
            LUISA_INFO("example_embed_text: stdin produced no sentences");
            return 1;
        }
    }

    // device + tokenizer + session
    compute::Context ctx{argv[0]};
    auto device = ctx.create_device(args.backend);
    if(!device) {
        luisa::string installed;
        for(const auto &b : ctx.installed_backends()) {
            if(!installed.empty()) {
                installed += ", ";
            }
            installed += b;
        }
        LUISA_INFO("example_embed_text: backend '{}' not available (installed: '{}')",
                   args.backend, installed);
        return 1;
    }
    auto tkn = embed::Tokenizer::load(args.tokenizer_json);
    auto session = embed::Session::create(device, args.bgm, args.tokenizer_json);
    if(session == nullptr) {
        LUISA_INFO("example_embed_text: Session::create failed for package '{}'", args.bgm.string());
        return 1;
    }

    embed::EncodeOptions enc_opt;// max_seq_len = 512 (plan §3.2 default)
    enc_opt.want_sparse = args.want_sparse;
    enc_opt.want_colbert = args.want_colbert;

    size_t failures = 0u;
    for(size_t i = 0u; i < args.sentences.size(); i++) {
        const auto &text = args.sentences[i];
        const auto ids = tkn->encode(text, enc_opt.max_seq_len);
        auto emb = session->encode(std::string_view{text.data(), text.size()}, enc_opt);
        if(emb.dense.empty()) {
            LUISA_WARNING("example_embed_text: sentence {} produced an empty dense vector", i);
            failures++;
            continue;
        }
        float sumsq = 0.f;
        for(const auto v : emb.dense) {
            sumsq += v * v;
        }
        const float norm = std::sqrt(sumsq);
        const auto shown = text.size() > 64u ? luisa::string{text.data(), 64u} + "..." : text;
        LUISA_INFO("sentence {}: \"{}\"", i, shown);
        LUISA_INFO("  tokens : {} (with <s>/</s>, truncated to {})", ids.size(), enc_opt.max_seq_len);
        LUISA_INFO("  dense  : first 8 of {} = [{}], |v| = {:.6f}", emb.dense.size(), join_head(emb.dense, 8u), norm);
        if(std::abs(norm - 1.0f) > 1e-3f) {
            LUISA_WARNING("  dense norm is {:.6f}, expected ~1.0 (L2-normalized output)", norm);
            failures++;
        }
        if(enc_opt.want_sparse) {
            size_t terms = 0u;
            luisa::string listing;
            for(const auto &pr : emb.sparse) {
                if(pr.second > 0.0f) {
                    terms++;
                    listing += luisa::format("({},{:.5f}) ", pr.first, pr.second);
                    if(terms % 12u == 0u) {
                        listing += "\n";
                    }
                }
            }
            LUISA_INFO("  sparse : {} term(s) with weight > 0:{}", terms,
                       listing.empty() ? " <none>" : "\n" + listing);
        }
        if(enc_opt.want_colbert) {
            if(emb.colbert.empty()) {
                LUISA_WARNING("  colbert: requested but empty");
                failures++;
            } else {
                const size_t rows = emb.colbert.size() / 1024u;
                LUISA_INFO("  colbert: {} rows x 1024, first row first 8 = [{}]", rows, join_head(emb.colbert, 8u));
            }
        }
    }
    if(failures > 0u) {
        LUISA_INFO("example_embed_text: {} failure(s)", failures);
        return 1;
    }
    return 0;
}
#endif
