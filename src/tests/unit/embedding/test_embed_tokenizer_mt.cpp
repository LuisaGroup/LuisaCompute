// test_embed_tokenizer_mt.cpp -- bge-m3 tokenizer reentrancy (thread pool)
// (embedding_model_test_plan.md §2 L0 row "test_embed_tokenizer_mt": "8 threads x
// 64 sentences, outputs identical to single-thread run"; §H8 of the optimization
// plan requires the shared Tokenizer to be usable from the engine's worker pool).
// Host-only test (LUISA_EMBED_HOST_ONLY=1): it creates no LuisaCompute device and
// takes no backend argument.
//
// This test covers:
// - one shared embed::Tokenizer instance (tokenizer.h documents instances as
//   immutable after load and safe to share across threads) hammered by 8 worker
//   threads x 64 fixed corpus sentences (184 encodes-worth of unique inputs,
//   512 concurrent encodes total), with every thread's output compared
//   BIT-IDENTICALLY (full id vectors) against a single-threaded sequential run of
//   the same sentences; each thread visits the shared instance in a different
//   order (rotated by its thread index) so the interleaving is not uniform
// - a second sequential pass, asserting the sequential reference is itself
//   reproducible (a nondeterministic encode would otherwise pass by luck)
// - the empty-string encode inside every worker as well (the shortest possible
//   output, exercised concurrently)
// - the two §2 edge cases that test_embed_tokenizer.cpp does NOT implement (it
//   only compares the pinned corpus goldens; corpus.txt has no empty line and no
//   >8192-token sentence):
//     * encode("") == exactly [<s>, </s>] == [0, 2]
//     * an input longer than 8192 tokens truncates to 8192 with </s> last; the
//       truncation is verified to keep [bos] + body[:max_len-2] + [eos] by
//       requiring the max_len=512 result to be the exact prefix of the
//       max_len=8192 result (which also proves the body exceeds the 8192 ceiling,
//       i.e. truncation actually fired rather than the input happening to fit)
//
// Sentence selection is deterministic: 64 indices evenly spread over corpus.txt
// (k * (count-1) / 63), so index 0 and the last -- a truncated-path sentence --
// are always included. corpus.txt lines escape backslash, \n and \r and are
// unescaped per golden_meta.json conventions before encoding.
//
// Run: xmake run test_embed_tokenizer_mt
#include "ut/ut.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <thread>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "embedding/embed/tokenizer.h"

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {
    luisa::vector<char> read_file(const std::filesystem::path &p) {
        if(std::FILE *f = std::fopen(p.string().c_str(), "rb"); f != nullptr) {
            luisa::vector<char> data;
            char buf[65536];
            size_t got = 0u;
            while((got = std::fread(buf, 1u, sizeof(buf), f)) > 0u) {
                data.insert(data.end(), buf, buf + static_cast<ptrdiff_t>(got));
            }
            std::fclose(f);
            return data;
        }
        LUISA_ERROR("cannot open test data file \"{}\"", p.string());
    }
    // corpus.txt escapes backslash, \n and \r (see golden_meta.json)
    luisa::string unescape_corpus_line(luisa::string_view line) {
        luisa::string out;
        out.reserve(line.size());
        size_t i = 0u;
        while(i < line.size()) {
            if(line[i] == '\\' && i + 1u < line.size()) {
                const auto c = line[i + 1u];
                if(c == 'n') {
                    out.push_back('\n');
                    i += 2u;
                    continue;
                }
                if(c == 'r') {
                    out.push_back('\r');
                    i += 2u;
                    continue;
                }
                if(c == '\\') {
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
        const auto emit_line = [&](size_t end) {
            luisa::string_view line{raw.data() + start, end - start};
            if(!line.empty() && line.back() == '\r') line.remove_suffix(1u);// CRLF tolerance
            lines.push_back(unescape_corpus_line(line));
            start = end + 1u;
        };
        for(size_t i = 0u; i < raw.size(); i++) {
            if(raw[i] == '\n') emit_line(i);
        }
        if(start < raw.size()) emit_line(raw.size());// final line without newline
        return lines;
    }
    const std::filesystem::path &resolve_data_dir() {
        // Same three-step resolution as test_embed_tokenizer.cpp: the compiled-in
        // LUISA_EMBED_TEST_DATA_DIR macro (xmake injects it with native
        // backslashes, which the compiler consumes as escapes, so it must be
        // verified before use), then this file's own path, then a walk up from cwd.
        static const std::filesystem::path dir = [] {
            std::filesystem::path cand{luisa::string{LUISA_EMBED_TEST_DATA_DIR}};
            if(std::filesystem::exists(cand / "corpus.txt")) return cand;
            auto p = std::filesystem::path{__FILE__};
            for(int i = 0; i < 3; i++) p = p.parent_path();
            p /= "data";
            p /= "embedding";
            if(std::filesystem::exists(p / "corpus.txt")) return p;
            p = std::filesystem::current_path();
            while(!p.empty()) {
                auto q = p / "src" / "tests" / "data" / "embedding";
                if(std::filesystem::exists(q / "corpus.txt")) return q;
                auto parent = p.parent_path();
                if(parent == p) break;
                p = parent;
            }
            LUISA_ERROR("cannot locate src/tests/data/embedding (corpus.txt) from cwd \"{}\"",
                        std::filesystem::current_path().string());
        }();
        return dir;
    }
    std::filesystem::path tokenizer_json_path() {
        // bge-m3 assets sit at the repo root (walk up: embedding -> data -> tests
        // -> src -> repo root), same as test_embed_tokenizer.cpp
        auto root = resolve_data_dir();
        for(int i = 0; i < 4; i++) root = root.parent_path();
        return root / "bge-m3" / "bge-m3" / "tokenizer.json";
    }

    constexpr uint32_t k_thread_count = 8u;
    constexpr uint32_t k_sentences_per_thread = 64u;

    // deterministic selection: 64 indices evenly spread over the corpus, always
    // including the first and the last (the last corpus sentences exercise the
    // truncated path)
    luisa::vector<uint32_t> select_sentence_indices(size_t corpus_size) {
        luisa::vector<uint32_t> idx;
        idx.reserve(k_sentences_per_thread);
        for(uint32_t k = 0u; k < k_sentences_per_thread; k++) {
            idx.push_back(static_cast<uint32_t>(static_cast<uint64_t>(k) * (corpus_size - 1ull) /
                                                (k_sentences_per_thread - 1ull)));
        }
        return idx;
    }

    // a >8192-token input: the same short sentence repeated 1024 times (~10
    // tokens each, so the body is well past the 8192 ceiling and the truncation
    // path must fire)
    luisa::string make_overlong_input() {
        luisa::string s;
        constexpr luisa::string_view unit = "The quick brown fox jumps over the lazy dog. ";
        s.reserve(unit.size() * 1024u);
        for(uint32_t i = 0u; i < 1024u; i++) s += unit;
        return s;
    }

    void test_mt_reentrancy(const luisa::embed::Tokenizer &tkn,
                            const luisa::vector<luisa::string> &corpus,
                            const luisa::vector<uint32_t> &chosen) {
        // sequential reference pass
        luisa::vector<luisa::vector<uint32_t>> ref;
        ref.reserve(chosen.size());
        for(const auto ci : chosen) {
            ref.push_back(tkn.encode(corpus[ci]));
        }
        // reproducibility of the reference itself
        uint32_t nondeterministic = 0u;
        for(size_t j = 0u; j < chosen.size(); j++) {
            if(tkn.encode(corpus[chosen[j]]) != ref[j]) nondeterministic++;
        }
        expect(nondeterministic == 0u)
            << luisa::format("sequential encode is not reproducible ({} of {} sentences differ on a second pass)",
                             nondeterministic, chosen.size());

        // concurrent pass: k_thread_count threads x 64 sentences over ONE shared
        // instance, each thread rotating its visit order by its index
        luisa::vector<uint32_t> per_thread_mismatch(k_thread_count, 0u);
        luisa::vector<uint32_t> per_thread_count(k_thread_count, 0u);
        luisa::vector<std::thread> threads;
        threads.reserve(k_thread_count);
        const auto t0 = std::chrono::steady_clock::now();
        for(uint32_t t = 0u; t < k_thread_count; t++) {
            threads.emplace_back([&, t]() {
                auto &mismatch = per_thread_mismatch[t];
                auto &done = per_thread_count[t];
                for(uint32_t r = 0u; r < k_sentences_per_thread; r++) {
                    const size_t j = (static_cast<size_t>(t) + r) % k_sentences_per_thread;
                    const auto got = tkn.encode(corpus[chosen[j]]);
                    done++;
                    if(got != ref[j]) {
                        mismatch++;
                        size_t div = 0u;
                        while(div < got.size() && div < ref[j].size() && got[div] == ref[j][div]) div++;
                        LUISA_WARNING("thread {} sentence {} ({} ids vs reference {} ids); first divergence "
                                      "at {}: got {}, reference {}",
                                      t, j, got.size(), ref[j].size(), div,
                                      div < got.size() ? got[div] : 0xFFFFFFFFu,
                                      div < ref[j].size() ? ref[j][div] : 0xFFFFFFFFu);
                    }
                    // the shortest possible output, concurrently as well
                    const auto e = tkn.encode("");
                    if(e.size() != 2u || e[0] != 0u || e[1] != 2u) {
                        mismatch++;
                        LUISA_WARNING("thread {}: encode(\"\") is not [<s>, </s>]", t);
                    }
                }
            });
        }
        for(auto &th : threads) th.join();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();

        uint32_t total_mismatch = 0u, total_encodes = 0u;
        for(uint32_t t = 0u; t < k_thread_count; t++) {
            total_mismatch += per_thread_mismatch[t];
            total_encodes += per_thread_count[t];
        }
        LUISA_INFO("test_embed_tokenizer_mt: {} threads x {} sentences (+ {} empty-string encodes) over one "
                   "shared Tokenizer in {} ms; concurrent encodes {}, mismatches vs the sequential run {}",
                   k_thread_count, k_sentences_per_thread, k_sentences_per_thread * k_thread_count,
                   ms, total_encodes, total_mismatch);
        expect(total_encodes == k_thread_count * k_sentences_per_thread)
            << "every worker must have encoded all 64 sentences";
        expect(total_mismatch == 0u)
            << "concurrent encodes must be bit-identical to the single-thread sequential run";
    }

    void test_truncation_edge_cases(const luisa::embed::Tokenizer &tkn) {
        const auto overlong = make_overlong_input();
        const auto full = tkn.encode(overlong, 8192u);
        uint32_t mismatch_count = 0u;
        if(full.size() != 8192u) {
            LUISA_WARNING("overlong input encoded to {} ids at max_len 8192 (expected a full 8192, "
                          "i.e. the truncation path must fire)",
                          full.size());
            mismatch_count++;
        }
        if(full.empty() || full.front() != 0u) {
            LUISA_WARNING("truncated sequence must start with <s> (id 0)");
            mismatch_count++;
        }
        if(full.size() != 8192u || full[8191u] != 2u) {
            LUISA_WARNING("truncated sequence must end with </s> (id 2), got {}",
                          full.size() == 8192u ? full[8191u] : 0xFFFFFFFFu);
            mismatch_count++;
        }
        // prefix consistency across two max_len values proves [bos] +
        // body[:max_len-2] + [eos] and that the body is longer than 8192 - 2
        const auto shorty = tkn.encode(overlong, 512u);
        if(shorty.size() != 512u || shorty.front() != 0u || shorty.back() != 2u) {
            LUISA_WARNING("the same input at max_len 512 must truncate to [0, ..., 2] of size 512, got size {}",
                          shorty.size());
            mismatch_count++;
        } else {
            // [bos] + body[:max_len-2] must be the exact prefix of the longer
            // truncation; the final element of each is </s>, which only lines up
            // past the cut, hence size()-1 elements compared
            for(size_t i = 0u; i + 1u < shorty.size(); i++) {
                if(shorty[i] != full[i]) {
                    LUISA_WARNING("truncation is not prefix-consistent at position {}: {} vs {}",
                                  i, shorty[i], full[i]);
                    mismatch_count++;
                    break;
                }
            }
        }
        LUISA_INFO("test_embed_tokenizer_mt: overlong input ({} bytes) -> exactly 8192 ids with <s> first and "
                   "</s> last, prefix-consistent with the max_len=512 truncation (so the truncated body is "
                   "longer than 8192 tokens); {} problems",
                   overlong.size(), mismatch_count);
        expect(mismatch_count == 0u) << ">8192-token inputs must truncate to 8192 keeping <s> first and </s> last";
    }
}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    const auto corpus = load_corpus(resolve_data_dir());
    expect(corpus.size() >= k_sentences_per_thread)
        << luisa::format("the corpus must hold at least {} sentences, it holds {}",
                         k_sentences_per_thread, corpus.size());

    auto tkn = luisa::embed::Tokenizer::load(tokenizer_json_path());
    expect(tkn != nullptr) << "tokenizer load must succeed";
    if(tkn == nullptr) return 1;

    const auto chosen = select_sentence_indices(corpus.size());
    expect(chosen.size() == k_sentences_per_thread);
    expect(chosen.front() == 0u && chosen.back() == static_cast<uint32_t>(corpus.size() - 1u))
        << "the deterministic selection must span the corpus from the first to the last sentence";

    test_mt_reentrancy(*tkn, corpus, chosen);
    test_truncation_edge_cases(*tkn);
    return 0;
}
