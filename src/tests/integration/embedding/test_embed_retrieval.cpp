// L3 retrieval suite for the bge-m3 engine (embedding_model_test_plan.md
// S4.3, test_embed_retrieval): corpus-library build + search + ranking/metrics
// parity + negative controls + cross-backend checksum. Device test, backend =
// argv[1] (dx / vk / cuda / fallback); optional package path argv[2]; optional
// "--rebuild-cache" token anywhere in argv forces a cold corpus-embedding cache
// rebuild.
//
// Run (per-agent isolation -- ALWAYS pass -o build/embed_t8, CONVENTIONS.md):
//   xmake f -p windows -a x64 -m debug -y -o build/embed_t8 \
//           --lc_fallback_backend=true --lc_embree_path=src/ext/HIPRT/contrib/embree
//   xmake build -y test_embed_retrieval
//   ./bin/debug/test_embed_retrieval.exe dx        (first run builds the cache)
//   ./bin/debug/test_embed_retrieval.exe dx        (second run = warm-cache leg)
//   ./bin/debug/test_embed_retrieval.exe vk
//   xmake run -o build/embed_t8 test_embed_retrieval fallback
//
// Fixtures (FROZEN -- never regenerated to pass, plan S1.1) under
// src/tests/data/embedding/retrieval/ (see its README.md; consumed fields and
// layout were verified against the committed qrels.json / *.bin / *.jsonl):
//   corpus.jsonl  5000 {"id":"pNNNN","text":...}  (LF, raw text, no control chars)
//   queries.jsonl  200 {"id":"qNNN", "text":...}
//   golden_dense_scores.fp32.bin  200x5000 f32-le row-major [query][passage]
//   golden_sparse_scores.fp32.bin 200x5000 f32-le row-major [query][passage]
//   qrels.json    per query: dense_top10 / dense_top10_scores / dense_tie_groups,
//                 sparse_top10 / sparse_top10_scores / sparse_tie_groups, gold_ids,
//                 query_id / query_lang / query_src;  meta: tie_eps, top_k,
//                 max_seq_len, counts.queries_per_lang.xx_gibberish[_candidates],
//                 negative_controls.{gibberish_criterion.measured, gibberish_queries,
//                 duplicate_passages}, tie_stats.
// Model: D:/compute/bge-m3/bge-m3/model.bgm ; tokenizer.json under the parent data dir.
//
// Engine-produced cache files (owned by THIS test, written at review time -- NOT
// shipped with the frozen fixtures, README.md line "deliberately not shipped"):
//   corpus_embeddings.fp32.bin  5000x1024 f32-le (the engine's dense CLS rows,
//                               plan S4.3 "corpus_embeddings cached so CI does not
//                               re-encode 5k passages per run")
//   corpus_sparse.json          {"pNNNN":{"<token_id>":weight,...},...} derived from
//                               Embedding.sparse (specials dropped, id>2, w>0).
// The test loads these when present and no --rebuild-cache token is given (CI
// discipline); it ALWAYS re-encodes a 20-passage subset and asserts it matches the
// cache on disk (freshness gate -- catches a stale corpus.jsonl behind a warm cache).
//
// ---------------------------------------------------------------------------
// TIE HANDLING -- implemented EXACTLY as plan S4.3 step 3. qrels tie_eps = 1e-5,
// top_k = 10. A golden top-10 whose entries fall in a documented tie group
// (dense_tie_groups / sparse_tie_groups, |d score| < 1e-5) is compared UNORDERED
// within the group; everything else positionally. The test canonicalises each
// ranking by sorting the members of every tie group ascending (ids are zero-padded
// pNNNN so id-ascending == passage-index-ascending == the frozen ranking's own
// within-tie order), then compares sequences -- this is literally "pairs with
// |d score| < 1e-5 compared unordered". Tie-pair counts are REPORTED per query
// (LUISA_INFO) and totalled; the frozen qrels report 6 dense tie-groups / 6 pairs
// and 11 sparse tie-groups / 79 pairs, so any engine run reporting materially
// fewer/more groups is a red flag surfaced in the log.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// CROSS-BACKEND ORACLE (plan S1.3/S6). The test prints a stable FNV-1a-64 checksum
// of the 200 query dense embeddings and dumps them to
//   <temp>/embed_retrieval_<backend>_query_dense.bin  (200x1024 f32-le)
// Reviewers compare backends by running the SAME binary on two backends and
// diffing the printed checksum / the per-backend dump (plan S6 cross-backend
// numerical policy: GPU-vs-GPU within fp op-order noise, cosine >= 0.99999;
// the fallback dump is the deterministic oracle). The checksum must be stable
// per-backend across warm/cold runs; it is NOT required to be identical across
// backends (that difference IS the cross-backend oracle signal).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 65535-DISPATCH-CAP WORKAROUND (known open engine bug, being fixed in
// parallel). On dx a single atomic batch job aborts when any 1D kernel grid
// exceeds 65535 groups. The attention softmax dispatches B*num_heads*S row
// groups (num_heads = 16); a padded bucket of 512 therefore aborts once
// B*S*16 > 65535, i.e. B*512 > 4095 -> B >= 8. (T3 measured 8 x bucket-512 =
// 65536 groups -> "Dispatch size ... out of range 65535".) The engine has no
// 2D-flatten/clamp for this path yet. WORKAROUND here: on EVERY GPU backend
// each encode_batch job is chunked to kGpuChunkPassages = 7 (7*16*512 = 57344
// <= 65535 for any bucket <= 512), logged loudly as "chunked under the
// <backend> 65535-cap". OBSERVED (this RTX 4080 SUPER): vk is NOT exempt -- a
// bucket-512 batch of 64 passages died "vkQueueSubmit ... ERROR_DEVICE_LOST",
// the same 1D-group overflow surfacing as a device loss instead of an explicit
// abort; so vk/cuda are chunked like dx. Only the CPU fallback (no 1D group
// limit) uses a larger kFallbackChunkPassages = 64, and that split is a host
// memory/queue bound (a 5000-row atomic batch would request a ~80 GB
// [B,heads,S,S] logits workspace and overflow the scheduler queue), NOT the
// 65535 workaround. If a bucket-512 batch of 8 still trips on any backend,
// lower the chunk here and record the observed threshold.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HYBRID LEG (plan S4.3 step 5). The standard bge-m3 hybrid is dense + sparse
// with EQUAL WEIGHTS here (documented: this is a deliberate choice, NOT the
// bge-m3 reference alpha=0.5 the retrieval README's "hybrid is derivable as
// dense + alpha*sparse" note mentions; equal weights make the leg independent of
// a tunable alpha and the comparison is reference-vs-engine on the SAME formula
// so the alpha value cancels). There is NO hybrid field in qrels, so this leg's
// reference is computed IN-TEST by summing the two FROZEN score matrices
// (golden_dense_scores + golden_sparse_scores from the .bin files) -- it is NOT
// re-derived from engine embeddings. That keeps the frozen-reference discipline:
// the hybrid baseline comes only from committed golden data; only the thing being
// scored (engine hybrid = engine dense cosines + engine sparse dots) is live.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// NEGATIVE CONTROLS (plan S4.3) -- DOCUMENTED DEVIATION, reported to parent.
// Plan S4.3 literally says a gibberish query's max score must be "below the
// median golden score". The retrieval README explicitly records that the FROZEN
// PyTorch reference does NOT satisfy that literal bound on real multilingual
// data (meta.negative_controls.gibberish_criterion.literal_holds == false:
// median_dense_all ~ 0.243 < gibberish max_dense ~ 0.44-0.45, because bge-m3
// scores unrelated passage pairs below a noise string's best match). Per plan
// S1.1/S8(b) the goldens are frozen and are never regenerated to make a test
// pass, so this test operationalises the control exactly as the README prescribes:
//   (a) the ENGINE's gibberish row must MATCH the frozen golden max_dense /
//       max_sparse values within tolerance, AND
//   (b) it must stay below the real-query top-1 score distribution recorded in
//       the frozen meta (engine gibberish max_dense < real_query_top1_p25 ~ 0.598).
// The literal "max < whole-matrix median" bound is COMPUTED AND REPORTED as an
// informational number (it fails for the reference too) but is NOT a hard gate.
// The duplicate-passage control IS present in the frozen corpus (p4997==p0140,
// p4998==p2827, p4999==p2837 per README) and is gated: the engine must give the
// two copies near-identical dense embeddings and adjacent positions in every
// full ranking (the ascending-index tie-break decides which comes first; at the
// top-10 cut one member can be dropped -- the documented tie-break, not a dedup).
// ---------------------------------------------------------------------------
// Gates (plan S4.3 + task): dense / sparse / hybrid ranking parity >= 99% exact
// top-10 modulo ties; Recall@10 == 1.0 modulo documented ties (mean >= 0.99);
// NDCG@10 drop vs the frozen golden-dense baseline <= 0.005; negative controls
// (gibberish a+b, duplicate adjacency) pass. accumulate-then-expect-once with
// per-leg failure counts reported via LUISA_INFO.
#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <luisa/core/clock.h>
#include <luisa/core/fiber.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/memory.h>// luisa::span (= std::span)
#include <luisa/core/stl/string.h>// luisa::string_view (= std::string_view)
#include <luisa/core/stl/vector.h>

#include <yyjson.h>

#include "embedding/embed/session.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::embed;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// -------------------------------------------------------------- constants
constexpr size_t kDenseDim = 1024u;
constexpr size_t kCorpusN = 5000u;
constexpr size_t kQueryN = 200u;
constexpr uint32_t kHeads = 16u;// bge-m3 XLM-R attention heads (softmax row factor)

// GPU backends (dx/vk/cuda) on this driver all enforce a 1D workgroup/group cap:
// the attention softmax dispatches B*num_heads*S row groups (num_heads = 16), so
// any padded bucket <= 512 must keep B*16*S <= 65535 -> B <= 8.006 at S=512 -> B=7
// (8 would be 65536 > the cap). OBSERVED: dx aborts "Dispatch size ... out of range
// 65535"; vk on this RTX 4080 SUPER dies "vkQueueSubmit ... ERROR_DEVICE_LOST" for
// the same oversized batch (chunk 64). So GPU chunking is applied to every GPU
// backend, not just dx. (This contradicts the task's "vk is exempt" assumption and
// feeds evidence to the engine fix.)
constexpr size_t kGpuChunkPassages = 7u;
// CPU fallback has no 1D dispatch-group limit; a larger chunk only bounds the host
// padding/compute per atomic job (memory/queue measure, NOT a group-cap workaround).
constexpr size_t kFallbackChunkPassages = 64u;

// -------------------------------------------------------------- file io
std::vector<char> read_file(const std::filesystem::path &p) {
    if (std::FILE *f = std::fopen(p.string().c_str(), "rb"); f != nullptr) {
        std::vector<char> data;
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

std::vector<float> read_f32_file(const std::filesystem::path &p) {
    auto raw = read_file(p);
    std::vector<float> out(raw.size() / 4u);
    std::memcpy(out.data(), raw.data(), out.size() * 4u);
    return out;
}

void write_f32_file(const std::filesystem::path &p, const std::vector<float> &v) {
    if (std::FILE *f = std::fopen(p.string().c_str(), "wb"); f != nullptr) {
        if (!v.empty()) {
            std::fwrite(v.data(), 4u, v.size(), f);
        }
        std::fclose(f);
    } else {
        LUISA_ERROR("cannot open cache file for writing \"{}\"", p.string());
    }
}

// corpus_embeddings / golden matrices are f32 little-endian row-major with no
// header (README "matrix" definition). x86 host is LE, so a raw memcpy is the
// documented-correct read (same idiom as test_embed_e2e::read_f32_file).

// --------------------------------------------------------- JSONL ("id","text")
// corpus.jsonl / queries.jsonl are UTF-8 LF, one JSON object per line
// {"id": ..., "text": ...}; yyjson decodes the JSON string escapes, so the
// raw text is used verbatim (README: whitespace collapsed, no control chars).
void parse_jsonl_id_text(const std::vector<char> &raw,
                         std::vector<std::string> &ids,
                         std::vector<std::string> &texts) {
    const char *data = raw.data();
    const size_t len = raw.size();
    size_t start = 0u;
    auto line = [&](size_t s, size_t e) {
        while (e > s && data[e - 1u] == '\r') {
            e--;
        }
        if (e <= s) {
            return;
        }
        // yyjson_read is 3-arg (const buffer, no error out); NULL => malformed
        // line. Each JSONL object is valid JSON so this only trips on real corruption.
        yyjson_doc *doc = yyjson_read(data + s, e - s, 0u);
        if (doc == nullptr) {
            LUISA_ERROR("JSONL parse failed at byte {}", s);
        }
        const auto *root = yyjson_doc_get_root(doc);
        const auto *idv = yyjson_obj_get(root, "id");
        const auto *tv = yyjson_obj_get(root, "text");
        if (!yyjson_is_str(idv) || !yyjson_is_str(tv)) {
            yyjson_doc_free(doc);
            LUISA_ERROR("JSONL object at byte {} missing string \"id\"/\"text\"", s);
        }
        ids.emplace_back(yyjson_get_str(idv));
        texts.emplace_back(yyjson_get_str(tv));
        yyjson_doc_free(doc);
    };
    for (size_t i = 0u; i < len; i++) {
        if (data[i] == '\n') {
            line(start, i);
            start = i + 1u;
        }
    }
    if (start < len) {
        line(start, len);
    }
}

// "p0140" -> 140, "q199" -> 199 (zero-padded ids == index, verified against the
// corpus/queries files in main).
int32_t id_to_index(std::string_view id) {
    if (id.size() < 2u || (id[0] != 'p' && id[0] != 'q')) {
        return -1;
    }
    int32_t v = 0;
    for (size_t i = 1u; i < id.size(); i++) {
        if (id[i] < '0' || id[i] > '9') {
            return -1;
        }
        v = v * 10 + (id[i] - '0');
    }
    return v;
}

// join a string vector for logging (no luisa::join in this repo)
std::string str_join(const std::vector<std::string> &v, const char *sep = ",") {
    std::string s;
    for (size_t i = 0; i < v.size(); i++) {
        if (i != 0u) {
            s += sep;
        }
        s += v[i];
    }
    return s;
}

// ------------------------------------------------------------- numerics
double cosine(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) {
        return -1.0;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

uint64_t fnv1a64(const void *data, size_t len) {
    // FNV-1a 64-bit over the raw bytes.
    uint64_t h = 0xcbf29ce484222325ull;
    const auto *b = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < len; i++) {
        h ^= b[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

// indices sorted by (-score, index ascending) -- the frozen ranking convention
// (README "ranking: sort by (-score, passage index ascending)").
std::vector<uint32_t> topk_indices(const float *scores, size_t n, size_t k) {
    std::vector<uint32_t> idx(n);
    for (size_t i = 0; i < n; i++) {
        idx[i] = static_cast<uint32_t>(i);
    }
    std::stable_sort(idx.begin(), idx.end(),
                     [&](uint32_t a, uint32_t b) {
                         const double sa = scores[a], sb = scores[b];
                         if (sa != sb) {
                             return sa > sb;
                         }
                         return a < b;
                     });
    idx.resize(std::min(k, n));
    return idx;
}

// full rank (0-based) of `target` in the same (-score, idx asc) order, WITHOUT
// materialising the whole permutation -- retained for potential diagnostics; the
// duplicate control below uses a tie-aware separator count instead (robust to the
// engine's ~1e-6 fp difference between two chunk-built copies of the same passage).
[[maybe_unused]] int32_t rank_of(const float *scores, size_t n, uint32_t target) {
    const double st = scores[target];
    int32_t r = 0;
    for (size_t j = 0; j < n; j++) {
        if (j == target) {
            continue;
        }
        const double s = scores[j];
        if (s > st || (s == st && static_cast<uint32_t>(j) < target)) {
            r++;
        }
    }
    return r;
}

// ------------------------------------------------------------- qrels model
struct QrelQuery {
    std::string query_id;
    std::string query_lang;
    std::string query_src;
    std::vector<std::string> dense_top10;
    std::vector<float> dense_top10_scores;
    std::vector<std::vector<std::string>> dense_tie_groups;
    std::vector<std::string> sparse_top10;
    std::vector<float> sparse_top10_scores;
    std::vector<std::vector<std::string>> sparse_tie_groups;
    std::vector<std::string> gold_ids;
};

struct QrelsMeta {
    float tie_eps = 1e-5f;
    uint32_t top_k = 10u;
    uint32_t max_seq_len = 512u;
    uint32_t gibberish_candidates = 0u;
    float median_dense_all = 0.0f;
    float real_query_top1_p25 = 0.0f;
    std::vector<std::pair<std::string, std::string>> duplicate_pairs;
    // query_id -> (max_dense, max_sparse) frozen reference for the gibberish rows
    std::unordered_map<std::string, std::pair<float, float>> gibberish_ref;
};

struct Qrels {
    QrelsMeta meta;
    std::vector<QrelQuery> queries;
};

// read a JSON array of strings into out.
void read_str_arr(yyjson_val *arr, std::vector<std::string> &out) {
    if (!yyjson_is_arr(arr)) {
        LUISA_ERROR("expected string array");
    }
    yyjson_val *item;
    yyjson_arr_iter it = yyjson_arr_iter_with(arr);
    while ((item = yyjson_arr_iter_next(&it)) != nullptr) {
        out.emplace_back(yyjson_get_str(item));
    }
}

void read_float_arr(yyjson_val *arr, std::vector<float> &out) {
    if (!yyjson_is_arr(arr)) {
        LUISA_ERROR("expected float array");
    }
    yyjson_val *item;
    yyjson_arr_iter it = yyjson_arr_iter_with(arr);
    while ((item = yyjson_arr_iter_next(&it)) != nullptr) {
        out.push_back(static_cast<float>(yyjson_get_num(item)));
    }
}

Qrels load_qrels(const std::filesystem::path &p) {
    Qrels q;
    yyjson_read_err err;
    yyjson_doc *doc = yyjson_read_file(p.string().c_str(), 0u, nullptr, &err);
    if (doc == nullptr) {
        LUISA_ERROR("cannot read qrels.json: {} (pos {})",
                    (err.msg ? err.msg : "?"), err.pos);
    }
    const auto *root = yyjson_doc_get_root(doc);

    // ---- meta
    const auto *meta = yyjson_obj_get(root, "meta");
    q.meta.tie_eps = static_cast<float>(yyjson_get_num(yyjson_obj_get(meta, "tie_eps")));
    q.meta.top_k = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(meta, "top_k")));
    q.meta.max_seq_len =
        static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(meta, "max_seq_len")));
    if (const auto *counts = yyjson_obj_get(meta, "counts"); yyjson_is_obj(counts)) {
        if (const auto *qpl = yyjson_obj_get(counts, "queries_per_lang"); yyjson_is_obj(qpl)) {
            if (const auto *c = yyjson_obj_get(qpl, "xx_gibberish_candidates"); c != nullptr) {
                q.meta.gibberish_candidates = static_cast<uint32_t>(yyjson_get_uint(c));
            }
        }
    }
    if (const auto *nc = yyjson_obj_get(meta, "negative_controls"); yyjson_is_obj(nc)) {
        if (const auto *crit = yyjson_obj_get(nc, "gibberish_criterion"); yyjson_is_obj(crit)) {
            if (const auto *meas = yyjson_obj_get(crit, "measured"); yyjson_is_obj(meas)) {
                q.meta.median_dense_all =
                    static_cast<float>(yyjson_get_num(yyjson_obj_get(meas, "median_dense_all")));
                q.meta.real_query_top1_p25 =
                    static_cast<float>(yyjson_get_num(yyjson_obj_get(meas, "real_query_top1_p25")));
            }
        }
        if (const auto *gq = yyjson_obj_get(nc, "gibberish_queries"); yyjson_is_arr(gq)) {
            yyjson_val *item;
            yyjson_arr_iter it = yyjson_arr_iter_with(gq);
            while ((item = yyjson_arr_iter_next(&it)) != nullptr) {
                const auto *qid = yyjson_get_str(yyjson_obj_get(item, "query_id"));
                const auto md =
                    static_cast<float>(yyjson_get_num(yyjson_obj_get(item, "max_dense")));
                const auto ms =
                    static_cast<float>(yyjson_get_num(yyjson_obj_get(item, "max_sparse")));
                q.meta.gibberish_ref[qid] = {md, ms};
            }
        }
        if (const auto *dp = yyjson_obj_get(nc, "duplicate_passages"); yyjson_is_arr(dp)) {
            yyjson_val *item;
            yyjson_arr_iter it = yyjson_arr_iter_with(dp);
            while ((item = yyjson_arr_iter_next(&it)) != nullptr) {
                const auto *pair = yyjson_obj_get(item, "pair");
                if (yyjson_is_arr(pair) && yyjson_arr_size(pair) == 2u) {
                    const auto *a = yyjson_get_str(yyjson_arr_get(pair, 0));
                    const auto *b = yyjson_get_str(yyjson_arr_get(pair, 1));
                    q.meta.duplicate_pairs.emplace_back(a, b);
                }
            }
        }
    }

    // ---- queries
    const auto *queries = yyjson_obj_get(root, "queries");
    if (!yyjson_is_arr(queries)) {
        yyjson_doc_free(doc);
        LUISA_ERROR("qrels.json has no \"queries\" array");
    }
    yyjson_val *item;
    yyjson_arr_iter it = yyjson_arr_iter_with(queries);
    while ((item = yyjson_arr_iter_next(&it)) != nullptr) {
        QrelQuery qq;
        qq.query_id = yyjson_get_str(yyjson_obj_get(item, "query_id"));
        if (const auto *v = yyjson_obj_get(item, "query_lang"); v != nullptr) {
            qq.query_lang = yyjson_get_str(v);
        }
        if (const auto *v = yyjson_obj_get(item, "query_src"); v != nullptr) {
            qq.query_src = yyjson_get_str(v);
        }
        read_str_arr(yyjson_obj_get(item, "dense_top10"), qq.dense_top10);
        read_float_arr(yyjson_obj_get(item, "dense_top10_scores"), qq.dense_top10_scores);
        read_str_arr(yyjson_obj_get(item, "sparse_top10"), qq.sparse_top10);
        read_float_arr(yyjson_obj_get(item, "sparse_top10_scores"), qq.sparse_top10_scores);
        read_str_arr(yyjson_obj_get(item, "gold_ids"), qq.gold_ids);
        const auto parse_groups = [&](yyjson_val *garr,
                                      std::vector<std::vector<std::string>> &dst) {
            if (!yyjson_is_arr(garr)) {
                return;
            }
            yyjson_val *sub;
            yyjson_arr_iter sit = yyjson_arr_iter_with(garr);
            while ((sub = yyjson_arr_iter_next(&sit)) != nullptr) {
                std::vector<std::string> grp;
                read_str_arr(sub, grp);
                if (grp.size() >= 2u) {
                    dst.push_back(std::move(grp));
                }
            }
        };
        parse_groups(yyjson_obj_get(item, "dense_tie_groups"), qq.dense_tie_groups);
        parse_groups(yyjson_obj_get(item, "sparse_tie_groups"), qq.sparse_tie_groups);
        q.queries.push_back(std::move(qq));
    }
    yyjson_doc_free(doc);
    return q;
}

// ------------------------------------------------- tie canonicalisation (S4.3 step 3)
// Map every tie-group member id -> group index, used to canonicalise a ranking.
std::unordered_map<std::string, int> build_group_of(
    const std::vector<std::vector<std::string>> &groups) {
    std::unordered_map<std::string, int> g;
    for (size_t i = 0; i < groups.size(); i++) {
        for (const auto &id : groups[i]) {
            g[id] = static_cast<int>(i);
        }
    }
    return g;
}

// Replace every maximal contiguous run of same-group ids by that run sorted
// ascending; ids outside any group pass through. This makes "unordered within a
// tie group" comparable as a single deterministic sequence.
std::vector<std::string> canonicalize(std::vector<std::string> seq,
                                      const std::unordered_map<std::string, int> &group_of) {
    std::vector<std::string> out;
    out.reserve(seq.size());
    size_t i = 0u;
    while (i < seq.size()) {
        const auto it = group_of.find(seq[i]);
        if (it == group_of.end()) {
            out.push_back(seq[i]);
            i++;
            continue;
        }
        const int g = it->second;
        size_t j = i;
        std::vector<std::string> block;
        while (j < seq.size()) {
            const auto k = group_of.find(seq[j]);
            if (k != group_of.end() && k->second == g) {
                block.push_back(seq[j]);
                j++;
            } else {
                break;
            }
        }
        std::sort(block.begin(), block.end());
        out.insert(out.end(), block.begin(), block.end());
        i = j;
    }
    return out;
}

// count of tie groups actually touched (present in both) for reporting
size_t count_tie_pairs(const std::vector<std::vector<std::string>> &groups,
                       const std::unordered_set<std::string> &engine_set) {
    size_t n = 0u;
    for (const auto &g : groups) {
        bool any = false;
        for (const auto &id : g) {
            any = any || engine_set.count(id) != 0u;
        }
        if (any) {
            n += g.size() >= 2u ? (g.size() * (g.size() - 1u) / 2u) : 0u;
        }
    }
    return n;
}

// ------------------------------------------------- hybrid in-test tie groups
// Merge consecutive positions of the reference top-10 whose scores are within
// tie_eps into an unordered group (no qrels hybrid field exists). `ref_row` is
// the reference-hybrid score row for this query (length kCorpusN), NOT the full
// matrix -- ref_idx are passage indices into that row.
std::vector<std::vector<std::string>> hybrid_tie_groups(const std::vector<uint32_t> &ref_idx,
                                                        const std::vector<std::string> &corpus_ids,
                                                        const float *ref_row,
                                                        float tie_eps) {
    std::vector<std::vector<std::string>> groups;
    size_t i = 0u;
    while (i < ref_idx.size()) {
        size_t j = i + 1u;
        while (j < ref_idx.size() &&
               std::fabs(ref_row[ref_idx[j]] - ref_row[ref_idx[j - 1u]]) < tie_eps) {
            j++;
        }
        if (j - i > 1u) {
            std::vector<std::string> grp;
            for (size_t m = i; m < j; m++) {
                grp.push_back(corpus_ids[ref_idx[m]]);
            }
            groups.push_back(std::move(grp));
        }
        i = j;
    }
    return groups;
}

// --------------------------------------------------- corpus_sparse cache format
void pool_sparse_inplace(std::vector<std::pair<uint32_t, float>> &v);
void write_corpus_sparse(const std::filesystem::path &p,
                         const std::vector<std::string> &corpus_ids,
                         const std::vector<std::vector<std::pair<uint32_t, float>>> &sparse) {
    std::FILE *f = std::fopen(p.string().c_str(), "wb");
    if (f == nullptr) {
        LUISA_ERROR("cannot open corpus_sparse.json for writing \"{}\"", p.string());
    }
    std::fputc('{', f);
    for (size_t i = 0; i < sparse.size(); i++) {
        if (i != 0u) {
            std::fputc(',', f);
        }
        std::fprintf(f, "\"%s\":{", corpus_ids[i].c_str());
        const auto &v = sparse[i];
        // emit ascending by token id for a stable, reviewable cache
        std::vector<std::pair<uint32_t, float>> sorted = v;
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });
        for (size_t k = 0; k < sorted.size(); k++) {
            if (k != 0u) {
                std::fputc(',', f);
            }
            std::fprintf(f, "\"%u\":%.9g", sorted[k].first, static_cast<double>(sorted[k].second));
        }
        std::fputc('}', f);
    }
    std::fputc('}', f);
    std::fclose(f);
}

void read_corpus_sparse(const std::filesystem::path &p,
                        const std::unordered_map<std::string, uint32_t> &id_to_idx,
                        size_t n,
                        std::vector<std::vector<std::pair<uint32_t, float>>> &sparse) {
    sparse.assign(n, {});
    yyjson_read_err err;
    yyjson_doc *doc = yyjson_read_file(p.string().c_str(), 0u, nullptr, &err);
    if (doc == nullptr) {
        LUISA_ERROR("cannot read corpus_sparse.json: {} (pos {})",
                    (err.msg ? err.msg : "?"), err.pos);
    }
    const auto *root = yyjson_doc_get_root(doc);
    yyjson_obj_iter it = yyjson_obj_iter_with(root);
    yyjson_val *key;
    while ((key = yyjson_obj_iter_next(&it)) != nullptr) {
        const auto *ks = yyjson_get_str(key);
        const auto *val = yyjson_obj_iter_get_val(key);
        const auto found = id_to_idx.find(ks);
        if (found == id_to_idx.end()) {
            continue;
        }
        const uint32_t idx = found->second;
        std::vector<std::pair<uint32_t, float>> pairs;
        if (yyjson_is_obj(val)) {
            yyjson_obj_iter vit = yyjson_obj_iter_with(val);
            yyjson_val *tk;
            while ((tk = yyjson_obj_iter_next(&vit)) != nullptr) {
                const auto *tks = yyjson_get_str(tk);
                const auto *wv = yyjson_obj_iter_get_val(tk);
                uint32_t tid = static_cast<uint32_t>(std::strtoul(tks, nullptr, 10));
                pairs.emplace_back(tid, static_cast<float>(yyjson_get_num(wv)));
            }
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });
        // defensive: normalise to the reference max-per-token-id semantics even if a
        // stale/unpooled cache (duplicate token-id keys, which yyjson collapses to the
        // LAST value, not the max) were loaded.
        pool_sparse_inplace(pairs);
        sparse[idx] = std::move(pairs);
    }
    yyjson_doc_free(doc);
}

// The frozen reference max-pools the sparse head per token id (README "sparse":
// "one lexical weight per token id ... max-pooled per token id inside a document").
// The engine's Embedding.sparse emits one (token_id, weight) per OCCURRENCE (specials
// already dropped, weight>0), so it must be collapsed here or the shared-token dot
// double-counts repeated tokens and the sparse ranking is inflated/wrong. Drop any
// id<=2 defensively (matches the reference id>2 filter) and keep the MAX weight per id.
std::vector<std::pair<uint32_t, float>> pool_sparse(
    const luisa::vector<std::pair<uint32_t, float>> &raw) {
    std::unordered_map<uint32_t, float> m;
    for (const auto &pr : raw) {
        if (pr.first <= 2u || !(pr.second > 0.0f)) {
            continue;
        }
        const auto it = m.find(pr.first);
        if (it == m.end() || it->second < pr.second) {
            m[pr.first] = pr.second;
        }
    }
    std::vector<std::pair<uint32_t, float>> out;
    out.reserve(m.size());
    for (const auto &kv : m) {
        out.emplace_back(kv.first, kv.second);
    }
    std::sort(out.begin(), out.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    return out;
}

// pool an already-materialised (possibly duplicate-keyed) pair vector (idempotent).
void pool_sparse_inplace(std::vector<std::pair<uint32_t, float>> &v) {
    std::unordered_map<uint32_t, float> m;
    for (const auto &pr : v) {
        if (pr.first <= 2u || !(pr.second > 0.0f)) {
            continue;
        }
        const auto it = m.find(pr.first);
        if (it == m.end() || it->second < pr.second) {
            m[pr.first] = pr.second;
        }
    }
    v.clear();
    v.reserve(m.size());
    for (const auto &kv : m) {
        v.emplace_back(kv.first, kv.second);
    }
    std::sort(v.begin(), v.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
}

// -------------------------------------------------------- chunked encode_batch
// Encode `texts` (corpus order) through the engine with want_sparse, appending
// 1024-float dense rows to `dense` and one sparse-pair vector per text to
// `sparse_out`. Chunked per backend (see file header for the dx cap math).
void build_embeddings(Session &session, const std::vector<std::string> &texts, bool is_gpu,
                      size_t &jobs, size_t &rows, std::vector<float> &dense,
                      std::vector<std::vector<std::pair<uint32_t, float>>> &sparse_out) {
    const size_t chunk = is_gpu ? kGpuChunkPassages : kFallbackChunkPassages;
    EncodeOptions opt;
    opt.max_seq_len = 512u;// plan S4.3 / README truncation: any other value changes the reference
    opt.want_sparse = true;
    opt.want_colbert = false;
    jobs = 0u;
    rows = 0u;
    const size_t N = texts.size();
    for (size_t start = 0u; start < N; start += chunk) {
        const size_t end = std::min(start + chunk, N);
        std::vector<std::string_view> views;
        views.reserve(end - start);
        for (size_t i = start; i < end; i++) {
            views.emplace_back(std::string_view{texts[i]});
        }
        auto batch = session.encode_batch(
            luisa::span<const luisa::string_view>{views.data(), views.size()}, opt);
        if (batch.size() != end - start) {
            LUISA_ERROR("encode_batch returned {} embeddings for {} texts (chunk @ {})",
                        batch.size(), end - start, start);
        }
        for (auto &emb : batch) {
            if (emb.dense.size() != kDenseDim) {
                LUISA_ERROR("dense row size {} != {}", emb.dense.size(), kDenseDim);
            }
            dense.insert(dense.end(), emb.dense.begin(), emb.dense.end());
            sparse_out.push_back(pool_sparse(emb.sparse));
        }
        jobs++;
        rows += end - start;
    }
}

}// namespace

int main(int argc, char *argv[]) {
    luisa::fiber::scheduler fiber_scheduler;

    auto [context, device] = luisa::test::create_device(argc, argv);
    luisa::test::log_test_backend(argv[1], device);
    const char *backend = argv[1];
    const bool is_fallback = std::strcmp(backend, "fallback") == 0;
    const bool is_gpu = !is_fallback;// dx/vk/cuda all enforce the ~65535 1D group cap here

    // scan optional args: --rebuild-cache token + optional .bgm package path
    bool rebuild = false;
    std::filesystem::path bgm;
    for (int i = 2; i < argc; i++) {
        const std::string a{argv[i]};
        if (a == "--rebuild-cache") {
            rebuild = true;
        } else if (a.size() >= 4u && a.compare(a.size() - 4u, 4u, ".bgm") == 0) {
            bgm = std::filesystem::path{a};
        }
    }
    if (bgm.empty()) {
        bgm = std::filesystem::path{"D:/compute/bge-m3/bge-m3/model.bgm"};
    }

    const std::filesystem::path data_dir{LUISA_EMBED_TEST_DATA_DIR};
    const std::filesystem::path ret_dir = data_dir / "retrieval";
    const std::filesystem::path tok_json = data_dir / "tokenizer.json";
    const std::filesystem::path cache_dense = ret_dir / "corpus_embeddings.fp32.bin";
    const std::filesystem::path cache_sparse = ret_dir / "corpus_sparse.json";

    luisa::compute::Clock total_clock;

    // ============================================================ fixtures (frozen)
    const auto corpus_raw = read_file(ret_dir / "corpus.jsonl");
    const auto queries_raw = read_file(ret_dir / "queries.jsonl");
    std::vector<std::string> corpus_ids, corpus_texts;
    std::vector<std::string> query_ids, query_texts;
    parse_jsonl_id_text(corpus_raw, corpus_ids, corpus_texts);
    parse_jsonl_id_text(queries_raw, query_ids, query_texts);
    expect(corpus_ids.size() == kCorpusN) << "corpus.jsonl has 5000 rows";
    expect(query_ids.size() == kQueryN) << "queries.jsonl has 200 rows";
    // id string == zero-padded index (verified so id<->index mapping is trusted)
    {
        bool ok = true;
        for (size_t i = 0; i < corpus_ids.size(); i++) {
            ok = ok && (id_to_index(corpus_ids[i]) == static_cast<int32_t>(i));
        }
        for (size_t i = 0; i < query_ids.size(); i++) {
            ok = ok && (id_to_index(query_ids[i]) == static_cast<int32_t>(i));
        }
        expect(ok) << "corpus/query ids are pNNNN / qNNN matching their line index";
    }
    std::unordered_map<std::string, uint32_t> pid_to_idx;
    for (uint32_t i = 0; i < corpus_ids.size(); i++) {
        pid_to_idx[corpus_ids[i]] = i;
    }

    const auto qrels = load_qrels(ret_dir / "qrels.json");
    expect(qrels.queries.size() == kQueryN) << "qrels.json has 200 queries";
    const float tie_eps = qrels.meta.tie_eps;
    const uint32_t top_k = qrels.meta.top_k;
    const std::vector<float> golden_dense = read_f32_file(ret_dir / "golden_dense_scores.fp32.bin");
    const std::vector<float> golden_sparse = read_f32_file(ret_dir / "golden_sparse_scores.fp32.bin");
    expect(golden_dense.size() == kQueryN * kCorpusN) << "golden_dense_scores shape 200x5000";
    expect(golden_sparse.size() == kQueryN * kCorpusN) << "golden_sparse_scores shape 200x5000";

    LUISA_INFO("L3 retrieval: backend={} corpus={} queries={} top_k={} tie_eps={} max_seq_len={} "
               "rebuild_cache={}",
               backend, corpus_ids.size(), query_ids.size(), top_k, tie_eps,
               qrels.meta.max_seq_len, rebuild);

    // ============================================================ session
    luisa::compute::Clock create_clock;
    auto session = Session::create(device, bgm, tok_json, 2u);
    LUISA_INFO("Session created in {:.1f} ms (package load + weight upload + kernel compile)",
               create_clock.toc());

    // ============================================================ LEG 1: corpus build
    // corpus_embeddings.fp32.bin + corpus_sparse.json cache (load when warm & no
    // --rebuild-cache; ALWAYS run a 20-passage freshness re-encode gate).
    std::vector<float> corpus_emb;
    std::vector<std::vector<std::pair<uint32_t, float>>> corpus_sparse;
    {
        const bool cache_present = std::filesystem::exists(cache_dense) &&
                                   std::filesystem::exists(cache_sparse);
        const size_t want_floats = kCorpusN * kDenseDim;
        std::error_code ec;
        const auto dsize = std::filesystem::file_size(cache_dense, ec);
        const bool cache_valid_size =
            cache_present && !ec && dsize == want_floats * 4u;

        if (cache_present && !rebuild && cache_valid_size) {
            LUISA_INFO("LEG1 cache: loading WARM corpus cache from disk ({} + {})",
                       cache_dense.filename().string(), cache_sparse.filename().string());
            corpus_emb = read_f32_file(cache_dense);
            expect(corpus_emb.size() == want_floats) << "cache dense size 5000x1024";
            read_corpus_sparse(cache_sparse, pid_to_idx, kCorpusN, corpus_sparse);
            expect(corpus_sparse.size() == kCorpusN) << "cache sparse passage count";
        } else {
            if (rebuild && cache_present) {
                LUISA_INFO("LEG1 cache: --rebuild-cache token present, forcing COLD rebuild");
            } else if (!cache_present) {
                LUISA_INFO("LEG1 cache: no cache on disk, building COLD (first run). "
                           "Expected ~70 s GPU encode on dx + model upload.");
            } else {
                LUISA_INFO("LEG1 cache: existing cache size mismatch, rebuilding COLD");
            }
            if (is_gpu) {
                LUISA_INFO("LEG1 cache: !!! chunked under the {} 65535-cap -- encode_batch jobs "
                           "capped to {} passages each (softmax B*16*S <= 65535; a bucket-512 "
                           "job with B>=8 aborts). WORKAROUND for the 1D-grid cap; OBSERVED on "
                           "both dx (Dispatch-size abort) and vk (ERROR_DEVICE_LOST) on this GPU.",
                           backend, kGpuChunkPassages);
            } else {
                LUISA_INFO("LEG1 cache: fallback (CPU) build chunk = {} passages (host memory/"
                           "queue bound only; CPU fallback has NO 1D dispatch-group limit).",
                           kFallbackChunkPassages);
            }
            corpus_emb.clear();
            corpus_emb.reserve(want_floats);
            corpus_sparse.clear();
            size_t jobs = 0u, rows = 0u;
            luisa::compute::Clock clock;
            build_embeddings(*session, corpus_texts, is_gpu, jobs, rows, corpus_emb, corpus_sparse);
            const double ms = clock.toc();
            LUISA_INFO("LEG1 cache: built COLD {} passages in {} atomic batch jobs in {:.1f} ms "
                       "({:.2f} ms/passage, {:.0f} passages/s)",
                       rows, jobs, ms, ms / static_cast<double>(rows),
                       static_cast<double>(rows) / (ms / 1000.0));
            expect(corpus_emb.size() == want_floats) << "built dense 5000x1024";
            expect(corpus_sparse.size() == kCorpusN) << "built sparse 5000 rows";
            // persist for the warm leg
            write_f32_file(cache_dense, corpus_emb);
            write_corpus_sparse(cache_sparse, corpus_ids, corpus_sparse);
            LUISA_INFO("LEG1 cache: wrote cache files {}", cache_dense.string());
        }

        // ------- freshness gate: ALWAYS re-encode a 20-passage subset and assert it
        // matches the cache that is now in memory / on disk.
        // Indices chosen to cover short rows, the 512-truncation long rows, the
        // deliberate duplicates (4997/4998/4999) and their originals (140/2827/2837).
        const std::vector<size_t> freshness = {0u, 1u, 2u, 3u, 4u, 140u, 500u, 1000u,
                                               1500u, 2000u, 2448u, 2827u, 2837u, 2838u, 3547u,
                                               3797u, 4697u, 4900u, 4997u, 4998u};
        std::vector<std::string> ftexts;
        ftexts.reserve(freshness.size());
        for (size_t i : freshness) {
            ftexts.push_back(corpus_texts[i]);
        }
        std::vector<float> femb;
        femb.reserve(freshness.size() * kDenseDim);
        std::vector<std::vector<std::pair<uint32_t, float>>> fsparse;
        size_t jobs = 0u, rows = 0u;
        luisa::compute::Clock clock;
        build_embeddings(*session, ftexts, is_gpu, jobs, rows, femb, fsparse);
        double worst_cos = 2.0;
        double worst_abs = 0.0;
        size_t worst_idx = 0u;
        uint32_t stale = 0u;
        for (size_t k = 0; k < freshness.size(); k++) {
            const size_t pidx = freshness[k];
            const float *cache_row = corpus_emb.data() + pidx * kDenseDim;
            const float *fresh_row = femb.data() + k * kDenseDim;
            const auto c = cosine(cache_row, fresh_row, kDenseDim);
            double mx = 0.0;
            for (size_t d = 0; d < kDenseDim; d++) {
                mx = std::max(mx, std::fabs(static_cast<double>(cache_row[d]) - fresh_row[d]));
            }
            if (c < worst_cos) {
                worst_cos = c;
                worst_abs = mx;
                worst_idx = pidx;
            }
            // cosine >= 0.999 distinguishes fp noise (>=0.99999 same-backend,
            // ~>=0.9999999 cross-backend per plan S6) from real staleness (wrong
            // text -> cosine far from 1). Tie-window irrelevant here: a genuine
            // corpus change moves the embedding by ~O(0.1), orders of magnitude above
            // the 1e-3 tolerance below.
            if (!(c >= 0.999)) {
                stale++;
                LUISA_WARNING("LEG1 freshness: passage p{:04d} re-encode does NOT match cache "
                              "(cosine {:.6f}, max abs diff {:.3e}) -- cache is STALE?",
                              pidx, c, mx);
            }
        }
        LUISA_INFO("LEG1 freshness gate: re-encoded {} passages in {:.1f} ms vs cache; worst "
                   "cosine {:.8f} (max abs {:.3e}, passage p{:04d}); {} of {} outside tolerance; "
                   "gate: every subset row matches the cache (cos>=0.999; catches stale text, "
                   "tolerates same-/cross-backend fp noise)",
                   freshness.size(), clock.toc(), worst_cos, worst_abs, worst_idx, stale,
                   freshness.size());
        expect(stale == 0u) << "LEG1: 20-passage subset re-encode matches the corpus cache";
    }

    // ============================================================ LEG 2: encode queries
    std::vector<float> query_emb;
    query_emb.reserve(kQueryN * kDenseDim);
    std::vector<std::vector<std::pair<uint32_t, float>>> query_sparse;
    {
        if (is_gpu) {
            LUISA_INFO("LEG2: query encode chunked under the {} 65535-cap (chunk {} / job)",
                       backend, kGpuChunkPassages);
        }
        size_t jobs = 0u, rows = 0u;
        luisa::compute::Clock clock;
        build_embeddings(*session, query_texts, is_gpu, jobs, rows, query_emb, query_sparse);
        LUISA_INFO("LEG2: encoded {} queries in {} atomic batch jobs in {:.1f} ms", rows, jobs,
                   clock.toc());
        expect(query_emb.size() == kQueryN * kDenseDim) << "query dense 200x1024";
        expect(query_sparse.size() == kQueryN) << "query sparse 200 rows";
    }

    // ---- cross-backend oracle: FNV-1a-64 checksum of the 200 query embeddings +
    // per-backend temp dump (reviewer diffs two backends by checksum / dump bytes).
    {
        const auto checksum = fnv1a64(query_emb.data(), query_emb.size() * 4u);
        const auto dump = std::filesystem::temp_directory_path() /
                          (std::string{"embed_retrieval_"} + backend + "_query_dense.bin");
        write_f32_file(dump, query_emb);
        LUISA_INFO("CROSS-BACKEND ORACLE: FNV-1a-64 over 200 query dense embeddings = "
                   "0x{:016x}; dumped {} bytes to {}",
                   checksum, query_emb.size() * 4u, dump.string());
        LUISA_INFO("CROSS-BACKEND ORACLE: reviewers compare backends by running this binary on "
                   "two backends and diffing the printed checksum / the per-backend dump "
                   "(plan S6: GPU-vs-GPU within fp op-order noise, cosine>=0.99999; fallback dump "
                   "is the deterministic oracle). Checksum must be STABLE per backend across runs.");
        // a stable-across-runs sanity gate: the checksum is non-trivial
        expect(checksum != 0xcbf29ce484222325ull) << "checksum was computed (hash not the init value)";
    }

    // ---- dense norms (engine dense rows are documented L2-normalized; used to turn
    // dots into cosines for robustness).
    std::vector<double> cnorm(kCorpusN), qnorm(kQueryN);
    for (size_t j = 0; j < kCorpusN; j++) {
        double s = 0.0;
        const float *r = corpus_emb.data() + j * kDenseDim;
        for (size_t d = 0; d < kDenseDim; d++) {
            s += static_cast<double>(r[d]) * r[d];
        }
        cnorm[j] = std::sqrt(s);
    }
    for (size_t i = 0; i < kQueryN; i++) {
        double s = 0.0;
        const float *r = query_emb.data() + i * kDenseDim;
        for (size_t d = 0; d < kDenseDim; d++) {
            s += static_cast<double>(r[d]) * r[d];
        }
        qnorm[i] = std::sqrt(s);
    }
    {
        // assert dense rows are unit-norm within 1e-3 (engine contract; sanity only)
        uint32_t bad = 0u;
        for (size_t j = 0; j < kCorpusN; j++) {
            if (!(std::fabs(cnorm[j] - 1.0) < 1e-3)) {
                bad++;
            }
        }
        for (size_t i = 0; i < kQueryN; i++) {
            if (!(std::fabs(qnorm[i] - 1.0) < 1e-3)) {
                bad++;
            }
        }
        LUISA_INFO("LEG2: corpus+query dense L2-norm == 1 within 1e-3: {} violations (expect 0)",
                   bad);
        expect(bad == 0u) << "dense rows are L2-normalized";
    }

    // ============================================================ score matrices (in-test,
    // plain host loops, no engine code -- reference search per plan S4.3 step 2).
    // engine_dense_scores[i][j] = cosine(engine q_i, corpus_j)
    std::vector<float> engine_dense(kQueryN * kCorpusN);
    {
        luisa::compute::Clock clock;
        for (size_t i = 0; i < kQueryN; i++) {
            const float *q = query_emb.data() + i * kDenseDim;
            const double qn = qnorm[i];
            for (size_t j = 0; j < kCorpusN; j++) {
                const float *c = corpus_emb.data() + j * kDenseDim;
                double dot = 0.0;
                for (size_t d = 0; d < kDenseDim; d++) {
                    dot += static_cast<double>(q[d]) * c[d];
                }
                double sc = (qn > 0.0 && cnorm[j] > 0.0) ? (dot / (qn * cnorm[j])) : -1.0;
                engine_dense[i * kCorpusN + j] = static_cast<float>(sc);
            }
        }
        LUISA_INFO("IN-TEST dense brute-force cosine over 200x5000 in {:.1f} ms", clock.toc());
    }

    // engine_sparse_scores[i][j] = dot over shared token ids (ascending token-id
    // accumulation, README "sparse" def). Inverted index token_id -> postings.
    std::vector<float> engine_sparse(kQueryN * kCorpusN);
    {
        std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, float>>> inv;
        for (uint32_t j = 0; j < kCorpusN; j++) {
            for (const auto &pr : corpus_sparse[j]) {
                inv[pr.first].emplace_back(j, pr.second);
            }
        }
        size_t corpus_nnz = 0u;
        for (const auto &kv : inv) {
            corpus_nnz += kv.second.size();
        }
        luisa::compute::Clock clock;
        std::vector<double> acc(kCorpusN);
        for (size_t i = 0; i < kQueryN; i++) {
            std::fill(acc.begin(), acc.end(), 0.0);
            std::vector<std::pair<uint32_t, float>> qsp = query_sparse[i];
            std::sort(qsp.begin(), qsp.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
            for (const auto &qp : qsp) {
                const auto found = inv.find(qp.first);
                if (found == inv.end()) {
                    continue;
                }
                for (const auto &pp : found->second) {
                    acc[pp.first] += static_cast<double>(qp.second) * pp.second;
                }
            }
            for (size_t j = 0; j < kCorpusN; j++) {
                engine_sparse[i * kCorpusN + j] = static_cast<float>(acc[j]);
            }
        }
        LUISA_INFO("IN-TEST sparse dot over shared token ids: corpus nnz={}, 200x5000 in {:.1f} ms",
                   corpus_nnz, clock.toc());
    }

    // engine_hybrid = engine_dense + engine_sparse (equal weights)
    std::vector<float> engine_hybrid(kQueryN * kCorpusN);
    for (size_t i = 0; i < kQueryN * kCorpusN; i++) {
        engine_hybrid[i] = engine_dense[i] + engine_sparse[i];
    }
    // reference_hybrid = golden_dense + golden_sparse (frozen matrices, equal weights)
    std::vector<float> ref_hybrid(kQueryN * kCorpusN);
    for (size_t i = 0; i < kQueryN * kCorpusN; i++) {
        ref_hybrid[i] = golden_dense[i] + golden_sparse[i];
    }

    // ============================================================ ranking-parity helper
    // For a query, build the engine top-10 id list (from a score matrix row) and
    // compare to a golden id list with the S4.3 tie rule. Returns pass/fail + tie info.
    auto engine_top_ids = [&](const std::vector<float> &scores, size_t i) {
        const auto idx = topk_indices(scores.data() + i * kCorpusN, kCorpusN, top_k);
        std::vector<std::string> ids;
        ids.reserve(idx.size());
        for (auto k : idx) {
            ids.push_back(corpus_ids[k]);
        }
        return ids;
    };

    // ---------------- LEG 3: dense ranking parity + metrics + recall ----------------
    size_t dense_par = 0u, dense_par_fail = 0u;
    double dense_recall_sum = 0.0;
    size_t dense_recall_fail = 0u;
    size_t dense_tie_pairs_total = 0u;
    size_t dense_queries_with_ties = 0u;
    // metrics
    double ndcg_engine = 0.0, ndcg_base = 0.0, mrr_engine = 0.0, mrr_base = 0.0;
    size_t metric_n = 0u;
    auto ndcg_mrr = [](const std::vector<std::string> &rank10,
                       const std::unordered_set<std::string> &gold, double &ndcg, double &mrr) {
        const double idcg_all = [&]() {
            double id = 0.0;
            size_t n = std::min<size_t>(gold.size(), 10u);
            for (size_t k = 0; k < n; k++) {
                id += 1.0 / std::log2(static_cast<double>(k) + 2.0);
            }
            return id;
        }();
        double dcg = 0.0, rr = 0.0;
        bool found = false;
        for (size_t k = 0; k < rank10.size() && k < 10u; k++) {
            if (gold.count(rank10[k])) {
                dcg += 1.0 / std::log2(static_cast<double>(k) + 2.0);
                if (!found) {
                    rr = 1.0 / static_cast<double>(k + 1u);
                    found = true;
                }
            }
        }
        ndcg = (idcg_all > 0.0) ? (dcg / idcg_all) : 0.0;
        mrr = rr;
    };
    for (size_t i = 0; i < kQueryN; i++) {
        const auto &qr = qrels.queries[i];
        const auto engine_ids = engine_top_ids(engine_dense, i);
        const auto g = build_group_of(qr.dense_tie_groups);
        const bool pass =
            (canonicalize(engine_ids, g) == canonicalize(qr.dense_top10, g));
        // recall (set-based => tie/order agnostic): |engine ∩ golden| / |golden|
        std::unordered_set<std::string> eset(engine_ids.begin(), engine_ids.end());
        size_t inter = 0u;
        for (const auto &id : qr.dense_top10) {
            if (eset.count(id)) {
                inter++;
            }
        }
        const double rec = qr.dense_top10.empty() ? 1.0 : (static_cast<double>(inter) / qr.dense_top10.size());
        dense_recall_sum += rec;
        if (rec < 0.99999) {
            dense_recall_fail++;
        }
        size_t tp = count_tie_pairs(qr.dense_tie_groups, eset);
        if (tp > 0u) {
            dense_tie_pairs_total += tp;
            dense_queries_with_ties++;
            {
                std::string gtxt;
                for (const auto &gr : qr.dense_tie_groups) {
                    gtxt += "[" + str_join(gr) + "] ";
                }
                LUISA_INFO("LEG3 dense: query {} has {} dense tie group(s): {}", qr.query_id,
                           qr.dense_tie_groups.size(), gtxt);
            }
        }
        if (pass) {
            dense_par++;
        } else {
            dense_par_fail++;
            LUISA_INFO("LEG3 dense PARITY MISS q={} engine=[{}] golden=[{}]", qr.query_id,
                       str_join(engine_ids), str_join(qr.dense_top10));
        }
        // MRR / NDCG (vs golden qrels baseline from golden_dense_scores)
        if (!qr.gold_ids.empty()) {
            std::unordered_set<std::string> gold(qr.gold_ids.begin(), qr.gold_ids.end());
            double ne, me, nb, mb;
            ndcg_mrr(engine_ids, gold, ne, me);
            ndcg_mrr(qr.dense_top10, gold, nb, mb);// golden baseline = frozen qrels/dense ranking
            ndcg_engine += ne;
            mrr_engine += me;
            ndcg_base += nb;
            mrr_base += mb;
            metric_n++;
        }
    }
    const double dense_par_rate = static_cast<double>(dense_par) / kQueryN;
    const double dense_recall = dense_recall_sum / static_cast<double>(kQueryN);
    LUISA_INFO("LEG 3 (dense ranking parity vs qrels, S4.3 step 3): exact-top10-parity {} ({:.1f}%), "
               "fail {} (gate >= 99%); Recall@10 modulo ties = {:.6f} (fail queries {}, gate ~1.0); "
               "tie pairs total {} across {} queries (frozen qrels report 6 groups / 6 pairs)",
               dense_par, dense_par_rate * 100.0, dense_par_fail, dense_recall, dense_recall_fail,
               dense_tie_pairs_total, dense_queries_with_ties);
    LUISA_INFO("LEG 3 metrics (MRR@10 / NDCG@10 over {} queries with gold, baseline from the frozen "
               "golden-dense ranking): engine MRR={:.6f} NDCG={:.6f}; baseline MRR={:.6f} NDCG={:.6f}; "
               "NDCG@10 drop = {:.6f} (alert gate > 0.005)",
               metric_n, metric_n ? mrr_engine / metric_n : 0.0, metric_n ? ndcg_engine / metric_n : 0.0,
               metric_n ? mrr_base / metric_n : 0.0, metric_n ? ndcg_base / metric_n : 0.0,
               metric_n ? (ndcg_base - ndcg_engine) / metric_n : 0.0);
    const double ndcg_drop = metric_n ? (ndcg_base - ndcg_engine) / metric_n : 0.0;

    // ---------------- LEG 4: sparse ranking parity ----------------
    size_t sparse_par = 0u, sparse_par_fail = 0u, sparse_tie_pairs_total = 0u;
    for (size_t i = 0; i < kQueryN; i++) {
        const auto &qr = qrels.queries[i];
        const auto engine_ids = engine_top_ids(engine_sparse, i);
        const auto g = build_group_of(qr.sparse_tie_groups);
        const bool pass = (canonicalize(engine_ids, g) == canonicalize(qr.sparse_top10, g));
        std::unordered_set<std::string> eset(engine_ids.begin(), engine_ids.end());
        size_t tp = count_tie_pairs(qr.sparse_tie_groups, eset);
        sparse_tie_pairs_total += tp;
        if (tp > 0u) {
            LUISA_INFO("LEG4 sparse: query {} sparse tie groups = {}", qr.query_id,
                       qr.sparse_tie_groups.size());
        }
        if (pass) {
            sparse_par++;
        } else {
            sparse_par_fail++;
            LUISA_INFO("LEG4 sparse PARITY MISS q={} engine=[{}] golden=[{}]", qr.query_id,
                       str_join(engine_ids), str_join(qr.sparse_top10));
        }
    }
    const double sparse_par_rate = static_cast<double>(sparse_par) / kQueryN;
    LUISA_INFO("LEG 4 (sparse ranking parity vs qrels, dot over shared token ids): parity {} "
               "({:.1f}%), fail {} (gate >= 99%); tie pairs total {} (frozen qrels report 11 groups / "
               "79 pairs)",
               sparse_par, sparse_par_rate * 100.0, sparse_par_fail, sparse_tie_pairs_total);

    // ---------------- LEG 5: hybrid ranking parity vs in-test reference ----------------
    size_t hyb_par = 0u, hyb_par_fail = 0u, hyb_tie_pairs_total = 0u;
    for (size_t i = 0; i < kQueryN; i++) {
        // reference hybrid top-10 from the frozen matrices
        const auto ref_idx = topk_indices(ref_hybrid.data() + i * kCorpusN, kCorpusN, top_k);
        std::vector<std::string> ref_ids;
        ref_ids.reserve(ref_idx.size());
        for (auto k : ref_idx) {
            ref_ids.push_back(corpus_ids[k]);
        }
        const auto hgroups = hybrid_tie_groups(ref_idx, corpus_ids,
                                               ref_hybrid.data() + i * kCorpusN, tie_eps);
        const auto g = build_group_of(hgroups);
        const auto engine_ids = engine_top_ids(engine_hybrid, i);
        const bool pass = (canonicalize(engine_ids, g) == canonicalize(ref_ids, g));
        std::unordered_set<std::string> eset(engine_ids.begin(), engine_ids.end());
        hyb_tie_pairs_total += count_tie_pairs(hgroups, eset);
        if (pass) {
            hyb_par++;
        } else {
            hyb_par_fail++;
            LUISA_INFO("LEG5 hybrid PARITY MISS q={} engine=[{}] ref=[{}]",
                       qrels.queries[i].query_id, str_join(engine_ids), str_join(ref_ids));
        }
    }
    const double hyb_par_rate = static_cast<double>(hyb_par) / kQueryN;
    LUISA_INFO("LEG 5 (hybrid = dense + sparse, equal weights; reference recomputed IN-TEST from the "
               "two FROZEN score matrices -- no qrels hybrid field): parity {} ({:.1f}%), fail {} "
               "(gate >= 99%); tie pairs total {}",
               hyb_par, hyb_par_rate * 100.0, hyb_par_fail, hyb_tie_pairs_total);

    // ---------------- LEG 6: negative controls ----------------
    // 6a: gibberish queries q198/q199 (query_src == "synthetic-gibberish").
    {
        LUISA_INFO("LEG6 gibberish: {} candidates flagged in qrels meta (xx_gibberish_candidates), "
                   "{} frozen into q198/q199",
                   qrels.meta.gibberish_candidates, qrels.meta.gibberish_ref.size());
        LUISA_INFO("LEG6 gibberish: plan S4.3 LITERAL bound = max dense < whole-matrix median "
                   "(median_dense_all={:.6f}); the frozen REFERENCE does not satisfy it "
                   "(README literal_holds=false) so the gate follows the README operationalisation: "
                   "(a) engine gibberish row matches frozen golden max, (b) stays below "
                   "real_query_top1_p25={:.6f}.",
                   qrels.meta.median_dense_all, qrels.meta.real_query_top1_p25);
        uint32_t gib_fail = 0u;
        for (size_t i = 0; i < kQueryN; i++) {
            const auto &qr = qrels.queries[i];
            if (qr.query_src != "synthetic-gibberish") {
                continue;
            }
            // engine max dense / sparse over the corpus for this gibberish row
            float max_e_dense = -1e30f, max_e_sparse = -1e30f;
            for (size_t j = 0; j < kCorpusN; j++) {
                max_e_dense = std::max(max_e_dense, engine_dense[i * kCorpusN + j]);
                max_e_sparse = std::max(max_e_sparse, engine_sparse[i * kCorpusN + j]);
            }
            const auto ref = qrels.meta.gibberish_ref.find(qr.query_id);
            const double ref_md = ref != qrels.meta.gibberish_ref.end() ? ref->second.first : 0.0;
            const double ref_ms = ref != qrels.meta.gibberish_ref.end() ? ref->second.second : 0.0;
            // (b) below the real-query top-1 p25 (operational gate)
            const bool below_p25 = max_e_dense < qrels.meta.real_query_top1_p25;
            // (a) match frozen golden max_dense within 5% rel OR 5e-2 abs (cross-backend fp +
            // engine-vs-torch noise; the absolute signal is "low score", already covered by (b))
            const bool match_md = std::fabs(max_e_dense - ref_md) < std::max(5e-2, 0.05 * ref_md);
            const bool match_ms = std::fabs(max_e_sparse - ref_ms) < std::max(5e-2, 0.05 * ref_ms);
            const bool literal_below_median = max_e_dense < qrels.meta.median_dense_all;
            LUISA_INFO("LEG6 gibberish {}: engine max_dense={:.6f} (frozen ref {:.6f}, match={}), "
                       "max_sparse={:.6f} (frozen ref {:.6f}, match={}); below_real_query_top1_p25={}; "
                       "literal §4.3 max<median({:.4f})={}(informational, NOT a gate)",
                       qr.query_id, max_e_dense, ref_md, match_md, max_e_sparse, ref_ms, match_ms,
                       below_p25, qrels.meta.median_dense_all, literal_below_median);
            if (!(below_p25 && match_md && match_ms)) {
                gib_fail++;
            }
        }
        LUISA_INFO("LEG6 gibberish control: {} gibberish rows checked, {} gate failures "
                   "(operational a+b)",
                   qrels.meta.gibberish_ref.size(), gib_fail);
        expect(gib_fail == 0u) << "LEG6: gibberish negative control (README operationalisation) passes";
    }
    // 6b: duplicate-passage adjacency.
    {
        if (qrels.meta.duplicate_pairs.empty()) {
            LUISA_INFO("LEG6 duplicates: NOT PRESENT in qrels meta -- no duplicate control to run "
                       "(not fabricating one).");
        } else {
            uint32_t dup_fail = 0u;
            for (const auto &pr : qrels.meta.duplicate_pairs) {
                const uint32_t a = pid_to_idx[pr.first], b = pid_to_idx[pr.second];
                // The frozen reference keeps the two copies bit-identical (same text ->
                // same column). The engine's cache encodes them in different batch jobs, so
                // the two dense embeddings differ by fp noise (~1e-6), and a strict
                // exact-float rank adjacency would be flaky for a query where a genuine
                // near-duplicate lands inside that ~1e-6 gap. So the §4.3 control is judged
                // MODULO the tie window (tie_eps, exactly as every ranking comparison in this
                // test): the two copies must (i) score equivalently (|d score| within the
                // tie window -> engine treats them as the same document, never silently
                // dropped) and (ii) be adjacent in the full ranking after collapsing
                // within-eps ties -> no passage with a score separated from BOTH copies by
                // more than tie_eps sits between them.
                const double cos = cosine(corpus_emb.data() + a * kDenseDim,
                                          corpus_emb.data() + b * kDenseDim, kDenseDim);
                double max_col_diff = 0.0;
                uint32_t col_gt_tie = 0u, separators = 0u;
                for (size_t i = 0; i < kQueryN; i++) {
                    const float *row = engine_dense.data() + i * kCorpusN;
                    const double d = std::fabs(static_cast<double>(row[a]) - row[b]);
                    max_col_diff = std::max(max_col_diff, d);
                    if (!(d < tie_eps)) {
                        col_gt_tie++;
                    }
                    const double lo = std::min(row[a], row[b]);
                    const double hi = std::max(row[a], row[b]);
                    // empty separator window when the copies tie within eps (the normal case)
                    if (hi - lo > 2.0 * tie_eps) {
                        for (uint32_t c = 0; c < kCorpusN; c++) {
                            if (c == a || c == b) {
                                continue;
                            }
                            const double sc = row[c];
                            if (sc > lo + tie_eps && sc < hi - tie_eps) {
                                separators++;
                            }
                        }
                    }
                }
                LUISA_INFO("LEG6 duplicates {}~{}: copy dense cosine {:.8f}; across 200 queries "
                           "max |d score| between copies = {:.3e} (queries over tie_eps {:.1e} ={}), "
                           "non-tied separators = {} (gate: scored equivalently within the tie window "
                           "and adjacent modulo ties -> engine keeps both, never silently dedups)",
                           pr.first, pr.second, cos, max_col_diff, tie_eps, col_gt_tie, separators);
                if (col_gt_tie != 0u || separators != 0u || !(cos >= 0.9999)) {
                    dup_fail++;
                }
            }
            LUISA_INFO("LEG6 duplicate control: {} pairs checked, {} failures (gate: copies score "
                       "equivalently within the tie window and are adjacent modulo ties)",
                       qrels.meta.duplicate_pairs.size(), dup_fail);
            expect(dup_fail == 0u) << "LEG6: duplicate passages score-equivalent + adjacent (no silent dedup)";
        }
    }

    // ============================================================ final gates (accumulate-then-expect)
    LUISA_INFO("========================================================");
    LUISA_INFO("GATES: dense parity {:.1f}% (>=99), sparse parity {:.1f}% (>=99), hybrid parity "
               "{:.1f}% (>=99), Recall@10 modulo ties {:.6f} (~1.0), NDCG@10 drop {:.6f} (<=0.005), "
               "negative controls (gibberish + duplicates) gated above.",
               dense_par_rate * 100.0, sparse_par_rate * 100.0, hyb_par_rate * 100.0, dense_recall,
               ndcg_drop);
    expect(dense_par >= 198u) << "GATE dense ranking parity >= 99% (<=2 fails of 200)";
    expect(sparse_par >= 198u) << "GATE sparse ranking parity >= 99%";
    expect(hyb_par >= 198u) << "GATE hybrid ranking parity >= 99%";
    expect(dense_recall >= 0.99) << "GATE Recall@10 modulo ties mean >= 0.99";
    expect(dense_recall_fail <= 2u) << "GATE at most 2 queries with set-recall<1 (== parity fails)";
    expect(ndcg_drop <= 0.005) << "GATE NDCG@10 drop vs frozen baseline <= 0.005";

    LUISA_INFO("L3 retrieval suite total wall time {:.1f} ms on backend {}", total_clock.toc(),
               backend);
    return 0;
}
