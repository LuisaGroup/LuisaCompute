// bge-m3 embedding engine — tokenizer implementation (embedding_model_plan.md
// §3.2, milestone M1). Pure host C++, zero new dependencies: the only external
// ingredient is the bundled yyjson (already linked through lc-example-embedding).
//
// The pipeline implemented here (charsmap double-array normalization, Metaspace
// pre-tokenization, Unigram Viterbi, <s>/</s> template + truncation) reproduces
// the HF `tokenizers` runtime exactly for the pinned golden corpus; see the
// header comment for the format spec and the tie/fuse semantics.

#include "embedding/embed/tokenizer.h"

#include <cstdint>
#include <cstring>
#include <thread>
#include <atomic>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/string.h>

#include <yyjson.h>

namespace luisa::embed {

namespace {

// U+2581 LOWER ONE EIGHT BLOCK (validated on load).
constexpr luisa::string_view kMetaspace = "\xe2\x96\x81";
// U+FFFD REPLACEMENT CHARACTER (sentencepiece malformed-byte output).
constexpr luisa::string_view kReplacementChar = "\xef\xbf\xbd";
// HF unigram K_UNK_PENALTY (unk node score = min vocab score - penalty).
constexpr double kUnkPenalty = 10.0;
// Hard model ceiling: max_position_embeddings = 8194 (plan §3.2).
constexpr uint32_t kMaxSeqLenCeiling = 8192u;
// Sentinel for "no piece" / "unk node" in the DP.
constexpr size_t kNoPiece = ~size_t{0u};

// Decode one code point's byte length from its lead byte, mirroring
// sentencepiece's strict IsValidDecodeUTF8 (rejects overlong forms,
// surrogates, out-of-range and truncated sequences; 0 = invalid).
size_t utf8_char_len(const uint8_t *d, size_t avail) noexcept {
    if (avail == 0u) return 0u;
    const uint32_t b0 = d[0];
    if (b0 < 0x80u) return 1u;
    size_t l = 0u;
    if (b0 >= 0xC2u && b0 <= 0xDFu)
        l = 2u;
    else if (b0 >= 0xE0u && b0 <= 0xEFu)
        l = 3u;
    else if (b0 >= 0xF0u && b0 <= 0xF4u)
        l = 4u;
    else
        return 0u;
    if (avail < l) return 0u;
    for (size_t k = 1u; k < l; k++)
        if (d[k] < 0x80u || d[k] > 0xBFu) return 0u;
    if (l == 3u) {
        const uint32_t cp = ((b0 & 0x0Fu) << 12u) | ((d[1] & 0x3Fu) << 6u) | (d[2] & 0x3Fu);
        if (cp < 0x800u || (cp >= 0xD800u && cp <= 0xDFFFu)) return 0u;
    } else if (l == 4u) {
        const uint32_t cp = ((b0 & 0x07u) << 18u) | ((d[1] & 0x3Fu) << 12u) | ((d[2] & 0x3Fu) << 6u) | (d[3] & 0x3Fu);
        if (cp > 0x10FFFFu) return 0u;
    }
    return l;
}

// darts-clone double-array unit accessors (sentencepiece
// NormalizerCharsMap blob; value type int32). Leaf units carry bit 31,
// which makes darts_label() never equal a byte — that is how the walk
// rejects them; child lookup is pos = base XOR byte.
uint32_t darts_offset(uint32_t u) noexcept {
    return (u >> 10u) << ((u & (1u << 9u)) >> 6u);
}
bool darts_has_leaf(uint32_t u) noexcept {
    return ((u >> 8u) & 1u) != 0u;
}
uint32_t darts_label(uint32_t u) noexcept {
    return u & ((1u << 31u) | 0xFFu);
}
uint32_t darts_value(uint32_t u) noexcept {
    return u & ((1u << 31u) - 1u);
}

const yyjson_val *obj_get_or_die(const yyjson_val *obj, const char *key) {
    auto v = yyjson_obj_get(obj, key);
    if (v == nullptr) LUISA_ERROR("tokenizer.json: missing field \"{}\"", key);
    return v;
}

// Copies the JSON string into owned storage (yyjson values alias the
// document's buffer — nothing borrowed may outlive the doc; yyjson skill).
luisa::string get_str_copy(const yyjson_val *v, luisa::string_view what) {
    if (!yyjson_is_str(v)) LUISA_ERROR("tokenizer.json: {} is not a string", what);
    return {yyjson_get_str(v), yyjson_get_len(v)};
}

// Standard base64 → raw bytes ('=' padding only at the tail, no
// whitespace); fatal on invalid input.
luisa::vector<uint8_t> base64_decode(luisa::string_view b64) {
    static auto value_of = [](char c) noexcept -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    luisa::vector<uint8_t> out;
    out.reserve(b64.size() / 4u * 3u);
    size_t end = b64.size();
    while (end > 0u && b64[end - 1u] == '=') end--;
    uint32_t bits = 0u;
    int nbits = 0;
    for (size_t i = 0u; i < end; i++) {
        const int v = value_of(b64[i]);
        if (v < 0) LUISA_ERROR("tokenizer.json: invalid base64 character");
        bits = (bits << 6u) | static_cast<uint32_t>(v);
        nbits += 6;
        if (nbits >= 8) {
            nbits -= 8;
            out.push_back(static_cast<uint8_t>((bits >> static_cast<uint32_t>(nbits)) & 0xFFu));
        }
    }
    if (nbits >= 6) LUISA_ERROR("tokenizer.json: invalid base64 length");
    return out;
}

}// namespace

struct Tokenizer::Impl {

    // charsmap (Precompiled normalizer)
    luisa::vector<uint32_t> cm_units;
    luisa::vector<char> cm_pool;// NUL-terminated replacement strings

    // template ids, taken from post_processor and its special_tokens map
    uint32_t bos_id = 0u;
    uint32_t eos_id = 2u;
    uint32_t unk_id = 3u;
    double unk_score = 0.0;

    // Unigram vocab sorted lexicographically by piece bytes: an exact-match
    // binary search per byte length enumerates every piece covering a
    // position (valid UTF-8 pieces can only start on character boundaries).
    struct piece_entry {
        luisa::string bytes;
        uint32_t id;
        double score;
    };
    luisa::vector<piece_entry> pieces;
    size_t max_piece_len = 0u;

    // Longest-match charsmap query over [p, p+avail). Returns the consumed
    // byte length (0 when no rule matches); `pool_offset` receives the
    // offset of the NUL-terminated replacement for the longest match.
    size_t charsmap_match(const uint8_t *p, size_t avail, size_t &pool_offset) const noexcept {
        size_t best_len = 0u;
        size_t best_off = 0u;
        uint64_t pos = darts_offset(cm_units[0]);
        for (size_t i = 0u; i < avail; i++) {
            const uint32_t c = p[i];
            pos ^= c;
            if (pos >= cm_units.size()) break;
            const uint32_t u = cm_units[static_cast<size_t>(pos)];
            if (darts_label(u) != c) break;
            pos ^= darts_offset(u);
            if (darts_has_leaf(u) && pos < cm_units.size()) {
                const size_t len = i + 1u;
                if (len > best_len) {
                    best_len = len;
                    best_off = darts_value(cm_units[static_cast<size_t>(pos)]);
                }
            }
        }
        pool_offset = best_off;
        return best_len;
    }

    // Index of the piece whose bytes are exactly [p, p+len), or kNoPiece.
    size_t find_piece(const uint8_t *p, size_t len) const noexcept {
        size_t lo = 0u, hi = pieces.size();
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2u;
            const auto &pb = pieces[mid].bytes;
            const size_t cmp_len = pb.size() < len ? pb.size() : len;
            const int c = std::memcmp(pb.data(), p, cmp_len);
            if (c < 0)
                lo = mid + 1u;
            else if (c > 0)
                hi = mid;
            else if (pb.size() < len)
                lo = mid + 1u;
            else
                hi = mid;
        }
        if (lo < pieces.size() && pieces[lo].bytes.size() == len &&
            std::memcmp(pieces[lo].bytes.data(), p, len) == 0)
            return lo;
        return kNoPiece;
    }

    // Normalizes one raw UTF-8 text into the pre-tokenizer input:
    // charsmap longest-match folding, then Replace(" {2,}" -> " ").
    luisa::string normalize(luisa::string_view utf8) const {
        const auto *in = reinterpret_cast<const uint8_t *>(utf8.data());
        const size_t n = utf8.size();
        luisa::string folded;
        folded.reserve(n + n / 2u);
        size_t i = 0u;
        while (i < n) {
            size_t off = 0u;
            const size_t mlen = charsmap_match(in + i, n - i, off);
            if (mlen > 0u) {
                if (off >= cm_pool.size()) LUISA_ERROR("tokenizer.json: broken charsmap value");
                const char *s = cm_pool.data() + off;
                const size_t rem = cm_pool.size() - off;
                size_t slen = 0u;
                while (slen < rem && s[slen] != '\0') slen++;
                if (slen == rem) LUISA_ERROR("tokenizer.json: unterminated charsmap pool entry");
                folded.append(s, slen);
                i += mlen;
            } else {
                const size_t clen = utf8_char_len(in + i, n - i);
                if (clen == 0u) {
                    folded.append(kReplacementChar);
                    i += 1u;// consume one byte of the malformed sequence
                } else {
                    folded.append(reinterpret_cast<const char *>(in + i), clen);
                    i += clen;
                }
            }
        }
        // Replace(" {2,}", " "): collapse runs of 2+ ASCII spaces to one.
        luisa::string out;
        out.reserve(folded.size());
        {
            size_t k = 0u;
            while (k < folded.size()) {
                if (folded[k] == ' ') {
                    size_t r = k;
                    while (r < folded.size() && folded[r] == ' ') r++;
                    out.push_back(' ');
                    k = r;
                } else {
                    out.push_back(folded[k]);
                    k++;
                }
            }
        }
        return out;
    }

    // Metaspace pre-tokenization + Unigram encoding of one normalized text.
    void pre_tokenize_and_encode(luisa::string_view norm, luisa::vector<uint32_t> &body) const {
        // HF Metaspace (prepend_scheme=always, split=true): replace every
        // ASCII space with U+2581, prepend U+2581 unless the text already
        // starts with it, then split at every U+2581 so each pre-token
        // begins with the replacement character (back-to-back separators
        // yield bare U+2581 pre-tokens). Empty input yields no pre-tokens.
        if (norm.empty()) return;
        luisa::string pre;
        pre.reserve(norm.size() + norm.size() / 3u + 3u);
        for (size_t i = 0u; i < norm.size(); i++) {
            if (norm[i] == ' ')
                pre.append(kMetaspace);
            else
                pre.push_back(norm[i]);
        }
        const bool needs_prefix = !(pre.size() >= kMetaspace.size() &&
                                    std::memcmp(pre.data(), kMetaspace.data(), kMetaspace.size()) == 0);
        if (needs_prefix) pre.insert(static_cast<size_t>(0), kMetaspace);
        const auto *pb = reinterpret_cast<const uint8_t *>(pre.data());
        const size_t n = pre.size();
        // Segment starts = every U+2581 occurrence (its byte E2 can never be
        // a UTF-8 continuation byte, so byte scanning is exact on the valid
        // normalized text).
        unigram_each_segment(pb, n, body);
    }

    void unigram_each_segment(const uint8_t *pb, size_t n, luisa::vector<uint32_t> &body) const {
        luisa::vector<size_t> starts;
        for (size_t i = 0u; i + 3u <= n; i++) {
            if (pb[i] == 0xE2u && pb[i + 1u] == 0x96u && pb[i + 2u] == 0x81u) {
                starts.push_back(i);
                i += 2u;
            }
        }
        for (size_t k = 0u; k < starts.size(); k++) {
            const size_t a = starts[k];
            const size_t b = (k + 1u < starts.size()) ? starts[k + 1u] : n;
            unigram_encode(pb + a, b - a, body);
        }
    }

    // Unigram Viterbi over one pre-token. Faithful emulation of HF
    // `encode_optimized`: byte positions advanced per character; at each
    // reachable start every covering piece (and an <unk> node when no
    // single-character piece exists) competes for its end position with
    // strict '>' updates (earliest candidate wins ties); the best path is
    // backtracked and maximal runs of consecutive <unk> nodes are fused
    // into a single string which maps to one id (the piece id if it is
    // somehow in the vocab, else <unk>).
    void unigram_encode(const uint8_t *seg, size_t n, luisa::vector<uint32_t> &body) const {
        struct dp_node {
            double score;
            int64_t start;// -1: unreachable
            size_t piece; // index into pieces, kNoPiece for an unk node
        };
        luisa::vector<dp_node> dp(n + 1u);
        for (auto &node : dp) {
            node.score = 0.0;
            node.start = -1;
            node.piece = kNoPiece;
        }
        size_t s = 0u;
        while (s < n) {
            const size_t mblen = utf8_char_len(seg + s, n - s);
            if (mblen == 0u) {
                // Normalization guarantees valid UTF-8 (malformed bytes
                // became U+FFFD); defensive only.
                s += 1u;
                continue;
            }
            const double till = dp[s].score;
            bool has_single = false;
            const size_t lmax = (max_piece_len < n - s) ? max_piece_len : (n - s);
            for (size_t l = 1u; l <= lmax; l++) {
                const size_t idx = find_piece(seg + s, l);
                if (idx != kNoPiece) {
                    const size_t e = s + l;
                    const double cand = till + pieces[idx].score;
                    if (dp[e].start < 0 || cand > dp[e].score) {
                        dp[e].score = cand;
                        dp[e].start = static_cast<int64_t>(s);
                        dp[e].piece = idx;
                    }
                    if (l == mblen) has_single = true;
                }
            }
            if (!has_single) {
                const size_t e = s + mblen;
                const double cand = till + unk_score;
                if (dp[e].start < 0 || cand > dp[e].score) {
                    dp[e].score = cand;
                    dp[e].start = static_cast<int64_t>(s);
                    dp[e].piece = kNoPiece;
                }
            }
            s += mblen;
        }
        // Backtrack into (begin, end, piece) entries, sequence order last→first.
        struct seg_token {
            size_t begin;
            size_t end;
            size_t piece;
        };
        luisa::vector<seg_token> toks;
        {
            size_t ends = n;
            while (ends > 0u) {
                const auto &node = dp[ends];
                if (node.start < 0) LUISA_ERROR("tokenizer: unreachable DP end at {}", ends);
                const size_t starts_at = static_cast<size_t>(node.start);
                toks.push_back(seg_token{starts_at, ends, node.piece});
                ends = starts_at;
            }
        }
        // Emit forward (toks walked in reverse), fusing unk runs.
        // Text order is toks[size-1] .. toks[0]; a maximal forward run of
        // unk nodes spans [begin of its first token, end of its last one).
        for (size_t k = toks.size(); k > 0u;) {
            k--;
            const auto &tok = toks[k];
            if (tok.piece != kNoPiece) {
                body.push_back(pieces[tok.piece].id);
            } else {
                size_t run_end = tok.end;
                while (k > 0u && toks[k - 1u].piece == kNoPiece) {
                    k--;
                    run_end = toks[k].end;
                }
                const size_t run_idx = find_piece(seg + tok.begin, run_end - tok.begin);
                body.push_back(run_idx != kNoPiece ? pieces[run_idx].id : unk_id);
            }
        }
    }
};

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;

luisa::unique_ptr<Tokenizer> Tokenizer::load(const std::filesystem::path &tokenizer_json) {

    // private ctor: constructed via the friend-context of load() only
    // (std::make_unique would instantiate outside the class's access).
    auto tkn = luisa::unique_ptr<Tokenizer>{new Tokenizer()};
    auto impl = luisa::make_unique<Impl>();

    yyjson_read_err rerr;
    const auto path_str = tokenizer_json.string();
    yyjson_doc *doc = yyjson_read_file(path_str.c_str(), 0u, nullptr, &rerr);
    if (doc == nullptr) LUISA_ERROR("cannot read tokenizer.json \"{}\": {} (pos {})",
                                    path_str, (rerr.msg != nullptr ? rerr.msg : "?"), rerr.pos);
    struct doc_guard {
        yyjson_doc *d;
        ~doc_guard() { yyjson_doc_free(d); }
    } guard{doc};

    const auto *root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) LUISA_ERROR("tokenizer.json: root is not an object");

    // ── model: Unigram vocab + unk_id ────────────────────────────────────
    const auto *model = obj_get_or_die(root, "model");
    if (get_str_copy(yyjson_obj_get(model, "type"), "model.type") != "Unigram")
        LUISA_ERROR("tokenizer.json: expected Unigram model");
    const auto *unk_val = yyjson_obj_get(model, "unk_id");
    if (!yyjson_is_uint(unk_val)) LUISA_ERROR("tokenizer.json: missing model.unk_id");
    impl->unk_id = static_cast<uint32_t>(yyjson_get_uint(unk_val));
    if (yyjson_is_true(yyjson_obj_get(model, "byte_fallback")))
        LUISA_ERROR("tokenizer.json: byte_fallback is not supported");

    const auto *vocab = obj_get_or_die(model, "vocab");
    if (!yyjson_is_arr(vocab) || yyjson_arr_size(vocab) == 0u)
        LUISA_ERROR("tokenizer.json: model.vocab is empty");
    impl->pieces.reserve(yyjson_arr_size(vocab));
    {
        double min_score = 0.0;
        bool first_entry = true;
        yyjson_val *item;
        yyjson_arr_iter iter = yyjson_arr_iter_with(vocab);
        uint32_t id = 0u;
        while ((item = yyjson_arr_iter_next(&iter)) != nullptr) {
            if (!yyjson_is_arr(item) || yyjson_arr_size(item) != 2u)
                LUISA_ERROR("tokenizer.json: bad vocab entry at id {}", id);
            auto piece = get_str_copy(yyjson_arr_get(item, 0u), "vocab piece");
            const auto *sv = yyjson_arr_get(item, 1u);
            if (!yyjson_is_num(sv)) LUISA_ERROR("tokenizer.json: bad vocab score at id {}", id);
            const double score = yyjson_get_num(sv);
            if (first_entry || score < min_score) {
                min_score = score;
                first_entry = false;
            }
            impl->pieces.push_back(Tokenizer::Impl::piece_entry{std::move(piece), id, score});
            id++;
        }
        impl->unk_score = min_score - kUnkPenalty;
    }
    luisa::sort(impl->pieces.begin(), impl->pieces.end(),
                [](const auto &a, const auto &b) noexcept { return a.bytes < b.bytes; });
    for (const auto &pe : impl->pieces)
        impl->max_piece_len = impl->max_piece_len < pe.bytes.size() ? pe.bytes.size() : impl->max_piece_len;

    // ── normalizer: Sequence[Precompiled, Replace(" {2,}" -> " ")] ───────
    const auto *normalizer = obj_get_or_die(root, "normalizer");
    if (get_str_copy(yyjson_obj_get(normalizer, "type"), "normalizer.type") != "Sequence")
        LUISA_ERROR("tokenizer.json: expected normalizer Sequence");
    const auto *subs = obj_get_or_die(normalizer, "normalizers");
    if (!yyjson_is_arr(subs) || yyjson_arr_size(subs) != 2u)
        LUISA_ERROR("tokenizer.json: expected exactly 2 sub-normalizers (Precompiled + Replace)");
    {
        const auto *pre = yyjson_arr_get(subs, 0u);
        const auto *rep = yyjson_arr_get(subs, 1u);
        if (get_str_copy(yyjson_obj_get(pre, "type"), "normalizers[0].type") != "Precompiled")
            LUISA_ERROR("tokenizer.json: normalizers[0] must be Precompiled");
        if (get_str_copy(yyjson_obj_get(rep, "type"), "normalizers[1].type") != "Replace")
            LUISA_ERROR("tokenizer.json: normalizers[1] must be Replace");
        const auto *pattern = obj_get_or_die(rep, "pattern");
        if (get_str_copy(yyjson_obj_get(pattern, "Regex"), "normalizers[1].pattern.Regex") != " {2,}")
            LUISA_ERROR("tokenizer.json: unsupported Replace pattern");
        if (get_str_copy(obj_get_or_die(rep, "content"), "normalizers[1].content") != " ")
            LUISA_ERROR("tokenizer.json: unsupported Replace content");

        const auto b64 = get_str_copy(obj_get_or_die(pre, "precompiled_charsmap"),
                                      "precompiled_charsmap");
        const auto blob = base64_decode(b64);
        // DecodePrecompiledCharsMap (sentencepiece normalizer.cc):
        // <u32-LE trie_bytes><double-array units><NUL-terminated pool>
        if (blob.size() <= 4u) LUISA_ERROR("tokenizer.json: charsmap blob too small");
        const uint32_t trie_bytes = static_cast<uint32_t>(blob[0]) |
                                    (static_cast<uint32_t>(blob[1]) << 8u) |
                                    (static_cast<uint32_t>(blob[2]) << 16u) |
                                    (static_cast<uint32_t>(blob[3]) << 24u);
        if (trie_bytes >= blob.size() - 4u)
            LUISA_ERROR("tokenizer.json: charsmap trie size exceeds blob");
        if (trie_bytes < 1024u || (trie_bytes & 0x3FFu) != 0u)
            LUISA_ERROR("tokenizer.json: charsmap trie size is not a multiple of 1024");
        impl->cm_units.resize(trie_bytes / 4u);
        for (size_t u = 0u; u < impl->cm_units.size(); u++) {
            const uint8_t *p = blob.data() + 4u + u * 4u;
            impl->cm_units[u] = static_cast<uint32_t>(p[0]) |
                                (static_cast<uint32_t>(p[1]) << 8u) |
                                (static_cast<uint32_t>(p[2]) << 16u) |
                                (static_cast<uint32_t>(p[3]) << 24u);
        }
        impl->cm_pool.assign(blob.begin() + static_cast<ptrdiff_t>(4u + trie_bytes), blob.end());
        if (impl->cm_pool.empty() || impl->cm_pool.back() != '\0')
            LUISA_ERROR("tokenizer.json: charsmap pool is not NUL-terminated");
    }

    // ── pre_tokenizer: Metaspace(U+2581, add_prefix_space, split) ────────
    {
        const auto *pt = obj_get_or_die(root, "pre_tokenizer");
        if (get_str_copy(yyjson_obj_get(pt, "type"), "pre_tokenizer.type") != "Metaspace")
            LUISA_ERROR("tokenizer.json: expected Metaspace pre_tokenizer");
        const auto replacement = get_str_copy(obj_get_or_die(pt, "replacement"),
                                              "pre_tokenizer.replacement");
        if (replacement != kMetaspace)
            LUISA_ERROR("tokenizer.json: Metaspace replacement must be U+2581");
        const auto *aps = yyjson_obj_get(pt, "add_prefix_space");
        if (aps != nullptr && !yyjson_is_true(aps))
            LUISA_ERROR("tokenizer.json: add_prefix_space must be true");
        const auto *ps = yyjson_obj_get(pt, "prepend_scheme");
        if (ps != nullptr && yyjson_is_str(ps) && get_str_copy(ps, "prepend_scheme") != "always")
            LUISA_ERROR("tokenizer.json: unsupported prepend_scheme");
        const auto *sp = yyjson_obj_get(pt, "split");
        if (sp != nullptr && !yyjson_is_true(sp))
            LUISA_ERROR("tokenizer.json: Metaspace split must be true");
    }

    // ── post_processor: TemplateProcessing <s> A </s> ────────────────────
    {
        const auto *pp = obj_get_or_die(root, "post_processor");
        if (get_str_copy(yyjson_obj_get(pp, "type"), "post_processor.type") != "TemplateProcessing")
            LUISA_ERROR("tokenizer.json: expected TemplateProcessing post_processor");
        const auto *single = obj_get_or_die(pp, "single");
        if (!yyjson_is_arr(single) || yyjson_arr_size(single) != 3u)
            LUISA_ERROR("tokenizer.json: post_processor.single must be [<s>, A, </s>]");
        if (yyjson_obj_get(yyjson_arr_get(single, 1u), "Sequence") == nullptr)
            LUISA_ERROR("tokenizer.json: post_processor.single[1] must be the sequence");
        const auto *specials = obj_get_or_die(pp, "special_tokens");
        auto special_id = [&](const yyjson_val *item) -> uint32_t {
            const auto *st = yyjson_obj_get(item, "SpecialToken");
            if (st == nullptr) LUISA_ERROR("tokenizer.json: post_processor.single entry is not a SpecialToken");
            const auto id = get_str_copy(obj_get_or_die(st, "id"), "special token id");
            const auto *entry = yyjson_obj_get(specials, id.c_str());
            if (entry == nullptr) LUISA_ERROR("tokenizer.json: special token \"{}\" has no entry", id);
            const auto *ids = obj_get_or_die(entry, "ids");
            if (!yyjson_is_arr(ids) || yyjson_arr_size(ids) != 1u)
                LUISA_ERROR("tokenizer.json: special token \"{}\" must map to exactly one id", id);
            return static_cast<uint32_t>(yyjson_get_uint(yyjson_arr_get(ids, 0u)));
        };
        impl->bos_id = special_id(yyjson_arr_get(single, 0u));
        impl->eos_id = special_id(yyjson_arr_get(single, 2u));
    }

    if (impl->bos_id >= impl->pieces.size() || impl->eos_id >= impl->pieces.size() ||
        impl->unk_id >= impl->pieces.size())
        LUISA_ERROR("tokenizer.json: template/unk ids out of vocab range");

    tkn->_impl = std::move(impl);
    return tkn;
}

luisa::vector<uint32_t> Tokenizer::encode(luisa::string_view utf8, uint32_t max_len) const {
    if (max_len < 2u || max_len > kMaxSeqLenCeiling)
        LUISA_ERROR("tokenizer: max_len {} outside [2, {}]", max_len, kMaxSeqLenCeiling);
    const auto &impl = *_impl;
    luisa::vector<uint32_t> ids;
    ids.reserve(max_len);
    ids.push_back(impl.bos_id);
    {
        const auto norm = impl.normalize(utf8);
        impl.pre_tokenize_and_encode(norm, ids);
    }
    ids.push_back(impl.eos_id);
    // Truncation applies to the finished sequence but keeps the template's
    // special tokens (HF TemplateProcessing + right-side truncation; verified
    // against the two truncated goldens): [bos] + body[:max_len-2] + [eos].
    if (ids.size() > max_len) {
        luisa::vector<uint32_t> truncated;
        truncated.reserve(max_len);
        truncated.push_back(ids[0]);
        truncated.insert(truncated.end(), ids.begin() + 1,
                         ids.begin() + static_cast<ptrdiff_t>(max_len - 1u));
        truncated.push_back(ids.back());
        ids.swap(truncated);
    }
    return ids;
}

luisa::vector<luisa::vector<uint32_t>> Tokenizer::encode_parallel(
    luisa::span<const luisa::string_view> texts,
    uint32_t max_len,
    uint32_t num_threads) const {

    if (max_len < 2u || max_len > kMaxSeqLenCeiling)
        LUISA_ERROR("tokenizer: max_len {} outside [2, {}]", max_len, kMaxSeqLenCeiling);

    const size_t n = texts.size();
    luisa::vector<luisa::vector<uint32_t>> results;
    results.resize(n);// default-constructed (empty) per slot; never resized again

    if (n == 0u) [[unlikely]] {
        return results;
    }

    // Effective worker count: caller request (or auto) clamped to the pool
    // cap and to the number of items. hw_concurrency of 0 (unsupported) is
    // treated as 1 so the serial path is taken.
    uint32_t wanted = num_threads;
    if (wanted == 0u || wanted > kTokenizerMaxPoolThreads) {
        auto hc = std::thread::hardware_concurrency();
        wanted = (hc == 0u) ? 1u : hc;
        if (wanted > kTokenizerMaxPoolThreads) { wanted = kTokenizerMaxPoolThreads; }
    }
    uint32_t workers = wanted;
    if (workers > static_cast<uint32_t>(n)) { workers = static_cast<uint32_t>(n); }
    if (workers < 2u) [[unlikely]] {
        // Serial fallback (also the single-item case): identical to a plain loop.
        for (size_t i = 0u; i < n; i++) {
            results[i] = encode(texts[i], max_len);
        }
        return results;
    }

    // Dynamic, cursor-based distribution so that ragged batches (mixed token
    // counts) load-balance. `results` is a fixed-size vector of distinct
    // luisa::vector objects: each thread writes a disjoint slot and the
    // container itself is never mutated after resize, so there is no data
    // race. The Tokenizer is reentrant (read-only after load).
    std::atomic<size_t> cursor{0u};
    luisa::vector<std::thread> pool;
    pool.reserve(workers);
    const Tokenizer *self = this;
    for (uint32_t t = 0u; t < workers; t++) {
        pool.emplace_back([self, &texts, &results, &cursor, max_len]() {
            size_t i = 0u;
            while ((i = cursor.fetch_add(1u, std::memory_order_relaxed)) < texts.size()) {
                results[i] = self->encode(texts[i], max_len);
            }
        });
    }
    for (auto &th : pool) { th.join(); }
    return results;
}

}// namespace luisa::embed
