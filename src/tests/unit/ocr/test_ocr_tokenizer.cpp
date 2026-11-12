// Test for examples/ocr/ocr/tokenizer.h/.cpp and detokenizer.h/.cpp.
// Host-only: no device, no backend, no model, no exceptions.
//
// This test covers:
// - byte-level BPE: byte encoder/decoder, ranks, added/special tokens,
//   pre-tokenization; Chinese/English/mixed/emoji round trips; exact IDs;
// - Unigram with byte fallback + metaspace;
// - invalid UTF-8 policy (encode rejects, no lossy replacement);
// - added-token longest match + special-token suppression;
// - unsupported normalizer/pre-tokenizer/post-processor/model fail closed;
// - incremental detokenizer: UTF-8 across token boundaries, EOS, stop strings
//   without splitting multibyte output or leaking stop text.
// Fixtures are frozen synthetic JSON (never regenerated here). These are NOT
// official tokenizer differentials (no official outputs are claimed).
#include "ut/ut.hpp"

#include <cstdio>
#include <filesystem>

#include <luisa/core/stl/filesystem.h>
#include <yyjson.h>

#include "ocr/detokenizer.h"
#include "ocr/tokenizer.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] luisa::filesystem::path data_dir() {
    return luisa::filesystem::path{LUISA_OCR_TEST_DATA_DIR};
}

void test_bpe_roundtrips() {
    auto r = Tokenizer::load(data_dir() / "tokenizer_bpe.json");
    expect(r.ok());
    if (!r.ok()) { return; }
    auto &tok = *r.value;
    expect(tok->model_kind() == TokenizerModelKind::byte_level_bpe);
    const char *cases[] = {"hello", "Hello World", "hi 你好", "你好世界",
                           "mixed 中文 text", "emoji 😀 test", "  leading spaces"};
    for (auto text : cases) {
        auto enc = tok->encode(text, false);
        expect(enc.ok()) << "encode: " << text;
        if (!enc.ok()) { continue; }
        auto dec = tok->decode(*enc.value, true);
        expect(dec.ok()) << "decode: " << text;
        if (dec.ok()) {
            expect(*dec.value == text) << "roundtrip: " << text << " got " << *dec.value;
        }
    }
}

void test_bpe_special_tokens() {
    auto r = Tokenizer::load(data_dir() / "tokenizer_bpe.json");
    expect(r.ok());
    if (!r.ok()) { return; }
    auto &tok = *r.value;
    // Added image token is matched longest-first and not split.
    auto enc = tok->encode("a<|image|>b", false);
    expect(enc.ok());
    if (enc.ok()) {
        // The image token id must appear exactly once.
        auto image_id = tok->token_to_id("<|image|>");
        expect(image_id.has_value());
        if (image_id.has_value()) {
            uint32_t count = 0;
            for (auto id : *enc.value) { if (id == *image_id) { count++; } }
            expect(count == 1u);
        }
        // Special tokens suppressed on decode.
        auto dec = tok->decode(*enc.value, true);
        expect(dec.ok() && *dec.value == "ab");
        // With skip_special_tokens=false the image token text is kept.
        auto dec2 = tok->decode(*enc.value, false);
        expect(dec2.ok() && dec2.value->find("<|image|>") != luisa::string::npos);
    }
    // Post-processor prefix/suffix (BOS/EOS) applied with add_special_tokens.
    auto enc2 = tok->encode("hello", true);
    expect(enc2.ok());
    if (enc2.ok() && tok->bos_id().has_value() && tok->eos_id().has_value()) {
        expect(enc2.value->front() == *tok->bos_id());
        expect(enc2.value->back() == *tok->eos_id());
    }
}

void test_bpe_invalid_utf8() {
    auto r = Tokenizer::load(data_dir() / "tokenizer_bpe.json");
    expect(r.ok());
    if (!r.ok()) { return; }
    auto &tok = *r.value;
    // Invalid UTF-8 (lone continuation byte) is rejected, not replaced.
    const char bad[] = {'a', static_cast<char>(0x80), 'b', 0};
    auto enc = tok->encode(bad, false);
    expect(!enc.ok());
    expect(enc.status.code == StatusCode::invalid_argument);
    // Truncated multibyte sequence rejected.
    const char trunc[] = {'a', static_cast<char>(0xe4), static_cast<char>(0xbd), 0};// incomplete 你
    expect(!tok->encode(trunc, false).ok());
}

void test_unigram() {
    auto r = Tokenizer::load(data_dir() / "tokenizer_unigram.json");
    expect(r.ok());
    if (!r.ok()) { return; }
    auto &tok = *r.value;
    expect(tok->model_kind() == TokenizerModelKind::unigram);
    // Official SentencePiece/HF Metaspace: decode strips exactly ONE leading
    // structural U+2581, so decode(encode(t)) == t. Verified against official
    // tokenizers 0.22.2 (see golden_manifest official tokenizer artifact).
    const char *cases[] = {"hello", "hello world", "你好"};
    for (auto text : cases) {
        auto enc = tok->encode(text, false);
        expect(enc.ok()) << "unigram encode: " << text;
        if (!enc.ok()) { continue; }
        auto dec = tok->decode(*enc.value, true);
        expect(dec.ok()) << "unigram decode: " << text;
        if (dec.ok()) {
            expect(*dec.value == text) << "unigram roundtrip: " << text << " got " << *dec.value;
        }
    }
    // Byte fallback: a character not in the vocab encodes via <0xHH> tokens.
    auto enc = tok->encode("xyz", false);// x/y/z not in the small vocab
    expect(enc.ok());
    if (enc.ok()) {
        auto dec = tok->decode(*enc.value, true);
        expect(dec.ok() && *dec.value == "xyz");
    }
}

void test_unsupported_semantics() {
    // Unsupported model type fails closed.
    const char *bad_model =
        "{\"model\":{\"type\":\"WordPiece\",\"vocab\":{\"a\":0}},\"added_tokens\":[]}";
    expect(!Tokenizer::parse(bad_model).ok());
    // Unsupported normalizer for BPE.
    const char *bad_norm =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[],\"byte_encoder\":{},\"byte_decoder\":{}},"
        "\"normalizer\":{\"type\":\"NFKC\"},\"added_tokens\":[]}";
    expect(!Tokenizer::parse(bad_norm).ok());
    // BPE without byte encoder/decoder fails closed.
    const char *no_bytes =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]},\"added_tokens\":[]}";
    expect(!Tokenizer::parse(no_bytes).ok());
}

void test_detokenizer_incremental() {
    auto r = Tokenizer::load(data_dir() / "tokenizer_bpe.json");
    expect(r.ok());
    if (!r.ok()) { return; }
    auto &tok = *r.value;
    // Encode a multibyte string, then feed tokens one at a time; the joined
    // incremental output must equal the full decode (no split characters).
    const luisa::string text = "你好世界 hello";
    auto enc = tok->encode(text, false);
    expect(enc.ok());
    if (!enc.ok()) { return; }
    DetokenizerOptions options;
    IncrementalDetokenizer detok{*tok, options};
    luisa::string accumulated;
    for (auto id : *enc.value) {
        auto piece = detok.push(id);
        expect(piece.ok());
        if (piece.ok()) { accumulated += *piece.value; }
    }
    auto tail = detok.finish();
    expect(tail.ok());
    accumulated += *tail.value;
    expect(accumulated == text) << "incremental got " << accumulated;
}

void test_detokenizer_stop_string() {
    auto r = Tokenizer::load(data_dir() / "tokenizer_bpe.json");
    expect(r.ok());
    if (!r.ok()) { return; }
    auto &tok = *r.value;
    const luisa::string text = "hello STOP world";
    auto enc = tok->encode(text, false);
    expect(enc.ok());
    if (!enc.ok()) { return; }
    DetokenizerOptions options;
    options.stop_strings = {"STOP"};
    IncrementalDetokenizer detok{*tok, options};
    luisa::string accumulated;
    for (auto id : *enc.value) {
        auto piece = detok.push(id);
        expect(piece.ok());
        if (piece.ok()) { accumulated += *piece.value; }
        if (detok.stopped()) { break; }
    }
    auto tail = detok.finish();
    accumulated += *tail.value;
    expect(detok.stopped());
    // Stop text must not leak; output is the prefix before STOP.
    expect(accumulated.find("STOP") == luisa::string::npos) << "stop leaked: " << accumulated;
    expect(accumulated == "hello ") << "stop prefix got '" << accumulated << "'";
}

void test_detokenizer_eos() {
    auto r = Tokenizer::load(data_dir() / "tokenizer_bpe.json");
    expect(r.ok());
    if (!r.ok()) { return; }
    auto &tok = *r.value;
    if (!tok->eos_id().has_value()) { return; }
    auto enc = tok->encode("hi", false);
    expect(enc.ok());
    if (!enc.ok()) { return; }
    DetokenizerOptions options;
    IncrementalDetokenizer detok{*tok, options};
    luisa::string accumulated;
    for (auto id : *enc.value) {
        auto piece = detok.push(id);
        accumulated += *piece.value;
    }
    // Feed EOS; stream stops.
    auto eos_piece = detok.push(*tok->eos_id());
    expect(eos_piece.ok());
    expect(detok.stopped());
    // Further pushes produce nothing.
    auto after = detok.push(*tok->eos_id());
    expect(after.ok() && after.value->empty());
    expect(accumulated == "hi");
}

// Build a minimal Unigram tokenizer JSON with a full <0xHH> byte-fallback
// vocab plus a few word pieces. `extra_normalizer`/`extra_model` inject fields.
[[nodiscard]] luisa::string unigram_json(luisa::string_view extra_normalizer = "",
                                         luisa::string_view extra_model = "",
                                         luisa::string_view replacement = "\\u2581") {
    luisa::string vocab = "[\"<unk>\",0.0],[\"<s>\",0.0],[\"</s>\",0.0],"
                          "[\"\\u2581\",-1.0],[\"\\u2581hello\",-2.0],[\"\\u2581world\",-2.5],"
                          "[\"h\",-3.0],[\"e\",-3.0],[\"l\",-3.0],[\"o\",-3.0],"
                          "[\"\\u2581\\u4f60\\u597d\",-2.0],[\"\\u4f60\",-3.0],[\"\\u597d\",-3.0]";
    for (uint32_t b = 0; b < 256u; b++) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), ",[\"<0x%02X>\",-10.0]", b);
        vocab += buf;
    }
    luisa::string normalizer = "{\"type\":\"Metaspace\",\"replacement\":\"";
    normalizer += replacement;
    normalizer += "\",\"add_prefix_space\":true";
    normalizer += extra_normalizer;
    normalizer += "}";
    luisa::string model = "{\"type\":\"Unigram\",\"unk_id\":0,\"byte_fallback\":true";
    model += extra_model;
    model += ",\"vocab\":[" + vocab + "]}";
    return "{\"version\":\"1.0\",\"added_tokens\":["
           "{\"id\":0,\"content\":\"<unk>\",\"special\":true},"
           "{\"id\":1,\"content\":\"<s>\",\"special\":true},"
           "{\"id\":2,\"content\":\"</s>\",\"special\":true}],"
           "\"normalizer\":" + normalizer + ",\"model\":" + model + "}";
}

// Independent oracle: the exact <0xHH> byte-token ids for a UTF-8 string,
// preceded by the structural metaspace token (id 3 in unigram_json).
[[nodiscard]] luisa::vector<uint32_t> expected_byte_fallback_ids(luisa::string_view text) {
    luisa::vector<uint32_t> ids;
    ids.emplace_back(3u);// "▁" structural prefix
    for (unsigned char c : text) {
        // <0xHH> pieces start at id 13 in unigram_json (after 13 word pieces).
        ids.emplace_back(13u + static_cast<uint32_t>(c));
    }
    return ids;
}

void test_unigram_byte_fallback_order() {
    // é (C3 A9), a CJK char absent from the vocab (U+6587 = E6 96 87), and an
    // emoji (U+1F40D = F0 9F 90 8D) must encode to byte tokens in FORWARD byte
    // order and decode back exactly (regression: reversed multibyte bytes).
    auto r = Tokenizer::parse(unigram_json());
    expect(r.ok());
    if (!r.ok()) { return; }
    auto &tok = *r.value;
    const char *cases[] = {"\xc3\xa9", "\xe6\x96\x87", "\xf0\x9f\x90\x8d", "a\xf0\x9f\x90\x8d" "b"};
    for (auto text : cases) {
        auto enc = tok->encode(text, false);
        expect(enc.ok()) << "byte-fallback encode";
        if (!enc.ok()) { continue; }
        // Exact ids: structural metaspace + forward byte tokens.
        const auto expected = expected_byte_fallback_ids(text);
        expect(enc.value->size() == expected.size()) << "byte-fallback id count";
        bool ids_match = enc.value->size() == expected.size();
        for (size_t i = 0; ids_match && i < expected.size(); i++) {
            ids_match = (*enc.value)[i] == expected[i];
        }
        expect(ids_match) << "byte-fallback exact forward ids";
        auto dec = tok->decode(*enc.value, true);
        expect(dec.ok()) << "byte-fallback decode";
        if (dec.ok()) {
            expect(*dec.value == text) << "byte-fallback roundtrip";
        }
    }
}

void test_unigram_incremental_equivalence() {
    // Batch decode and incremental detokenization must agree, including the
    // leading metaspace space and interior spaces (regression: per-token strip).
    auto r = Tokenizer::parse(unigram_json());
    expect(r.ok());
    if (!r.ok()) { return; }
    auto &tok = *r.value;
    const char *cases[] = {"hello world", "hello  world", "\xf0\x9f\x90\x8d", "a b c"};
    for (auto text : cases) {
        auto enc = tok->encode(text, false);
        expect(enc.ok());
        if (!enc.ok()) { continue; }
        auto whole = tok->decode(*enc.value, true);
        expect(whole.ok());
        DetokenizerOptions opt;
        IncrementalDetokenizer detok{*tok, opt};
        luisa::string acc;
        for (auto id : *enc.value) {
            auto p = detok.push(id);
            expect(p.ok());
            if (p.ok()) { acc += *p.value; }
        }
        auto tail = detok.finish();
        expect(tail.ok());
        if (tail.ok()) { acc += *tail.value; }
        if (whole.ok()) {
            expect(acc == *whole.value) << "batch/incremental equivalence: " << text;
        }
    }
    // Standalone metaspace token "▁" then "▁hello" streamed: spaces preserved.
    {
        auto enc = tok->encode("hello world", false);
        expect(enc.ok());
        if (enc.ok()) {
            DetokenizerOptions opt;
            IncrementalDetokenizer detok{*tok, opt};
            luisa::string acc;
            for (auto id : *enc.value) {
                auto p = detok.push(id);
                if (p.ok()) { acc += *p.value; }
            }
            auto tail = detok.finish();
            if (tail.ok()) { acc += *tail.value; }
            expect(acc == "hello world") << "streamed spaces preserved, got '" << acc << "'";
        }
    }
}

void test_unigram_fields() {
    // add_prefix_space=false honored: no structural prefix metaspace.
    {
        auto r = Tokenizer::parse(unigram_json(",\"add_prefix_space\":false"));
        expect(r.ok());
        if (r.ok()) {
            auto e = (*r.value)->encode("hello", false);
            expect(e.ok());
            if (e.ok() && !e.value->empty()) {
                auto first = (*r.value)->id_to_token((*e.value)[0]);
                expect(first.has_value() && *first != "\xe2\x96\x81") << "no prefix metaspace when add_prefix_space=false";
            }
        }
    }
    // replacement != U+2581 rejected.
    {
        auto r = Tokenizer::parse(unigram_json("", "", "_"));
        expect(!r.ok());
    }
    // add_bos_token=false / add_eos_token=false honored.
    {
        auto r = Tokenizer::parse(unigram_json("", ",\"add_bos_token\":false,\"add_eos_token\":false"));
        expect(r.ok());
        if (r.ok()) {
            auto e = (*r.value)->encode("hello", true);
            expect(e.ok());
            if (e.ok() && !e.value->empty()) {
                expect(e.value->front() != 1u) << "add_bos_token=false honored";
                expect(e.value->back() != 2u) << "add_eos_token=false honored";
            }
        }
    }
    // add_bos_token=true / add_eos_token=true honored (positive).
    {
        auto r = Tokenizer::parse(unigram_json("", ",\"add_bos_token\":true,\"add_eos_token\":true"));
        expect(r.ok());
        if (r.ok()) {
            auto e = (*r.value)->encode("hello", true);
            expect(e.ok());
            if (e.ok() && !e.value->empty()) {
                expect(e.value->front() == 1u) << "add_bos_token=true honored";
                expect(e.value->back() == 2u) << "add_eos_token=true honored";
            }
        }
    }
}

void test_unigram_byte_token_literal() {
    // Literal "<0x41>" text must NOT be matched as the byte-fallback token; it
    // encodes via ordinary pieces/byte-fallback of its characters and decodes
    // back to the literal text (regression: Viterbi literal byte-token match).
    auto r = Tokenizer::parse(unigram_json());
    expect(r.ok());
    if (!r.ok()) { return; }
    auto &tok = *r.value;
    auto enc = tok->encode("<0x41>", false);
    expect(enc.ok());
    if (!enc.ok()) { return; }
    // The single byte token <0x41> is id 13+0x41=78; it must NOT appear as a
    // lone literal match (the text is 6 characters, not one byte).
    bool has_lone_byte_token = false;
    for (auto id : *enc.value) { if (id == 78u) { has_lone_byte_token = true; } }
    expect(!has_lone_byte_token) << "literal <0x41> not matched as byte token";
    auto dec = tok->decode(*enc.value, true);
    expect(dec.ok());
    if (dec.ok()) {
        expect(*dec.value == "<0x41>") << "literal <0x41> roundtrip, got " << *dec.value;
    }
}

// Official leading-metaspace examples (tokenizers 0.22.2 differential):
//   decode([▁hello])          = 'hello'
//   decode([▁hello,▁world])   = 'hello world'
//   decode([▁,▁hello])        = ' hello'   (only ONE leading strip)
void test_unigram_leading_metaspace_official() {
    auto r = Tokenizer::parse(unigram_json());
    expect(r.ok());
    if (!r.ok()) { return; }
    auto &tok = *r.value;
    const auto hello = tok->token_to_id("\u2581hello");
    const auto world = tok->token_to_id("\u2581world");
    const auto space = tok->token_to_id("\u2581");
    expect(hello.has_value() && world.has_value() && space.has_value());
    if (!hello.has_value() || !world.has_value() || !space.has_value()) { return; }
    {
        const uint32_t ids[] = {*hello};
        auto d = tok->decode(ids, true);
        expect(d.ok() && *d.value == "hello") << "[▁hello] -> 'hello'";
    }
    {
        const uint32_t ids[] = {*hello, *world};
        auto d = tok->decode(ids, true);
        expect(d.ok() && *d.value == "hello world") << "[▁hello,▁world] -> 'hello world'";
    }
    {
        const uint32_t ids[] = {*space, *hello};
        auto d = tok->decode(ids, true);
        expect(d.ok() && *d.value == " hello") << "[▁,▁hello] -> ' hello' (one strip)";
    }
    // encode('hello') is exactly [▁hello]; round trip is 'hello'.
    {
        auto e = tok->encode("hello", false);
        expect(e.ok() && e.value->size() == 1u && (*e.value)[0] == *hello);
        auto d = tok->decode(*e.value, true);
        expect(d.ok() && *d.value == "hello");
    }
    // encode(' hello') is [▁,▁hello]; round trip keeps the real space.
    {
        auto e = tok->encode(" hello", false);
        expect(e.ok() && e.value->size() == 2u && (*e.value)[0] == *space && (*e.value)[1] == *hello);
        auto d = tok->decode(*e.value, true);
        expect(d.ok() && *d.value == " hello");
    }
    // Streaming the same id sequences must equal batch decode exactly.
    const uint32_t seqs[][2] = {{*hello, 0xffffffffu}, {*hello, *world}, {*space, *hello}};
    const char *wants[] = {"hello", "hello world", " hello"};
    for (int s = 0; s < 3; s++) {
        DetokenizerOptions opt;
        IncrementalDetokenizer detok{*tok, opt};
        luisa::string acc;
        for (int k = 0; k < 2; k++) {
            if (seqs[s][k] == 0xffffffffu) { break; }
            auto p = detok.push(seqs[s][k]);
            expect(p.ok());
            if (p.ok()) { acc += *p.value; }
        }
        auto t = detok.finish();
        expect(t.ok());
        if (t.ok()) { acc += *t.value; }
        expect(acc == wants[s]) << "streamed seq " << s << " got '" << acc << "'";
    }
}

// Frozen OFFICIAL differential (tokenizers 0.22.2 + sentencepiece 0.2.2,
// generated once; tests only READ it). Verifies our validated subsets against
// official encode IDs and decode text, including the leading-metaspace rule.
void test_official_tokenizer_differential() {
    const auto path = data_dir() / "official_tokenizer_diff.json";
    std::FILE *f = std::fopen(luisa::to_string(path).c_str(), "rb");
    expect(f != nullptr) << "official differential fixture present";
    if (f == nullptr) { return; }
    luisa::string json;
    char buf[65536];
    size_t got = 0;
    while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) { json.append(buf, got); }
    std::fclose(f);
    yyjson_doc *doc = yyjson_read(json.data(), json.size(), 0);
    expect(doc != nullptr);
    if (doc == nullptr) { return; }
    auto *root = yyjson_doc_get_root(doc);
    auto *uni = yyjson_obj_get(root, "unigram");
    auto *bpe = yyjson_obj_get(root, "bpe");
    expect(uni != nullptr && bpe != nullptr);
    const auto run_cases = [&](yyjson_val *section, const Tokenizer &tok, const char *label) {
        auto *arr = yyjson_obj_get(section, "encode_decode");
        expect(arr != nullptr && yyjson_is_arr(arr));
        if (arr == nullptr || !yyjson_is_arr(arr)) { return; }
        yyjson_val *item;
        size_t i, m;
        yyjson_arr_foreach(arr, i, m, item) {
            const char *text = yyjson_get_str(yyjson_obj_get(item, "text"));
            const char *want = yyjson_get_str(yyjson_obj_get(item, "decode"));
            auto *ids_arr = yyjson_obj_get(item, "ids");
            luisa::vector<uint32_t> ids;
            yyjson_val *iv;
            size_t j, n;
            yyjson_arr_foreach(ids_arr, j, n, iv) { ids.emplace_back(static_cast<uint32_t>(yyjson_get_uint(iv))); }
            auto enc = tok.encode(text, false);
            expect(enc.ok()) << label << " encode " << text;
            if (enc.ok()) {
                expect(*enc.value == ids) << label << " official IDs for " << text;
                auto dec = tok.decode(*enc.value, true);
                expect(dec.ok() && *dec.value == want) << label << " official decode " << text;
            }
        }
    };
    {
        auto r = Tokenizer::load(data_dir() / "tokenizer_unigram.json");
        expect(r.ok());
        if (r.ok()) { run_cases(uni, **r.value, "unigram"); }
    }
    {
        auto r = Tokenizer::load(data_dir() / "tokenizer_bpe.json");
        expect(r.ok());
        if (r.ok()) { run_cases(bpe, **r.value, "bpe"); }
    }
    // Official decode-only leading-metaspace cases.
    {
        auto r = Tokenizer::load(data_dir() / "tokenizer_unigram.json");
        expect(r.ok());
        if (r.ok()) {
            auto *arr = yyjson_obj_get(uni, "decode_only");
            yyjson_val *item;
            size_t i, m;
            yyjson_arr_foreach(arr, i, m, item) {
                const char *want = yyjson_get_str(yyjson_obj_get(item, "decode"));
                auto *ids_arr = yyjson_obj_get(item, "ids");
                luisa::vector<uint32_t> ids;
                yyjson_val *iv;
                size_t j, n;
                yyjson_arr_foreach(ids_arr, j, n, iv) { ids.emplace_back(static_cast<uint32_t>(yyjson_get_uint(iv))); }
                auto dec = (*r.value)->decode(ids, true);
                expect(dec.ok() && *dec.value == want) << "official decode-only -> " << want;
            }
        }
    }
    // Official NFC singleton IDs: our subset must REJECT these inputs (the
    // official mapping is recorded here; we never pass raw bytes through).
    {
        auto r = Tokenizer::load(data_dir() / "tokenizer_bpe.json");
        expect(r.ok());
        if (r.ok()) {
            auto *arr = yyjson_obj_get(bpe, "nfc_singletons");
            yyjson_val *item;
            size_t i, m;
            yyjson_arr_foreach(arr, i, m, item) {
                const char *text = yyjson_get_str(yyjson_obj_get(item, "text"));
                auto *ids_arr = yyjson_obj_get(item, "ids");
                expect(yyjson_arr_size(ids_arr) > 0) << "official NFC ids recorded";
                auto e = (*r.value)->encode(text, false);
                expect(!e.ok()) << "NFC singleton rejected (official maps it)";
            }
        }
    }
    yyjson_doc_free(doc);
}

// NFC fail-closed must cover singleton mappings, not just combining marks.
// Official HF (tokenizers 0.22.2, NFC normalizer) maps these singletons before
// byte-level encoding; our validated subset REJECTS them (never passes raw
// bytes). Expected official IDs for the six listed chars are pinned in the
// frozen official differential fixture; here we assert the rejection.
void test_bpe_nfc_singletons_rejected() {
    auto r = Tokenizer::load(data_dir() / "tokenizer_bpe.json");
    expect(r.ok());
    if (!r.ok()) { return; }
    auto &tok = *r.value;
    // U+2126 OHM, U+212A KELVIN, U+212B ANGSTROM, U+037E GREEK QUESTION MARK,
    // U+0374 GREEK NUMERAL SIGN, U+0387 GREEK ANO TELEIA.
    const char *singletons[] = {
        "\xe2\x84\xa6", "\xe2\x84\xaa", "\xe2\x84\xab",
        "\xcd\xbe", "\xcd\xb4", "\xce\x87",
    };
    for (auto text : singletons) {
        auto e = tok->encode(text, false);
        expect(!e.ok()) << "NFC singleton must be rejected";
        if (!e.ok()) { expect(e.status.code == StatusCode::unsupported); }
    }
    // A CJK compatibility ideograph (U+F900) also maps under NFC -> rejected.
    expect(!tok->encode("\xef\xa4\x80", false).ok());
    // Precomposed NFC-invariant text still passes (Ω U+03A9, K, Å U+00C5).
    expect(tok->encode("\xce\xa9", false).ok());
    expect(tok->encode("K", false).ok());
    expect(tok->encode("\xc3\x85", false).ok());
}

void test_uint32_overflow_rejected() {
    // Oversized uint ids must be rejected, not silently narrowed.
    const char *big_added =
        "{\"model\":{\"type\":\"Unigram\",\"unk_id\":0,\"vocab\":[[\"a\",0.0]]},"
        "\"added_tokens\":[{\"id\":4294967296,\"content\":\"<x>\",\"special\":true}]}";
    expect(!Tokenizer::parse(big_added).ok());
    const char *big_unk =
        "{\"model\":{\"type\":\"Unigram\",\"unk_id\":4294967296,\"vocab\":[[\"a\",0.0]]},\"added_tokens\":[]}";
    expect(!Tokenizer::parse(big_unk).ok());
    const char *big_bpe_vocab =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":4294967296},\"merges\":[],"
        "\"byte_encoder\":{},\"byte_decoder\":{}},\"added_tokens\":[]}";
    expect(!Tokenizer::parse(big_bpe_vocab).ok());
}

void test_bpe_digit_split() {
    // Arabic-Indic / Devanagari digits are \p{N}, not \p{L}: they must round-trip
    // and not be classified as letters (regression: is_letter_cp digit overlap).
    auto r = Tokenizer::load(data_dir() / "tokenizer_bpe.json");
    expect(r.ok());
    if (!r.ok()) { return; }
    auto &tok = *r.value;
    const char *cases[] = {
        "\xd9\xa0\xd9\xa1\xd9\xa2",       // Arabic-Indic 012
        "\xe0\xa5\xa6\xe0\xa5\xa7",       // Devanagari 67
        "abc\xd9\xa0\xd9\xa1",            // mixed letters+digits
        "\xe4\xbd\xa0\xe5\xa5\xbd" "abc", // CJK + ASCII letters
        "123abc",                          // ASCII digits + letters
    };
    for (auto text : cases) {
        auto enc = tok->encode(text, false);
        expect(enc.ok()) << "digit encode";
        if (!enc.ok()) { continue; }
        auto dec = tok->decode(*enc.value, true);
        expect(dec.ok()) << "digit decode";
        if (dec.ok()) {
            expect(*dec.value == text) << "digit roundtrip: " << text;
        }
    }
}

}// namespace

static auto test_ocr_tokenizer_registration = [] {
    "bpe_roundtrips"_test = [] { test_bpe_roundtrips(); };
    "bpe_special_tokens"_test = [] { test_bpe_special_tokens(); };
    "bpe_invalid_utf8"_test = [] { test_bpe_invalid_utf8(); };
    "unigram"_test = [] { test_unigram(); };
    "unsupported_semantics"_test = [] { test_unsupported_semantics(); };
    "detokenizer_incremental"_test = [] { test_detokenizer_incremental(); };
    "detokenizer_stop_string"_test = [] { test_detokenizer_stop_string(); };
    "detokenizer_eos"_test = [] { test_detokenizer_eos(); };
    "unigram_leading_metaspace_official"_test = [] { test_unigram_leading_metaspace_official(); };
    "unigram_byte_fallback_order"_test = [] { test_unigram_byte_fallback_order(); };
    "unigram_incremental_equivalence"_test = [] { test_unigram_incremental_equivalence(); };
    "unigram_fields"_test = [] { test_unigram_fields(); };
    "unigram_byte_token_literal"_test = [] { test_unigram_byte_token_literal(); };
    "official_tokenizer_differential"_test = [] { test_official_tokenizer_differential(); };
    "bpe_nfc_singletons_rejected"_test = [] { test_bpe_nfc_singletons_rejected(); };
    "uint32_overflow_rejected"_test = [] { test_uint32_overflow_rejected(); };
    "bpe_digit_split"_test = [] { test_bpe_digit_split(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}