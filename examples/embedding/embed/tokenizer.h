#pragma once
// bge-m3 embedding engine — tokenizer (embedding_model_plan.md §3.2, milestone M1).
//
// Host-only, pure-C++ re-implementation of the HF fast-tokenizer pipeline that
// bge-m3's tokenizer.json specifies (Unigram / XLM-RoBERTa style). No dependency
// beyond luisa core STL + the bundled yyjson (used in the .cpp for JSON loading).
//
// Pipeline (semantics verified byte-for-byte against HF `tokenizers` 0.22 golden
// token ids for the pinned corpus under src/tests/data/embedding/):
//
//  1. Normalizer = Sequence[Precompiled charsmap, Replace(" {2,}" -> " ")].
//     The charsmap is sentencepiece's NormalizerCharsMap blob (base64 in the
//     JSON): <u32-LE trie_bytes><darts-clone double-array units (4 bytes each)>
//     <NUL-terminated replacement pool>.  Unit layout (darts-clone DoubleArrayUnit):
//     label = bits 0..7 (bit 31 set on leaf units makes their label never match a
//     byte), has_leaf = bit 8, offset = (unit >> 10) << (bit 9 ? 8 : 0), leaf value
//     = bits 0..30 (pool byte offset).  Normalization walks the input as UTF-8
//     bytes with a longest-match prefix query (pos ^= offset ^ byte, XOR scheme);
//     unmatched code points pass through unchanged, malformed bytes emit one
//     U+FFFD per bad lead byte (sentencepiece behavior).  The Replace step then
//     collapses runs of 2+ ASCII spaces to a single space.
//  2. Pre-tokenizer = Metaspace(replacement U+2581, add_prefix_space): each ASCII
//     space becomes "\xe2\x96\x81", a leading one is prepended (empty input yields
//     no pre-tokens, matching HF), then the string splits at every U+2581 so each
//     pre-token begins with it (back-to-back separators produce bare "\xe2\x96\x81"
//     pre-tokens).
//  3. Model = Unigram Viterbi over the 250002 (piece, score) entries, run
//     independently per pre-token.  Pieces are treated as byte strings (prefix
//     search via sorted-piece array + binary search at each reachable byte
//     position).  The DP maximizes the total log-prob; strict '>' updates keep
//     the earliest-arriving candidate on ties (HF encode_optimized behavior).
//     A position whose character has no single-character piece gets an <unk>
//     node (score = min(vocab scores) - 10); consecutive <unk> nodes are fused
//     into one <unk> id (3) unless the fused run is itself a vocab piece.
//  4. Post-processor = TemplateProcessing: ids = [bos] + body + [eos] (0 + body
//     + 2).  Truncation cuts the finished sequence to max_len while keeping the
//     leading/trailing special tokens: [bos] + body[:max_len - 2] + [eos]
//     (verified against the two truncated golden sentences).
//
// Output is token ids only; the attention mask is all-ones for the returned span
// and is not produced here (padding is applied by the caller/session).

#include <cstdint>
#include <filesystem>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::embed {

    /**
     * \brief bge-m3 (XLM-RoBERTa Unigram) tokenizer.
     *
     * Instances are immutable after \ref load and safe to share across threads.
     */
    class Tokenizer {

    public:
        /**
         * \brief Loads and parses an HF fast-tokenizer JSON (tokenizer.json).
         *
         * \param tokenizer_json Path to the tokenizer.json file.
         * \return The ready tokenizer. Fatal LUISA_ERROR if the file cannot be
         * read or the JSON deviates from the expected Unigram/Metaspace/Template
         * spec (this implementation is fail-closed; see the header comment).
         */
        static luisa::unique_ptr<Tokenizer> load(const std::filesystem::path &tokenizer_json);

        ~Tokenizer();

        Tokenizer(const Tokenizer &) = delete;
        Tokenizer(Tokenizer &&) = delete;
        Tokenizer &operator=(const Tokenizer &) = delete;
        Tokenizer &operator=(Tokenizer &&) = delete;

        /**
         * \brief Encodes one raw UTF-8 text into token ids.
         *
         * \param utf8 The input text.
         * \param max_len Truncation length applied after the template wraps the
         * sequence with <s>/</s>; must satisfy 2 <= max_len <= 8192 (hard model
         * ceiling is max_position_embeddings = 8194). Default 512 per plan §3.2.
         * \return ids: [0 (=<s>), ...body..., 2 (=</s>)], truncated to max_len as
         * described in the header comment. Never empty for a valid tokenizer.
         */
        luisa::vector<uint32_t> encode(luisa::string_view utf8, uint32_t max_len = 512u) const;

    private:
        Tokenizer();
        struct Impl;
        luisa::unique_ptr<Impl> _impl;
    };

}// namespace luisa::embed
