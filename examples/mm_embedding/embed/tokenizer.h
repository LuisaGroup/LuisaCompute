#pragma once
// mm_embedding engine -- Gemma BPE Tokenizer.
//
// Pure C++ implementation of SentencePiece / Gemma BPE tokenization:
// - Parses HF tokenizer.json using yyjson
// - BPE merges and vocabulary lookup
// - Byte fallback (<0x00>..<0xFF>)
// - Special tokens (<pad>=0, <eos>=1, <bos>=2, <unk>=3)

#include <cstdint>
#include <filesystem>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::mm_embed {

class Tokenizer {
public:
    static luisa::unique_ptr<Tokenizer> load(const std::filesystem::path &tokenizer_json);

    virtual ~Tokenizer() = default;

    Tokenizer(const Tokenizer &) = delete;
    Tokenizer(Tokenizer &&) = delete;
    Tokenizer &operator=(const Tokenizer &) = delete;
    Tokenizer &operator=(Tokenizer &&) = delete;

    [[nodiscard]] virtual luisa::vector<uint32_t> encode(luisa::string_view utf8, uint32_t max_len = 512u) const = 0;

    [[nodiscard]] virtual luisa::vector<luisa::vector<uint32_t>> encode_batch(
        luisa::span<const luisa::string_view> texts,
        uint32_t max_len = 512u) const = 0;

    [[nodiscard]] virtual uint32_t bos_id() const noexcept = 0;
    [[nodiscard]] virtual uint32_t eos_id() const noexcept = 0;
    [[nodiscard]] virtual uint32_t pad_id() const noexcept = 0;
    [[nodiscard]] virtual uint32_t unk_id() const noexcept = 0;
    [[nodiscard]] virtual uint32_t vocab_size() const noexcept = 0;

protected:
    Tokenizer() = default;
};

}// namespace luisa::mm_embed
