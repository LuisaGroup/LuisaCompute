#pragma once

#include <luisa/core/stl.h>

namespace tokenize {

// UTF-8 aware word splitter used to build the n-gram token library.
//
// Tokenization rules:
//   - text is normalized with normalize() first;
//   - ASCII alphanumeric runs become one token each;
//   - whitespace and punctuation separate tokens;
//   - CJK codepoints (see is_cjk) become one token each;
//   - any other (non-CJK) multibyte codepoint becomes one token each.
//
// All members are static; the class cannot be instantiated.
class NgramTokenizer {
public:
    NgramTokenizer() = delete;
    ~NgramTokenizer() = delete;

    static luisa::string normalize(luisa::string_view text);
    static bool is_cjk(char32_t cp) noexcept;

    // Decode one UTF-8 codepoint at text[i], advancing i past its bytes.
    // Malformed bytes are consumed one at a time and decoded as U+FFFD.
    static char32_t decode_utf8(luisa::string_view text, size_t &i) noexcept;

    // Split text into word tokens following the rules above.
    [[nodiscard]] static luisa::vector<luisa::string> split(luisa::string_view text);
};

}// namespace tokenize
