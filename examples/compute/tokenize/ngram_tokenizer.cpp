#include "ngram_tokenizer.h"

#include <cctype>

namespace tokenize {

luisa::string NgramTokenizer::normalize(luisa::string_view text) {
    luisa::string s{text};
    for (auto &c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

bool NgramTokenizer::is_cjk(char32_t cp) noexcept {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0xAC00 && cp <= 0xD7AF) ||
           (cp >= 0x3040 && cp <= 0x309F) ||
           (cp >= 0x30A0 && cp <= 0x30FF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2EBEF);
}

char32_t NgramTokenizer::decode_utf8(luisa::string_view text, size_t &i) noexcept {
    const auto n = text.size();
    const auto start = i;
    const auto byte_at = [&](size_t k) noexcept {
        return static_cast<unsigned char>(text[k]);
    };
    unsigned char c = byte_at(i);
    if (c < 0x80) {
        ++i;
        return static_cast<char32_t>(c);
    }
    if ((c & 0xE0) == 0xC0 && i + 1 < n) {
        char32_t cp = ((c & 0x1F) << 6) | (byte_at(i + 1) & 0x3F);
        i += 2;
        return cp;
    }
    if ((c & 0xF0) == 0xE0 && i + 2 < n) {
        char32_t cp = ((c & 0x0F) << 12) |
                      ((byte_at(i + 1) & 0x3F) << 6) |
                      (byte_at(i + 2) & 0x3F);
        i += 3;
        return cp;
    }
    if ((c & 0xF8) == 0xF0 && i + 3 < n) {
        char32_t cp = ((c & 0x07) << 18) |
                      ((byte_at(i + 1) & 0x3F) << 12) |
                      ((byte_at(i + 2) & 0x3F) << 6) |
                      (byte_at(i + 3) & 0x3F);
        i += 4;
        return cp;
    }
    // Malformed sequence: consume one byte and return U+FFFD.
    ++i;
    (void)start;
    return 0xFFFD;
}

luisa::vector<luisa::string> NgramTokenizer::split(luisa::string_view text) const {
    luisa::string norm = normalize(text);
    luisa::vector<luisa::string> tokens;
    const auto n = norm.size();
    auto is_ascii_delim = [](char c) noexcept {
        auto uc = static_cast<unsigned char>(c);
        return std::isspace(uc) || std::ispunct(uc);
    };
    size_t i = 0;
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(norm[i]);
        if (c < 0x80) {
            if (is_ascii_delim(norm[i])) {
                ++i;
                continue;
            }
            size_t j = i;
            while (j < n) {
                unsigned char cj = static_cast<unsigned char>(norm[j]);
                if (cj >= 0x80 || is_ascii_delim(norm[j])) break;
                ++j;
            }
            tokens.emplace_back(norm.substr(i, j - i));
            i = j;
        } else {
            size_t begin = i;
            decode_utf8(norm, i);
            // One token per multibyte codepoint (CJK or not).
            tokens.emplace_back(norm.substr(begin, i - begin));
        }
    }
    return tokens;
}

}// namespace tokenize
