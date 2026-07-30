#include "ngram_tokenizer.h"
#include <luisa/vstl/string_utility.h>
#include <luisa/core/fiber.h>
#include <cctype>

namespace tokenize {

luisa::string NgramTokenizer::normalize(luisa::string_view text) {
    luisa::string s{text};
    vstd::StringUtil::to_lower(s);
    bool ascii = true;
    for (char c : s) {
        if (static_cast<unsigned char>(c) > 127) {
            ascii = false;
            break;
        }
    }
    if (ascii) return s;
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

int NgramTokenizer::detect_n(luisa::string_view text) const {
    if (text.empty()) return _n;
    bool ascii = true;
    for (char c : text) {
        if (static_cast<unsigned char>(c) > 127) {
            ascii = false;
            break;
        }
    }
    if (ascii) return (_n < 3) ? 3 : _n;

    size_t cjk_count = 0;
    size_t threshold = text.size() * 3 / 10;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        char32_t cp = 0;
        if (c < 0x80) {
            cp = c;
            ++i;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
            cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3F);
            i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            cp = ((c & 0x0F) << 12) |
                 ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                 (static_cast<unsigned char>(text[i + 2]) & 0x3F);
            i += 3;
        } else if (i + 3 < text.size()) {
            cp = ((c & 0x07) << 18) |
                 ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12) |
                 ((static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6) |
                 (static_cast<unsigned char>(text[i + 3]) & 0x3F);
            i += 4;
        } else {
            ++i;
            continue;
        }
        if (is_cjk(cp)) {
            ++cjk_count;
            if (cjk_count > threshold) return 2;
        }
    }
    return (_n < 3) ? 3 : _n;
}

luisa::vector<luisa::string> NgramTokenizer::tokenize(luisa::string_view text, int n) const {
    luisa::string norm = normalize(text);
    size_t start = 0;
    while (start < norm.size() && std::isspace(static_cast<unsigned char>(norm[start]))) ++start;
    size_t end = norm.size();
    while (end > start && std::isspace(static_cast<unsigned char>(norm[end - 1]))) --end;
    luisa::string_view trimmed(norm.data() + start, end - start);
    if (trimmed.empty()) return {};

    int use_n = (n >= 0) ? n : detect_n(trimmed);
    luisa::vector<luisa::string> result;
    if (static_cast<int>(trimmed.size()) < use_n) {
        result.emplace_back(trimmed);
        return result;
    }
    for (size_t i = 0; i + use_n <= trimmed.size(); ++i) {
        result.emplace_back(trimmed.substr(i, use_n));
    }
    return result;
}

luisa::vector<luisa::vector<luisa::string>> NgramTokenizer::tokenize_batch(const luisa::vector<luisa::string> &texts, int n) const {
    if (texts.empty()) return {};
    luisa::vector<luisa::vector<luisa::string>> results(texts.size());
    luisa::fiber::parallel(static_cast<uint32_t>(texts.size()), [&](uint32_t i) noexcept {
        results[i] = tokenize(texts[i], n);
    });
    return results;
}

}// namespace tokenize
