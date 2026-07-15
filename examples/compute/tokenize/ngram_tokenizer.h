#pragma once

#include <luisa/core/stl.h>

namespace tokenize {

class NgramTokenizer {
public:
    explicit NgramTokenizer(int n = 2) noexcept : _n(n) {}

    static luisa::string normalize(luisa::string_view text);
    static bool is_cjk(char32_t cp) noexcept;

    int detect_n(luisa::string_view text) const;
    luisa::vector<luisa::string> tokenize(luisa::string_view text, int n = -1) const;

    // Batch tokenization: embarrassingly parallel over documents
    luisa::vector<luisa::vector<luisa::string>> tokenize_batch(const luisa::vector<luisa::string> &texts, int n = -1) const;

private:
    int _n;
};

}// namespace tokenize
