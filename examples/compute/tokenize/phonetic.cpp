#include "phonetic.h"
#include <cctype>
#include <luisa/vstl/string_utility.h>

namespace tokenize {

luisa::string soundex(luisa::string_view word) {
    luisa::string w{word};
    vstd::StringUtil::to_upper(w);
    if (w.empty()) return "";
    char first = w[0];
    auto map_char = [](char ch) -> char {
        switch (ch) {
            case 'B': case 'F': case 'P': case 'V': return '1';
            case 'C': case 'G': case 'J': case 'K': case 'Q': case 'S': case 'X': case 'Z': return '2';
            case 'D': case 'T': return '3';
            case 'L': return '4';
            case 'M': case 'N': return '5';
            case 'R': return '6';
            default: return '0';
        }
    };
    luisa::string digits;
    digits.push_back(first);
    char prev = map_char(first);
    for (size_t i = 1; i < w.size(); ++i) {
        char d = map_char(w[i]);
        if (d != '0') {
            if (d != prev) digits.push_back(d);
            prev = d;
        } else {
            prev = '0';
        }
    }
    luisa::string result;
    result.push_back(digits[0]);
    for (size_t i = 1; i < digits.size(); ++i) {
        if (digits[i] != '0') result.push_back(digits[i]);
    }
    while (result.size() < 4) result.push_back('0');
    if (result.size() > 4) result.resize(4);
    return result;
}

luisa::string metaphone(luisa::string_view word) {
    luisa::string w{word};
    vstd::StringUtil::to_upper(w);
    if (w.empty()) return "";
    luisa::string result;
    size_t i = 0, n = w.size();
    auto at = [&](size_t idx) -> char {
        return (idx < n) ? w[idx] : '\0';
    };
    while (i < n) {
        char ch = w[i];
        char next_ch = at(i + 1);
        char next2 = at(i + 2);
        char prev = (i > 0) ? w[i - 1] : '\0';

        if (std::string_view{"AEIOU"}.find(ch) != std::string_view::npos) {
            if (i == 0) result.push_back(ch);
            ++i;
            continue;
        }
        if (ch == 'B') {
            if (!(i == n - 1 && prev == 'M')) result.push_back('B');
            ++i;
        } else if (ch == 'C') {
            if (next_ch == 'H' && prev != 'S') {
                result.push_back('X');
                i += 2;
            } else if (next_ch == 'I' && next2 == 'A') {
                result.push_back('X');
                i += 3;
            } else if (std::string_view{"IEY"}.find(next_ch) != std::string_view::npos) {
                result.push_back('S');
                i += 2;
            } else {
                result.push_back('K');
                ++i;
            }
        } else if (ch == 'D') {
            if (next_ch == 'G' && std::string_view{"IEY"}.find(next2) != std::string_view::npos) {
                result.push_back('J');
                i += 3;
            } else {
                result.push_back('T');
                ++i;
            }
        } else if (ch == 'F') {
            result.push_back('F');
            ++i;
        } else if (ch == 'G') {
            if (next_ch == 'H') {
                if (i > 0 && std::string_view{"AEIOU"}.find(prev) == std::string_view::npos) result.push_back('K');
                i += 2;
            } else if (next_ch == 'N') {
                if (i != n - 2) result.push_back('N');
                i += 2;
            } else if (next_ch == 'E' && next2 == 'L') {
                result.push_back('K');
                i += 3;
            } else if (next_ch == 'I' && next2 == 'O') {
                result.push_back('J');
                i += 3;
            } else if (std::string_view{"IEY"}.find(next_ch) != std::string_view::npos) {
                result.push_back('J');
                i += 2;
            } else {
                result.push_back('K');
                ++i;
            }
        } else if (ch == 'H') {
            if (std::string_view{"AEIOU"}.find(prev) != std::string_view::npos || std::string_view{"AEIOU"}.find(next_ch) != std::string_view::npos)
                result.push_back('H');
            ++i;
        } else if (ch == 'J') {
            result.push_back('J');
            ++i;
        } else if (ch == 'K') {
            if (prev != 'C') result.push_back('K');
            ++i;
        } else if (ch == 'L') {
            result.push_back('L');
            ++i;
        } else if (ch == 'M') {
            result.push_back('M');
            ++i;
        } else if (ch == 'N') {
            result.push_back('N');
            ++i;
        } else if (ch == 'P') {
            if (next_ch == 'H') {
                result.push_back('F');
                i += 2;
            } else {
                result.push_back('P');
                ++i;
            }
        } else if (ch == 'Q') {
            result.push_back('K');
            ++i;
        } else if (ch == 'R') {
            result.push_back('R');
            ++i;
        } else if (ch == 'S') {
            if (next_ch == 'H' || (next_ch == 'I' && std::string_view{"OA"}.find(next2) != std::string_view::npos)) {
                result.push_back('X');
                i += (next_ch == 'H') ? 2 : 3;
            } else {
                result.push_back('S');
                ++i;
            }
        } else if (ch == 'T') {
            if (next_ch == 'I' && std::string_view{"OA"}.find(next2) != std::string_view::npos) {
                result.push_back('X');
                i += 3;
            } else if (next_ch == 'H') {
                result.push_back('0');
                i += 2;
            } else if (next_ch == 'C' && next2 == 'H') {
                ++i;
            } else {
                result.push_back('T');
                ++i;
            }
        } else if (ch == 'V') {
            result.push_back('F');
            ++i;
        } else if (ch == 'W') {
            if (std::string_view{"AEIOU"}.find(next_ch) != std::string_view::npos) result.push_back('W');
            ++i;
        } else if (ch == 'X') {
            result.push_back('K');
            result.push_back('S');
            ++i;
        } else if (ch == 'Y') {
            if (std::string_view{"AEIOU"}.find(next_ch) != std::string_view::npos) result.push_back('Y');
            ++i;
        } else if (ch == 'Z') {
            result.push_back('S');
            ++i;
        } else {
            ++i;
        }
    }

    // Remove duplicate adjacent letters
    luisa::string final;
    for (char c : result) {
        if (final.empty() || c != final.back()) final.push_back(c);
    }
    return final;
}

}// namespace tokenize
