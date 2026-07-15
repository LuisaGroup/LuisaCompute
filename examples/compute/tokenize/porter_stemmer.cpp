#include "porter_stemmer.h"
#include <cctype>

namespace tokenize {

static bool is_vowel(char ch, char prev) {
    char lc = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (lc == 'a' || lc == 'e' || lc == 'i' || lc == 'o' || lc == 'u') return true;
    return lc == 'y' && (prev == '\0' || std::string_view{"aeiou"}.find(std::tolower(static_cast<unsigned char>(prev))) == std::string_view::npos);
}

static int measure(const luisa::string &stem) {
    size_t n = stem.size();
    if (n == 0) return 0;
    luisa::string seq;
    seq.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        char prev = (i > 0) ? stem[i - 1] : '\0';
        seq.push_back(is_vowel(stem[i], prev) ? 'V' : 'C');
    }
    int m = 0;
    char prev = seq[0];
    for (size_t i = 1; i < seq.size(); ++i) {
        if (prev == 'C' && seq[i] == 'V') ++m;
        prev = seq[i];
    }
    return m;
}

static bool ends_with(const luisa::string &word, luisa::string_view suffix) {
    return word.size() >= suffix.size() && luisa::string_view{word}.substr(word.size() - suffix.size()) == suffix;
}

static luisa::string replace_suffix(const luisa::string &word, luisa::string_view suffix, luisa::string_view repl) {
    return luisa::string{word.substr(0, word.size() - suffix.size())} + luisa::string{repl};
}

static bool ends_cvc(const luisa::string &word) {
    if (word.size() < 3) return false;
    char a = word[word.size() - 3];
    char b = word[word.size() - 2];
    char c = word[word.size() - 1];
    if (std::string_view{"wxy"}.find(std::tolower(static_cast<unsigned char>(c))) != std::string_view::npos) return false;
    return !is_vowel(a, '\0') && is_vowel(b, a) && !is_vowel(c, b);
}

luisa::string porter_stem(luisa::string_view word) {
    luisa::string w{word};
    for (char &c : w) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (w.size() <= 2) return w;

    // Step 1a
    if (ends_with(w, "sses")) {
        w = replace_suffix(w, "sses", "ss");
    } else if (ends_with(w, "ies")) {
        w = replace_suffix(w, "ies", "i");
    } else if (ends_with(w, "s") && !ends_with(w, "ss")) {
        w.pop_back();
    }

    // Step 1b
    bool step1b_done = false;
    if (ends_with(w, "eed")) {
        luisa::string stem = w.substr(0, w.size() - 3);
        if (measure(stem) > 0) w = stem + "ee";
    } else if (ends_with(w, "ed")) {
        luisa::string stem = w.substr(0, w.size() - 2);
        bool has_vowel = false;
        for (size_t i = 0; i < stem.size(); ++i) {
            char prev = (i > 0) ? stem[i - 1] : '\0';
            if (is_vowel(stem[i], prev)) { has_vowel = true; break; }
        }
        if (has_vowel) {
            w = std::move(stem);
            step1b_done = true;
        }
    } else if (ends_with(w, "ing")) {
        luisa::string stem = w.substr(0, w.size() - 3);
        bool has_vowel = false;
        for (size_t i = 0; i < stem.size(); ++i) {
            char prev = (i > 0) ? stem[i - 1] : '\0';
            if (is_vowel(stem[i], prev)) { has_vowel = true; break; }
        }
        if (has_vowel) {
            w = std::move(stem);
            step1b_done = true;
        }
    }

    if (step1b_done) {
        if (ends_with(w, "at")) w += "e";
        else if (ends_with(w, "bl")) w += "e";
        else if (ends_with(w, "iz")) w += "e";
        else if (w.size() >= 2) {
            char last = w.back();
            char prev = w[w.size() - 2];
            if (last == prev && !is_vowel(last, '\0') && std::string_view{"lsz"}.find(last) == std::string_view::npos) {
                w.pop_back();
            } else if (measure(w) == 1 && ends_cvc(w)) {
                w += "e";
            }
        }
    }

    // Step 1c
    if (ends_with(w, "y")) {
        luisa::string stem = w.substr(0, w.size() - 1);
        bool has_vowel = false;
        for (size_t i = 0; i < stem.size(); ++i) {
            char prev = (i > 0) ? stem[i - 1] : '\0';
            if (is_vowel(stem[i], prev)) { has_vowel = true; break; }
        }
        if (has_vowel) {
            w = std::move(stem);
            w += "i";
        }
    }

    // Step 2
    struct SuffixRepl { luisa::string_view suffix; luisa::string_view repl; };
    static const SuffixRepl step2_map[] = {
        {"ational", "ate"}, {"tional", "tion"}, {"enci", "ence"}, {"anci", "ance"},
        {"izer", "ize"}, {"abli", "able"}, {"alli", "al"}, {"entli", "ent"},
        {"eli", "e"}, {"ousli", "ous"}, {"ization", "ize"}, {"ation", "ate"},
        {"ator", "ate"}, {"alism", "al"}, {"iveness", "ive"}, {"fulness", "ful"},
        {"ousness", "ous"}, {"aliti", "al"}, {"iviti", "ive"}, {"biliti", "ble"}
    };
    for (const auto &sr : step2_map) {
        if (ends_with(w, sr.suffix)) {
            luisa::string stem = w.substr(0, w.size() - sr.suffix.size());
            if (measure(stem) > 0) w = stem + luisa::string{sr.repl};
            break;
        }
    }

    // Step 3
    static const SuffixRepl step3_map[] = {
        {"icate", "ic"}, {"ative", ""}, {"alize", "al"}, {"iciti", "ic"},
        {"ical", "ic"}, {"ful", ""}, {"ness", ""}
    };
    for (const auto &sr : step3_map) {
        if (ends_with(w, sr.suffix)) {
            luisa::string stem = w.substr(0, w.size() - sr.suffix.size());
            if (measure(stem) > 0) w = stem + luisa::string{sr.repl};
            break;
        }
    }

    // Step 4
    static const luisa::string_view step4_suffixes[] = {
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
        "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
        "ous", "ive", "ize"
    };
    for (const auto &suffix : step4_suffixes) {
        if (ends_with(w, suffix)) {
            if (suffix == "ion" && w.size() > 3) {
                char before = w[w.size() - 4];
                if (before == 's' || before == 't') {
                    luisa::string stem = w.substr(0, w.size() - 3);
                    if (measure(stem) > 1) w = std::move(stem);
                    break;
                }
            }
            luisa::string stem = w.substr(0, w.size() - suffix.size());
            if (measure(stem) > 1) w = std::move(stem);
            break;
        }
    }

    // Step 5a
    if (ends_with(w, "e")) {
        luisa::string stem = w.substr(0, w.size() - 1);
        if (measure(stem) > 1) {
            w = std::move(stem);
        } else if (measure(stem) == 1 && !ends_cvc(stem)) {
            w = std::move(stem);
        }
    }

    // Step 5b
    if (measure(w) > 1 && w.size() >= 2 && w.back() == 'l' && w[w.size() - 2] == 'l') {
        w.pop_back();
    }

    return w;
}

}// namespace tokenize
