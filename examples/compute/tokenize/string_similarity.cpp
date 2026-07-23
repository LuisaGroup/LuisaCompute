#include "string_similarity.h"
#include <algorithm>
#include <cmath>

namespace tokenize {

double jaro_similarity(luisa::string_view s, luisa::string_view t) {
    if (s == t) return 1.0;
    size_t len_s = s.size(), len_t = t.size();
    if (len_s == 0 || len_t == 0) return 0.0;
    size_t match_distance = std::max(len_s, len_t) / 2 - 1;
    luisa::vector<bool> s_matches(len_s, false), t_matches(len_t, false);
    size_t matches = 0;
    for (size_t i = 0; i < len_s; ++i) {
        size_t start = (i >= match_distance) ? (i - match_distance) : 0;
        size_t end = std::min(i + match_distance + 1, len_t);
        for (size_t j = start; j < end; ++j) {
            if (t_matches[j] || s[i] != t[j]) continue;
            s_matches[i] = true;
            t_matches[j] = true;
            ++matches;
            break;
        }
    }
    if (matches == 0) return 0.0;
    size_t transpositions = 0;
    size_t k = 0;
    for (size_t i = 0; i < len_s; ++i) {
        if (!s_matches[i]) continue;
        while (!t_matches[k]) ++k;
        if (s[i] != t[k]) ++transpositions;
        ++k;
    }
    return (static_cast<double>(matches) / static_cast<double>(len_s) +
            static_cast<double>(matches) / static_cast<double>(len_t) +
            (static_cast<double>(matches) - static_cast<double>(transpositions) / 2.0) / static_cast<double>(matches)) /
           3.0;
}

double jaro_winkler_similarity(luisa::string_view s, luisa::string_view t, double p, int max_prefix) {
    double jaro = jaro_similarity(s, t);
    int prefix = 0;
    int limit = static_cast<int>(std::min({static_cast<size_t>(max_prefix), s.size(), t.size()}));
    for (int i = 0; i < limit; ++i) {
        if (s[i] == t[i]) ++prefix;
        else break;
    }
    return jaro + prefix * p * (1.0 - jaro);
}

double sorensen_dice_coefficient(luisa::string_view s, luisa::string_view t) {
    if (s.empty() && t.empty()) return 1.0;
    if (s.empty() || t.empty()) return 0.0;
    luisa::unordered_set<luisa::string> s_bigrams, t_bigrams;
    if (s.size() >= 2) {
        for (size_t i = 0; i + 2 <= s.size(); ++i) s_bigrams.emplace(s.substr(i, 2));
    } else {
        s_bigrams.emplace(s);
    }
    if (t.size() >= 2) {
        for (size_t i = 0; i + 2 <= t.size(); ++i) t_bigrams.emplace(t.substr(i, 2));
    } else {
        t_bigrams.emplace(t);
    }
    size_t intersection = 0;
    for (const auto &bg : s_bigrams) {
        if (t_bigrams.contains(bg)) ++intersection;
    }
    size_t denom = s_bigrams.size() + t_bigrams.size();
    return denom == 0 ? 0.0 : 2.0 * static_cast<double>(intersection) / static_cast<double>(denom);
}

double ngram_overlap(luisa::string_view s, luisa::string_view t, int n) {
    if (s.empty() || t.empty()) return 0.0;
    luisa::unordered_set<luisa::string> s_grams, t_grams;
    if (static_cast<int>(s.size()) >= n) {
        for (size_t i = 0; i + n <= s.size(); ++i) s_grams.emplace(s.substr(i, n));
    } else {
        s_grams.emplace(s);
    }
    if (static_cast<int>(t.size()) >= n) {
        for (size_t i = 0; i + n <= t.size(); ++i) t_grams.emplace(t.substr(i, n));
    } else {
        t_grams.emplace(t);
    }
    size_t intersection = 0;
    for (const auto &g : s_grams) {
        if (t_grams.contains(g)) ++intersection;
    }
    size_t un = s_grams.size() + t_grams.size() - intersection;
    return un == 0 ? 0.0 : static_cast<double>(intersection) / static_cast<double>(un);
}

double jaccard_similarity_tokens(const luisa::unordered_set<luisa::string> &a,
                                  const luisa::unordered_set<luisa::string> &b) {
    size_t intersection = 0;
    for (const auto &x : a) {
        if (b.contains(x)) ++intersection;
    }
    size_t un = a.size() + b.size() - intersection;
    return un == 0 ? 0.0 : static_cast<double>(intersection) / static_cast<double>(un);
}

int hamming_distance(luisa::string_view s, luisa::string_view t) {
    if (s.size() != t.size()) return -1;
    int dist = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != t[i]) ++dist;
    }
    return dist;
}

double cosine_similarity_tfidf(const luisa::unordered_map<luisa::string, double> &vec_a,
                                const luisa::unordered_map<luisa::string, double> &vec_b) {
    if (vec_a.empty() || vec_b.empty()) return 0.0;
    double dot = 0.0;
    double norm_a = 0.0;
    for (const auto &[term, w] : vec_a) {
        norm_a += w * w;
        auto it = vec_b.find(term);
        if (it != vec_b.end()) dot += w * it->second;
    }
    double norm_b = 0.0;
    for (const auto &[term, w] : vec_b) norm_b += w * w;
    double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return denom == 0.0 ? 0.0 : dot / denom;
}

}// namespace tokenize
