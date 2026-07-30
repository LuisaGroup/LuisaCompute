#include "levenshtein_automaton.h"
#include <algorithm>
#include <cstdint>
#include <luisa/vstl/vstring.h>
#include <luisa/core/fiber.h>

namespace tokenize {

LevenshteinAutomaton::LevenshteinAutomaton(luisa::string_view pattern, int max_edits, int prefix_length)
    : _pattern(pattern), _max_edits(max_edits), _prefix_length(prefix_length) {
    for (char c : pattern) {
        auto it = _pattern_counts.find(c);
        if (it == _pattern_counts.end()) _pattern_counts.emplace(c, 1);
        else ++(it->second);
    }
    for (const auto &p : _pattern_counts) _pattern_counts_items.push_back(p);
    _pattern_deletes = InvertedIndex::generate_deletes(pattern, max_edits);
}

int LevenshteinAutomaton::auto_fuzziness(luisa::string_view term) noexcept {
    size_t length = term.size();
    if (length <= 2) return 0;
    if (length <= 5) return 1;
    return 2;
}

int LevenshteinAutomaton::damerau_levenshtein(luisa::string_view s, luisa::string_view t) {
    if (s.size() < t.size()) std::swap(s, t);
    size_t m = s.size(), n = t.size();
    if (n == 0) return static_cast<int>(m);
    if (n == 1) return (s[0] == t[0]) ? 0 : 1;
    if (m == 2 && n == 2) {
        if (s == t) return 0;
        if (s[0] == t[0] || s[1] == t[1]) return 1;
        if (s[0] == t[1] && s[1] == t[0]) return 1;
        return 2;
    }

    luisa::vector<int> prev_prev(n + 1), prev(n + 1), curr(n + 1);
    for (size_t j = 0; j <= n; ++j) prev_prev[j] = static_cast<int>(j);
    for (size_t j = 0; j <= n; ++j) prev[j] = static_cast<int>(j);

    for (size_t i = 1; i <= m; ++i) {
        curr[0] = static_cast<int>(i);
        char si_1 = s[i - 1];
        for (size_t j = 1; j <= n; ++j) {
            int cost = (si_1 == t[j - 1]) ? 0 : 1;
            curr[j] = std::min({curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost});
            if (i > 1 && j > 1 && si_1 == t[j - 2] && s[i - 2] == t[j - 1]) {
                curr[j] = std::min(curr[j], prev_prev[j - 2] + 1);
            }
        }
        std::swap(prev_prev, prev);
        std::swap(prev, curr);
    }
    return prev[n];
}

int LevenshteinAutomaton::freq_lower_bound(luisa::string_view term) const {
    int total = 0;
    int matched = 0;
    int term_len = static_cast<int>(term.size());
    if (term_len <= 32) {
        for (const auto &[c, pc] : _pattern_counts_items) {
            int tc_c = 0;
            for (char ch : term) if (ch == c) ++tc_c;
            matched += tc_c;
            if (pc != tc_c) total += std::abs(pc - tc_c);
        }
    } else {
        luisa::unordered_map<char, int> tc;
        for (char ch : term) {
            auto it = tc.find(ch);
            if (it == tc.end()) tc.emplace(ch, 1);
            else ++(it->second);
        }
        for (const auto &[c, pc] : _pattern_counts_items) {
            auto it = tc.find(c);
            int tc_c = (it == tc.end()) ? 0 : it->second;
            matched += tc_c;
            if (pc != tc_c) total += std::abs(pc - tc_c);
        }
    }
    total += term_len - matched;
    return (total + 1) / 2;
}

luisa::vector<luisa::string> LevenshteinAutomaton::match(const InvertedIndex &dictionary, int max_expansions) const {
    luisa::vector<luisa::string> candidates;
    int pattern_len = static_cast<int>(_pattern.size());
    int max_edits = _max_edits;
    int prefix_length = _prefix_length;
    luisa::string prefix = (prefix_length > 0 && !_pattern.empty()) ? luisa::string{_pattern.substr(0, prefix_length)} : luisa::string{};
    bool has_freq_filter = static_cast<int>(_pattern.size()) <= 64;

    // Helper: parallel evaluation of collected candidates
    auto evaluate = [&](const luisa::vector<luisa::string> &cands) -> luisa::vector<luisa::string> {
        if (cands.empty()) return {};
        if (cands.size() == 1) {
            luisa::vector<luisa::string> results;
            const auto &term = cands[0];
            if (!has_freq_filter || freq_lower_bound(term) <= max_edits) {
                if (damerau_levenshtein(_pattern, term) <= max_edits) {
                    results.push_back(term);
                }
            }
            return results;
        }

        luisa::vector<luisa::vector<luisa::string>> local_results;
        luisa::fiber::mutex mtx;

        luisa::fiber::parallel(static_cast<uint32_t>(cands.size()), [&](uint32_t begin, uint32_t end) noexcept {
            luisa::vector<luisa::string> local;
            for (uint32_t i = begin; i < end; ++i) {
                const auto &term = cands[i];
                if (has_freq_filter && freq_lower_bound(term) > max_edits) continue;
                if (damerau_levenshtein(_pattern, term) <= max_edits) {
                    local.push_back(term);
                }
            }
            {
                luisa::fiber::lock lck(mtx);
                local_results.push_back(std::move(local));
            }
        });

        luisa::vector<luisa::string> results;
        results.reserve(static_cast<size_t>(max_expansions));
        for (auto &local : local_results) {
            for (auto &term : local) {
                results.push_back(std::move(term));
                if (static_cast<int>(results.size()) >= max_expansions) break;
            }
            if (static_cast<int>(results.size()) >= max_expansions) break;
        }
        return results;
    };

    // Try symmetric delete index
    if (prefix_length == 1 && max_edits > 0) {
        const_cast<InvertedIndex &>(dictionary).build_symmetric_delete_index();
        const auto &sd = dictionary.symmetric_delete_index(max_edits);
        if (!sd.empty()) {
            luisa::unordered_set<luisa::string> cand_set;
            for (const auto &variant : _pattern_deletes) {
                auto it = sd.find(variant);
                if (it != sd.end()) {
                    for (const auto &term : it->second) cand_set.insert(term);
                }
            }
            if (dictionary.has_term(_pattern)) cand_set.insert(_pattern);
            for (const auto &term : cand_set) {
                int term_len = static_cast<int>(term.size());
                if (std::abs(term_len - pattern_len) > max_edits) continue;
                if (!prefix.empty() && (term.empty() || luisa::string_view{term}.substr(0, 1) != prefix)) continue;
                candidates.push_back(term);
            }
            return evaluate(candidates);
        }
    }

    // Use terms_by_length_prefix
    const auto &terms_by_prefix = dictionary.terms_by_length_prefix();
    if (!terms_by_prefix.empty()) {
        for (int length = std::max(pattern_len - max_edits, prefix_length);
             length <= pattern_len + max_edits; ++length) {
            luisa::string key = vstd::to_string(length) + ":" + prefix;
            auto it = terms_by_prefix.find(key);
            luisa::vector<luisa::string> bucket;
            if (it != terms_by_prefix.end()) {
                bucket = it->second;
            } else if (prefix_length > 0) {
                continue;
            } else {
                const auto &by_len = dictionary.terms_by_length();
                auto lit = by_len.find(length);
                if (lit != by_len.end()) bucket = lit->second;
            }
            for (const auto &term : bucket) {
                if (prefix_length > 1) {
                    if (static_cast<int>(term.size()) < prefix_length) continue;
                    if (luisa::string_view{term}.substr(0, prefix_length) != prefix) continue;
                }
                candidates.push_back(term);
            }
        }
        return evaluate(candidates);
    }

    // Fallback: iterate all terms
    const auto &by_len = dictionary.terms_by_length();
    if (!by_len.empty()) {
        for (int length = std::max(pattern_len - max_edits, prefix_length);
             length <= pattern_len + max_edits; ++length) {
            auto lit = by_len.find(length);
            if (lit == by_len.end()) continue;
            for (const auto &term : lit->second) {
                if (prefix_length == 1) {
                    if (term.empty() || term[0] != prefix[0]) continue;
                } else if (prefix_length > 0) {
                    if (static_cast<int>(term.size()) < prefix_length) continue;
                    if (luisa::string_view{term}.substr(0, prefix_length) != prefix) continue;
                }
                candidates.push_back(term);
            }
        }
        return evaluate(candidates);
    }

    // Ultimate fallback
    auto all_terms = dictionary.terms();
    for (const auto &term : all_terms) {
        int term_len = static_cast<int>(term.size());
        if (std::abs(term_len - pattern_len) > max_edits) continue;
        if (prefix_length == 1) {
            if (term.empty() || term[0] != prefix[0]) continue;
        } else if (prefix_length > 0) {
            if (term_len < prefix_length) continue;
            if (luisa::string_view{term}.substr(0, prefix_length) != prefix) continue;
        }
        candidates.push_back(term);
    }
    return evaluate(candidates);
}

}// namespace tokenize
