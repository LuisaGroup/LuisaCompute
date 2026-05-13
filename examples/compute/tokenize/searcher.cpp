#include "searcher.h"
#include <cctype>
#include <luisa/vstl/vstring.h>

namespace tokenize {

Searcher::Searcher(const InvertedIndex &index,
                   const NgramTokenizer *tokenizer,
                   const BM25Scorer *scorer,
                   double k1,
                   double b,
                   double min_should_match,
                   luisa::string fuzziness,
                   int max_expansions,
                   int prefix_length)
    : _index(index),
      _tokenizer(tokenizer ? *tokenizer : NgramTokenizer()),
      _scorer(scorer ? *scorer : BM25Scorer(index, k1, b)),
      _k1(k1), _b(b),
      _min_should_match(min_should_match),
      _fuzziness(std::move(fuzziness)),
      _max_expansions(max_expansions),
      _prefix_length(prefix_length) {
    if (_fuzziness != "0") {
        const_cast<InvertedIndex &>(_index).build_symmetric_delete_index();
    }
}

bool Searcher::is_latin_token(luisa::string_view token) {
    if (token.empty()) return false;
    for (char c : token) {
        if (static_cast<unsigned char>(c) > 127) return false;
    }
    return true;
}

luisa::vector<luisa::string> Searcher::expand_token(luisa::string_view token) {
    if (!is_latin_token(token)) {
        if (_index.has_term(token)) return {luisa::string{token}};
        return {};
    }

    int max_edits = 0;
    if (_fuzziness == "AUTO") {
        max_edits = LevenshteinAutomaton::auto_fuzziness(token);
    } else {
        max_edits = std::stoi(std::string{_fuzziness.data(), _fuzziness.size()});
    }
    if (max_edits == 0) {
        if (_index.has_term(token)) return {luisa::string{token}};
        return {};
    }

    luisa::string cache_key = luisa::string{token} + "|" + luisa::string{_fuzziness} + "|" +
                              vstd::to_string(_prefix_length) + "|" + vstd::to_string(_max_expansions);
    {
        luisa::fiber::lock lck(_expand_cache_mtx);
        auto it = _expand_cache.find(cache_key);
        if (it != _expand_cache.end()) return it->second;
    }

    LevenshteinAutomaton automaton(token, max_edits, _prefix_length);
    auto matches = automaton.match(_index, _max_expansions);
    luisa::vector<luisa::string> result;
    if (matches.empty()) {
        if (_index.has_term(token)) result.push_back(luisa::string{token});
    } else {
        result = std::move(matches);
    }
    {
        luisa::fiber::lock lck(_expand_cache_mtx);
        _expand_cache.emplace(std::move(cache_key), result);
    }
    return result;
}

luisa::vector<std::pair<int, double>> Searcher::search(luisa::string_view query, int top_k) {
    if (_index.N() == 0) return {};

    auto query_tokens = _tokenizer.tokenize(query);
    if (query_tokens.empty()) return {};

    luisa::vector<luisa::string> unique_query;
    luisa::unordered_set<luisa::string> seen;
    for (auto &t : query_tokens) {
        if (!seen.contains(t)) {
            seen.insert(t);
            unique_query.push_back(t);
        }
    }

    int min_match = std::max(1, static_cast<int>(static_cast<double>(unique_query.size()) * _min_should_match));

    // Parallel token expansion: each query token is independent
    luisa::vector<luisa::vector<luisa::string>> token_expansions(unique_query.size());
    luisa::fiber::parallel(static_cast<uint32_t>(unique_query.size()), [&](uint32_t i) noexcept {
        token_expansions[i] = expand_token(unique_query[i]);
    });

    int hits = 0;
    for (const auto &expanded : token_expansions) {
        if (!expanded.empty()) ++hits;
    }
    if (hits < min_match) return {};

    // Parallel posting intersection with thread-local doc counts
    luisa::unordered_set<int> *candidate_docs_ptr = nullptr;
    luisa::unordered_set<int> candidate_docs;
    if (min_match > 1 || (_index.N() > 50000 && min_match >= 1)) {
        struct LocalCounts {
            luisa::unordered_map<int, int> counts;
        };
        luisa::vector<LocalCounts> locals;
        luisa::fiber::mutex mtx;

        luisa::fiber::parallel(static_cast<uint32_t>(token_expansions.size()), [&](uint32_t begin, uint32_t end) noexcept {
            LocalCounts local;
            for (uint32_t idx = begin; idx < end; ++idx) {
                const auto &expanded = token_expansions[idx];
                if (expanded.empty()) continue;
                luisa::vector<luisa::span<const int>> all_docs;
                for (const auto &t : expanded) {
                    auto p = _index.get_postings(t);
                    if (p) all_docs.push_back(p->docs);
                }
                if (all_docs.empty()) continue;
                size_t total_len = 0;
                for (const auto &sp : all_docs) total_len += sp.size();
                if (total_len > 512 || expanded.size() > 2) {
                    luisa::vector<int> concat;
                    concat.reserve(total_len);
                    for (const auto &sp : all_docs) {
                        for (int d : sp) concat.push_back(d);
                    }
                    std::sort(concat.begin(), concat.end());
                    int prev = -1;
                    for (int d : concat) {
                        if (d == prev) continue;
                        prev = d;
                        auto it = local.counts.find(d);
                        if (it == local.counts.end()) local.counts.emplace(d, 1);
                        else ++(it->second);
                    }
                } else {
                    luisa::unordered_set<int> seen_docs;
                    for (const auto &sp : all_docs) {
                        for (int d : sp) {
                            if (!seen_docs.contains(d)) {
                                seen_docs.insert(d);
                                auto it = local.counts.find(d);
                                if (it == local.counts.end()) local.counts.emplace(d, 1);
                                else ++(it->second);
                            }
                        }
                    }
                }
            }
            {
                luisa::fiber::lock lck(mtx);
                locals.push_back(std::move(local));
            }
        });

        luisa::unordered_map<int, int> doc_token_counts;
        for (auto &local : locals) {
            for (auto &[doc_id, count] : local.counts) {
                auto it = doc_token_counts.find(doc_id);
                if (it == doc_token_counts.end()) doc_token_counts.emplace(doc_id, count);
                else it->second += count;
            }
        }

        if (!doc_token_counts.empty()) {
            for (const auto &[doc_id, count] : doc_token_counts) {
                if (count >= min_match) candidate_docs.insert(doc_id);
            }
        }
        candidate_docs_ptr = &candidate_docs;
    }

    luisa::vector<luisa::string> expanded_tokens;
    for (const auto &expanded : token_expansions) {
        for (const auto &t : expanded) expanded_tokens.push_back(t);
    }
    if (expanded_tokens.empty()) return {};
    if (candidate_docs_ptr && candidate_docs.empty()) return {};

    // deduplicate expanded tokens
    luisa::vector<luisa::string> deduped;
    luisa::unordered_set<luisa::string> seen2;
    for (auto &t : expanded_tokens) {
        if (!seen2.contains(t)) {
            seen2.insert(t);
            deduped.push_back(std::move(t));
        }
    }

    return _scorer.score_topk(deduped, top_k, candidate_docs_ptr);
}

}// namespace tokenize
