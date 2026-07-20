#include "searcher.h"
#include <cctype>
#include <luisa/vstl/vstring.h>

namespace tokenize {

Searcher::Searcher(const InvertedIndex &index,
                   const NgramTokenizer *tokenizer,
                   double k1,
                   double b,
                   double min_should_match,
                   luisa::string fuzziness,
                   int max_expansions,
                   int prefix_length)
    : _index(index),
      _tokenizer(tokenizer ? *tokenizer : NgramTokenizer()),
      _scorer(index, k1, b),
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

luisa::vector<std::pair<int, double>> Searcher::search(
    luisa::compute::Device &device,
    luisa::compute::Stream &stream,
    luisa::string_view query, int top_k) {
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

    luisa::vector<luisa::string> expanded_tokens;
    for (const auto &expanded : token_expansions) {
        for (const auto &t : expanded) expanded_tokens.push_back(t);
    }
    if (expanded_tokens.empty()) return {};

    // deduplicate expanded tokens
    luisa::vector<luisa::string> deduped;
    luisa::unordered_set<luisa::string> seen2;
    for (auto &t : expanded_tokens) {
        if (!seen2.contains(t)) {
            seen2.insert(t);
            deduped.push_back(std::move(t));
        }
    }

    return _scorer.gpu_score_topk(device, stream, deduped, top_k);
}

}// namespace tokenize
