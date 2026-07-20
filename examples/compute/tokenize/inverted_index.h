#pragma once

#include <luisa/core/stl.h>
#include <cstdint>
#include <filesystem>

namespace tokenize {

class InvertedIndex {
public:
    struct PostingsResult {
        luisa::span<const int> docs;
        luisa::span<const uint16_t> tfs;
    };

    InvertedIndex() = default;

    void add_document(int doc_id, const luisa::vector<luisa::string> &tokens);
    void finalize(double stop_threshold = 0.5, luisa::optional<int> prune_df = luisa::nullopt);

    [[nodiscard]] int N() const noexcept { return _N; }
    [[nodiscard]] double avgdl() const noexcept { return _avgdl; }
    [[nodiscard]] const luisa::vector<int> &doc_lengths() const noexcept { return _doc_lengths; }

    [[nodiscard]] luisa::optional<PostingsResult> get_postings(luisa::string_view term) const;
    [[nodiscard]] int doc_freq(luisa::string_view term) const;
    [[nodiscard]] bool has_term(luisa::string_view term) const;
    [[nodiscard]] luisa::vector<luisa::string> terms() const;

    void save(const std::filesystem::path &path, bool include_forward_index = false) const;
    void load(const std::filesystem::path &path);

    // accessors for fuzzy matching
    [[nodiscard]] const luisa::unordered_map<int, luisa::vector<luisa::string>> &terms_by_length() const noexcept { return _terms_by_length; }
    [[nodiscard]] const luisa::unordered_map<luisa::string, luisa::vector<luisa::string>> &terms_by_length_prefix() const noexcept { return _terms_by_length_prefix; }
    void build_symmetric_delete_index(int sd_max_len = 8);
    [[nodiscard]] const luisa::unordered_map<luisa::string, luisa::unordered_set<luisa::string>> &
    symmetric_delete_index(int max_edits) const;

    [[nodiscard]] const luisa::unordered_map<luisa::string, int> &term_to_id() const noexcept { return _term_to_id; }
    [[nodiscard]] double collection_lm(luisa::string_view term) const;
    [[nodiscard]] int term_collection_freq(int tid) const;
    [[nodiscard]] const luisa::vector<luisa::unordered_map<luisa::string, int>> &doc_term_freqs() const noexcept { return _doc_term_freqs; }
    [[nodiscard]] const luisa::unordered_set<luisa::string> &doc_token_set(int doc_id) const;

    static luisa::unordered_set<luisa::string> generate_deletes(luisa::string_view term, int max_edits);

private:
    struct PostingList {
        luisa::vector<int> doc_ids;
        luisa::vector<uint16_t> tfs;
    };

    [[nodiscard]] bool is_stop_ngram(luisa::string_view token, int df, double threshold) const;
    void ensure_term_to_id();

    luisa::unordered_map<luisa::string, int> _term_to_id;
    luisa::unordered_map<luisa::string, PostingList> _temp_postings;
    luisa::vector<int> _doc_lengths;
    luisa::vector<int> _doc_lengths_arr;
    int64_t _sum_doc_lengths = 0;
    int _N = 0;
    double _avgdl = 0.0;

    // finalized flat arrays
    luisa::vector<int> _posting_docs_data;
    luisa::vector<int> _posting_docs_ptr{0};
    luisa::vector<uint16_t> _posting_tfs_data;
    luisa::vector<int> _posting_tfs_ptr{0};
    bool _finalized = false;

    luisa::unordered_map<int, luisa::vector<luisa::string>> _terms_by_length;
    luisa::unordered_map<luisa::string, luisa::vector<luisa::string>> _terms_by_length_prefix;
    luisa::unordered_map<int, luisa::unordered_map<luisa::string, luisa::unordered_set<luisa::string>>> _symmetric_delete_index;

    luisa::vector<luisa::unordered_map<luisa::string, int>> _doc_term_freqs;
    mutable luisa::vector<luisa::optional<luisa::unordered_set<luisa::string>>> _doc_token_strs;

    luisa::unordered_map<luisa::string, double> _collection_lm_cache;
    luisa::vector<int> _term_collection_freqs;

    bool _postings_sorted = true;
    int _last_doc_id = -1;
    bool _term_id_dirty = true;

    static constexpr std::array<uint8_t, 4> MAGIC = {'K', 'I', 'M', 'X'};
    static constexpr uint8_t VERSION = 2;
};

}// namespace tokenize
