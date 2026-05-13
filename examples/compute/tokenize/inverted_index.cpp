#include "inverted_index.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <luisa/core/logging.h>
#include <luisa/core/fiber.h>
#include <luisa/vstl/vstring.h>

namespace tokenize {

void InvertedIndex::add_document(int doc_id, const luisa::vector<luisa::string> &tokens) {
    if (_finalized) {
        LUISA_WARNING("Cannot add documents after finalize().");
        return;
    }
    luisa::unordered_map<luisa::string, int> counter;
    for (const auto &t : tokens) {
        auto it = counter.find(t);
        if (it == counter.end()) counter.emplace(t, 1);
        else ++(it->second);
    }
    _doc_lengths.push_back(static_cast<int>(tokens.size()));
    _sum_doc_lengths += static_cast<int>(tokens.size());
    _doc_term_freqs.push_back(counter);

    bool sequential = doc_id >= _last_doc_id;
    for (const auto &[token, freq] : counter) {
        auto it = _temp_postings.find(token);
        if (it == _temp_postings.end()) {
            PostingList pl;
            pl.doc_ids.push_back(doc_id);
            pl.tfs.push_back(static_cast<uint16_t>(freq));
            _temp_postings.emplace(token, std::move(pl));
        } else {
            if (!sequential && !it->second.doc_ids.empty() && it->second.doc_ids.back() > doc_id) {
                _postings_sorted = false;
            }
            it->second.doc_ids.push_back(doc_id);
            it->second.tfs.push_back(static_cast<uint16_t>(freq));
        }
    }
    if (doc_id >= _N) _N = doc_id + 1;
    _last_doc_id = doc_id;
    _term_id_dirty = true;
}

bool InvertedIndex::is_stop_ngram(luisa::string_view token, int df, double threshold) const {
    if (token.empty()) return true;
    if (df > static_cast<int>(_N * threshold)) return true;
    bool all_alpha = true;
    bool any_non_punct = false;
    for (char c : token) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalpha(uc)) {
            any_non_punct = true;
        } else if (!std::ispunct(uc)) {
            any_non_punct = true;
        }
        if (!std::isalpha(uc)) all_alpha = false;
    }
    if (all_alpha) return false;
    return !any_non_punct;
}

luisa::unordered_set<luisa::string> InvertedIndex::generate_deletes(luisa::string_view term, int max_edits) {
    if (max_edits == 0 || term.empty()) {
        luisa::unordered_set<luisa::string> result;
        result.emplace(term);
        return result;
    }
    if (max_edits == 1) {
        luisa::unordered_set<luisa::string> result;
        result.emplace(term);
        for (size_t i = 0; i < term.size(); ++i) {
            luisa::string del;
            if (i > 0) del.append(term.substr(0, i));
            if (i + 1 < term.size()) del.append(term.substr(i + 1));
            result.emplace(std::move(del));
        }
        return result;
    }
    luisa::unordered_set<luisa::string> deletes;
    deletes.emplace(term);
    for (int e = 0; e < max_edits; ++e) {
        luisa::unordered_set<luisa::string> new_deletes;
        for (const auto &t : deletes) {
            for (size_t i = 0; i < t.size(); ++i) {
                luisa::string del;
                if (i > 0) del.append(t.substr(0, i));
                if (i + 1 < t.size()) del.append(t.substr(i + 1));
                new_deletes.emplace(std::move(del));
            }
        }
        for (auto &nd : new_deletes) {
            deletes.emplace(std::move(nd));
        }
    }
    return deletes;
}

void InvertedIndex::build_symmetric_delete_index(int sd_max_len) {
    if (!_symmetric_delete_index.empty()) return;
    if (_term_to_id.empty()) {
        _symmetric_delete_index[1] = {};
        _symmetric_delete_index[2] = {};
        return;
    }

    luisa::vector<luisa::string> terms;
    terms.reserve(_term_to_id.size());
    for (const auto &[term, tid] : _term_to_id) {
        (void)tid;
        terms.push_back(term);
    }

    struct SDLocal {
        luisa::unordered_map<luisa::string, luisa::vector<luisa::string>> sd1;
        luisa::unordered_map<luisa::string, luisa::vector<luisa::string>> sd2;
    };
    luisa::vector<SDLocal> locals;
    luisa::fiber::mutex mtx;

    luisa::fiber::parallel(static_cast<uint32_t>(terms.size()), [&](uint32_t begin, uint32_t end) noexcept {
        SDLocal local;
        for (uint32_t i = begin; i < end; ++i) {
            const auto &term = terms[i];
            if (static_cast<int>(term.size()) > sd_max_len) continue;
            auto del1 = generate_deletes(term, 1);
            for (const auto &v : del1) {
                if (v != term) local.sd1[v].push_back(term);
            }
            auto del2 = generate_deletes(term, 2);
            for (const auto &v : del2) {
                if (v != term) local.sd2[v].push_back(term);
            }
        }
        {
            luisa::fiber::lock lck(mtx);
            locals.push_back(std::move(local));
        }
    });

    luisa::unordered_map<luisa::string, luisa::vector<luisa::string>> sd1_merged;
    luisa::unordered_map<luisa::string, luisa::vector<luisa::string>> sd2_merged;
    for (auto &local : locals) {
        for (auto &[k, v] : local.sd1) {
            auto &dest = sd1_merged[k];
            dest.insert(dest.end(), v.begin(), v.end());
        }
        for (auto &[k, v] : local.sd2) {
            auto &dest = sd2_merged[k];
            dest.insert(dest.end(), v.begin(), v.end());
        }
    }

    luisa::unordered_map<luisa::string, luisa::unordered_set<luisa::string>> sd1set;
    luisa::unordered_map<luisa::string, luisa::unordered_set<luisa::string>> sd2set;
    for (auto &[k, v] : sd1_merged) {
        luisa::unordered_set<luisa::string> s;
        for (auto &term : v) s.emplace(std::move(term));
        sd1set.emplace(k, std::move(s));
    }
    for (auto &[k, v] : sd2_merged) {
        luisa::unordered_set<luisa::string> s;
        for (auto &term : v) s.emplace(std::move(term));
        sd2set.emplace(k, std::move(s));
    }
    _symmetric_delete_index[1] = std::move(sd1set);
    _symmetric_delete_index[2] = std::move(sd2set);
}

const luisa::unordered_map<luisa::string, luisa::unordered_set<luisa::string>> &
InvertedIndex::symmetric_delete_index(int max_edits) const {
    auto it = _symmetric_delete_index.find(max_edits);
    if (it != _symmetric_delete_index.end()) return it->second;
    static const luisa::unordered_map<luisa::string, luisa::unordered_set<luisa::string>> empty;
    return empty;
}

void InvertedIndex::finalize(double stop_threshold, luisa::optional<int> prune_df) {
    if (_finalized) return;

    // --- Parallel posting-list pruning and sorting ---
    struct TermWork {
        luisa::string_view token;
        PostingList *pl;
        bool keep = false;
    };
    luisa::vector<TermWork> works;
    works.reserve(_temp_postings.size());
    for (auto &[token, pl] : _temp_postings) {
        works.push_back({token, &pl, false});
    }

    luisa::fiber::parallel(static_cast<uint32_t>(works.size()), [&](uint32_t i) noexcept {
        auto &w = works[i];
        int df = static_cast<int>(w.pl->doc_ids.size());
        if (is_stop_ngram(w.token, df, stop_threshold)) return;
        if (prune_df.has_value() && df > prune_df.value()) return;
        w.keep = true;
        if (!_postings_sorted && df > 1) {
            luisa::vector<std::pair<int, uint16_t>> postings;
            postings.reserve(df);
            for (size_t j = 0; j < w.pl->doc_ids.size(); ++j) {
                postings.emplace_back(w.pl->doc_ids[j], w.pl->tfs[j]);
            }
            luisa::sort(postings.begin(), postings.end(), [](const auto &a, const auto &b) {
                return a.first < b.first;
            });
            for (size_t j = 0; j < postings.size(); ++j) {
                w.pl->doc_ids[j] = postings[j].first;
                w.pl->tfs[j] = postings[j].second;
            }
        }
    });

    luisa::unordered_map<luisa::string, int> kept_terms;
    luisa::vector<luisa::vector<int>> kept_doc_ids;
    luisa::vector<luisa::vector<uint16_t>> kept_tfs;
    for (auto &w : works) {
        if (!w.keep) continue;
        int tid = static_cast<int>(kept_terms.size());
        kept_terms.emplace(luisa::string{w.token}, tid);
        kept_doc_ids.push_back(std::move(w.pl->doc_ids));
        kept_tfs.push_back(std::move(w.pl->tfs));
    }

    size_t total_postings = 0;
    for (const auto &p : kept_doc_ids) total_postings += p.size();

    _posting_docs_data.resize(total_postings);
    _posting_tfs_data.resize(total_postings);
    _posting_docs_ptr.resize(kept_doc_ids.size() + 1);
    _posting_tfs_ptr.resize(kept_tfs.size() + 1);
    _posting_docs_ptr[0] = 0;
    _posting_tfs_ptr[0] = 0;

    size_t idx = 0;
    for (size_t i = 0; i < kept_doc_ids.size(); ++i) {
        size_t n = kept_doc_ids[i].size();
        for (size_t j = 0; j < n; ++j) {
            _posting_docs_data[idx + j] = kept_doc_ids[i][j];
            _posting_tfs_data[idx + j] = kept_tfs[i][j];
        }
        idx += n;
        _posting_docs_ptr[i + 1] = static_cast<int>(idx);
        _posting_tfs_ptr[i + 1] = static_cast<int>(idx);
    }

    _term_to_id = std::move(kept_terms);

    // --- Parallel auxiliary map construction (thread-local buckets + merge) ---
    {
        struct AuxMaps {
            luisa::unordered_map<int, luisa::vector<luisa::string>> terms_by_length;
            luisa::unordered_map<luisa::string, luisa::vector<luisa::string>> terms_by_length_prefix;
        };
        luisa::vector<AuxMaps> locals;
        luisa::fiber::mutex aux_mtx;

        luisa::vector<luisa::string> all_terms;
        all_terms.reserve(_term_to_id.size());
        for (const auto &[term, tid] : _term_to_id) {
            (void)tid;
            all_terms.push_back(term);
        }

        luisa::fiber::parallel(static_cast<uint32_t>(all_terms.size()), [&](uint32_t begin, uint32_t end) noexcept {
            AuxMaps local;
            for (uint32_t i = begin; i < end; ++i) {
                const auto &term = all_terms[i];
                int len = static_cast<int>(term.size());
                local.terms_by_length[len].push_back(term);
                luisa::string prefix_key = vstd::to_string(len) + ":" + (term.empty() ? luisa::string{} : luisa::string{term.substr(0, 1)});
                local.terms_by_length_prefix[prefix_key].push_back(term);
            }
            {
                luisa::fiber::lock lck(aux_mtx);
                locals.push_back(std::move(local));
            }
        });

        _terms_by_length.clear();
        _terms_by_length_prefix.clear();
        for (auto &local : locals) {
            for (auto &[len, terms] : local.terms_by_length) {
                auto &dest = _terms_by_length[len];
                dest.insert(dest.end(), terms.begin(), terms.end());
            }
            for (auto &[key, terms] : local.terms_by_length_prefix) {
                auto &dest = _terms_by_length_prefix[key];
                dest.insert(dest.end(), terms.begin(), terms.end());
            }
        }
    }

    if (!_doc_lengths.empty()) {
        _avgdl = static_cast<double>(_sum_doc_lengths) / static_cast<double>(_doc_lengths.size());
        _doc_lengths_arr = _doc_lengths;
    }

    // --- Parallel forward-index rebuild ---
    if (static_cast<int>(_term_to_id.size()) < static_cast<int>(_temp_postings.size())) {
        luisa::fiber::parallel(static_cast<uint32_t>(_doc_term_freqs.size()), [&](uint32_t d) noexcept {
            luisa::unordered_map<luisa::string, int> filtered;
            for (const auto &[t, f] : _doc_term_freqs[d]) {
                if (_term_to_id.contains(t)) filtered.emplace(t, f);
            }
            _doc_term_freqs[d] = std::move(filtered);
        });
    }

    // --- Parallel collection frequency computation ---
    _term_collection_freqs.resize(_term_to_id.size(), 0);
    _collection_lm_cache.clear();
    if (_sum_doc_lengths > 0) {
        luisa::vector<luisa::string> terms_by_id(_term_to_id.size());
        for (const auto &[term, tid] : _term_to_id) {
            terms_by_id[tid] = term;
        }
        luisa::fiber::parallel(static_cast<uint32_t>(_term_to_id.size()), [&](uint32_t tid) noexcept {
            int start = _posting_tfs_ptr[tid];
            int end = _posting_tfs_ptr[tid + 1];
            int cf = 0;
            for (int i = start; i < end; ++i) cf += _posting_tfs_data[i];
            _term_collection_freqs[tid] = cf;
        });
        for (size_t tid = 0; tid < terms_by_id.size(); ++tid) {
            int cf = _term_collection_freqs[tid];
            _collection_lm_cache[terms_by_id[tid]] = static_cast<double>(cf) / static_cast<double>(_sum_doc_lengths);
        }
    }

    _term_id_dirty = false;
    _temp_postings.clear();
    _finalized = true;
}

void InvertedIndex::ensure_term_to_id() {
    if (!_term_id_dirty) return;
    _term_to_id.clear();
    int i = 0;
    for (const auto &[token, pl] : _temp_postings) {
        (void)pl;
        _term_to_id.emplace(token, i++);
    }
    _term_id_dirty = false;
}

luisa::optional<InvertedIndex::PostingsResult> InvertedIndex::get_postings(luisa::string_view term) const {
    if (!_finalized) {
        const_cast<InvertedIndex *>(this)->finalize();
    }
    auto it = _term_to_id.find(luisa::string{term});
    if (it == _term_to_id.end()) return luisa::nullopt;
    int tid = it->second;
    int start = _posting_docs_ptr[tid];
    int end = _posting_docs_ptr[tid + 1];
    return PostingsResult{
        luisa::span<const int>(_posting_docs_data.data() + start, end - start),
        luisa::span<const uint16_t>(_posting_tfs_data.data() + start, end - start)};
}

int InvertedIndex::doc_freq(luisa::string_view term) const {
    auto p = get_postings(term);
    if (!p) return 0;
    return static_cast<int>(p->docs.size());
}

bool InvertedIndex::has_term(luisa::string_view term) const {
    if (!_finalized) {
        return _temp_postings.contains(luisa::string{term});
    }
    return _term_to_id.contains(luisa::string{term});
}

luisa::vector<luisa::string> InvertedIndex::terms() const {
    luisa::vector<luisa::string> result;
    if (!_finalized) {
        result.reserve(_temp_postings.size());
        for (const auto &[k, v] : _temp_postings) {
            (void)v;
            result.push_back(k);
        }
    } else {
        result.reserve(_term_to_id.size());
        for (const auto &[k, v] : _term_to_id) {
            (void)v;
            result.push_back(k);
        }
    }
    return result;
}

void InvertedIndex::save(const std::filesystem::path &path, bool include_forward_index) const {
    if (!_finalized) {
        const_cast<InvertedIndex *>(this)->finalize();
    }
    std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    if (!f) return;

    auto write_u8 = [&](uint8_t v) { f.write(reinterpret_cast<const char *>(&v), 1); };
    auto write_u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char *>(&v), sizeof(v)); };
    auto write_u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char *>(&v), sizeof(v)); };
    auto write_f64 = [&](double v) { f.write(reinterpret_cast<const char *>(&v), sizeof(v)); };

    f.write(reinterpret_cast<const char *>(MAGIC.data()), 4);
    write_u8(VERSION);
    write_u32(static_cast<uint32_t>(_N));
    write_u32(static_cast<uint32_t>(_doc_lengths.size()));
    write_f64(_avgdl);
    write_u32(static_cast<uint32_t>(_term_to_id.size()));

    luisa::vector<luisa::string> term_list;
    term_list.reserve(_term_to_id.size());
    for (const auto &[term, tid] : _term_to_id) {
        (void)tid;
        term_list.push_back(term);
    }
    luisa::sort(term_list.begin(), term_list.end(), [&](const auto &a, const auto &b) {
        return _term_to_id.at(a) < _term_to_id.at(b);
    });

    for (const auto &term : term_list) {
        auto bytes = luisa::string_view{term};
        write_u16(static_cast<uint16_t>(bytes.size()));
        f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    if (!_doc_lengths.empty()) {
        f.write(reinterpret_cast<const char *>(_doc_lengths.data()), _doc_lengths.size() * sizeof(int));
    }

    for (const auto &term : term_list) {
        int tid = _term_to_id.at(term);
        int start = _posting_docs_ptr[tid];
        int end = _posting_docs_ptr[tid + 1];
        int df = end - start;
        write_u32(static_cast<uint32_t>(df));
        if (df > 0) {
            f.write(reinterpret_cast<const char *>(_posting_docs_data.data() + start), df * sizeof(int));
            f.write(reinterpret_cast<const char *>(_posting_tfs_data.data() + start), df * sizeof(uint16_t));
        }
    }

    if (include_forward_index) {
        for (size_t doc_id = 0; doc_id < _doc_lengths.size(); ++doc_id) {
            const auto &freqs = (doc_id < _doc_term_freqs.size()) ? _doc_term_freqs[doc_id] : luisa::unordered_map<luisa::string, int>{};
            write_u16(static_cast<uint16_t>(freqs.size()));
            for (const auto &[term, tf] : freqs) {
                int tid = _term_to_id.at(term);
                write_u32(static_cast<uint32_t>(tid));
                write_u16(static_cast<uint16_t>(tf));
            }
        }
    }
}

void InvertedIndex::load(const std::filesystem::path &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return;

    auto read_u8 = [&]() -> uint8_t {
        uint8_t v;
        f.read(reinterpret_cast<char *>(&v), 1);
        return v;
    };
    auto read_u16 = [&]() -> uint16_t {
        uint16_t v;
        f.read(reinterpret_cast<char *>(&v), sizeof(v));
        return v;
    };
    auto read_u32 = [&]() -> uint32_t {
        uint32_t v;
        f.read(reinterpret_cast<char *>(&v), sizeof(v));
        return v;
    };
    auto read_f64 = [&]() -> double {
        double v;
        f.read(reinterpret_cast<char *>(&v), sizeof(v));
        return v;
    };

    std::array<uint8_t, 4> magic{};
    f.read(reinterpret_cast<char *>(magic.data()), 4);
    if (magic != MAGIC) {
        LUISA_WARNING("Invalid file format");
        return;
    }
    uint8_t version = read_u8();
    if (version != 1 && version != VERSION) {
        LUISA_WARNING("Unsupported version");
        return;
    }

    _N = static_cast<int>(read_u32());
    uint32_t num_docs = read_u32();
    _avgdl = read_f64();
    uint32_t num_terms = read_u32();

    luisa::vector<luisa::string> terms;
    terms.reserve(num_terms);
    _term_to_id.clear();
    for (uint32_t i = 0; i < num_terms; ++i) {
        uint16_t term_len = read_u16();
        luisa::string term(term_len, '\0');
        f.read(term.data(), term_len);
        terms.push_back(term);
        _term_to_id.emplace(std::move(term), static_cast<int>(i));
    }

    if (num_docs > 0) {
        _doc_lengths.resize(num_docs);
        f.read(reinterpret_cast<char *>(_doc_lengths.data()), num_docs * sizeof(int));
    } else {
        _doc_lengths.clear();
    }
    _sum_doc_lengths = 0;
    for (int dl : _doc_lengths) _sum_doc_lengths += dl;
    _doc_lengths_arr = _doc_lengths;

    luisa::vector<luisa::vector<int>> docs_data_list(num_terms);
    luisa::vector<luisa::vector<uint16_t>> tfs_data_list(num_terms);
    _posting_docs_ptr.resize(num_terms + 1);
    _posting_tfs_ptr.resize(num_terms + 1);
    _posting_docs_ptr[0] = 0;
    _posting_tfs_ptr[0] = 0;

    for (uint32_t i = 0; i < num_terms; ++i) {
        uint32_t df = read_u32();
        if (df > 0) {
            docs_data_list[i].resize(df);
            tfs_data_list[i].resize(df);
            f.read(reinterpret_cast<char *>(docs_data_list[i].data()), df * sizeof(int));
            f.read(reinterpret_cast<char *>(tfs_data_list[i].data()), df * sizeof(uint16_t));
        }
        _posting_docs_ptr[i + 1] = _posting_docs_ptr[i] + static_cast<int>(df);
        _posting_tfs_ptr[i + 1] = _posting_tfs_ptr[i] + static_cast<int>(df);
    }

    size_t total = _posting_docs_ptr[num_terms];
    _posting_docs_data.resize(total);
    _posting_tfs_data.resize(total);
    for (uint32_t i = 0; i < num_terms; ++i) {
        int start = _posting_docs_ptr[i];
        int n = _posting_docs_ptr[i + 1] - start;
        for (int j = 0; j < n; ++j) {
            _posting_docs_data[start + j] = docs_data_list[i][j];
            _posting_tfs_data[start + j] = tfs_data_list[i][j];
        }
    }

    bool has_forward_chunk = false;
    luisa::vector<luisa::unordered_set<luisa::string>> doc_token_strs_chunk;
    luisa::vector<luisa::unordered_map<luisa::string, int>> doc_term_freqs_chunk;
    auto pos = f.tellg();
    f.seekg(0, std::ios::end);
    auto end_pos = f.tellg();
    f.seekg(pos);
    has_forward_chunk = end_pos > pos;

    if (has_forward_chunk && version >= 2) {
        doc_token_strs_chunk.resize(num_docs);
        doc_term_freqs_chunk.resize(num_docs);
        for (uint32_t d = 0; d < num_docs; ++d) {
            char buf[2];
            f.read(buf, 2);
            if (f.gcount() < 2) break;
            uint16_t num_terms_doc = *reinterpret_cast<uint16_t *>(buf);
            for (uint16_t j = 0; j < num_terms_doc; ++j) {
                uint32_t tid = read_u32();
                uint16_t tf = read_u16();
                if (tid < terms.size()) {
                    doc_token_strs_chunk[d].insert(terms[tid]);
                    doc_term_freqs_chunk[d][terms[tid]] = static_cast<int>(tf);
                }
            }
        }
    }

    // --- Parallel auxiliary map construction ---
    {
        struct AuxMaps {
            luisa::unordered_map<int, luisa::vector<luisa::string>> terms_by_length;
            luisa::unordered_map<luisa::string, luisa::vector<luisa::string>> terms_by_length_prefix;
        };
        luisa::vector<AuxMaps> locals;
        luisa::fiber::mutex aux_mtx;

        luisa::vector<luisa::string> all_terms;
        all_terms.reserve(_term_to_id.size());
        for (const auto &[term, tid] : _term_to_id) {
            (void)tid;
            all_terms.push_back(term);
        }

        luisa::fiber::parallel(static_cast<uint32_t>(all_terms.size()), [&](uint32_t begin, uint32_t end) noexcept {
            AuxMaps local;
            for (uint32_t i = begin; i < end; ++i) {
                const auto &term = all_terms[i];
                int len = static_cast<int>(term.size());
                local.terms_by_length[len].push_back(term);
                luisa::string prefix_key = vstd::to_string(len) + ":" + (term.empty() ? luisa::string{} : luisa::string{term.substr(0, 1)});
                local.terms_by_length_prefix[prefix_key].push_back(term);
            }
            {
                luisa::fiber::lock lck(aux_mtx);
                locals.push_back(std::move(local));
            }
        });

        _terms_by_length.clear();
        _terms_by_length_prefix.clear();
        for (auto &local : locals) {
            for (auto &[len, terms] : local.terms_by_length) {
                auto &dest = _terms_by_length[len];
                dest.insert(dest.end(), terms.begin(), terms.end());
            }
            for (auto &[key, terms] : local.terms_by_length_prefix) {
                auto &dest = _terms_by_length_prefix[key];
                dest.insert(dest.end(), terms.begin(), terms.end());
            }
        }
    }

    _term_collection_freqs.resize(num_terms, 0);
    int total_tokens = _sum_doc_lengths;
    _collection_lm_cache.clear();

    if (has_forward_chunk) {
        _doc_token_strs.clear();
        for (auto &s : doc_token_strs_chunk) {
            _doc_token_strs.emplace_back(std::move(s));
        }
        _doc_term_freqs = std::move(doc_term_freqs_chunk);

        luisa::vector<luisa::string> terms_by_id(_term_to_id.size());
        for (const auto &[term, tid] : _term_to_id) {
            terms_by_id[tid] = term;
        }
        luisa::fiber::parallel(static_cast<uint32_t>(num_terms), [&](uint32_t tid) noexcept {
            int start = _posting_tfs_ptr[tid];
            int end = _posting_tfs_ptr[tid + 1];
            int cf = 0;
            for (int i = start; i < end; ++i) cf += _posting_tfs_data[i];
            _term_collection_freqs[tid] = cf;
        });
        for (size_t tid = 0; tid < terms_by_id.size(); ++tid) {
            int cf = _term_collection_freqs[tid];
            if (total_tokens > 0) _collection_lm_cache[terms_by_id[tid]] = static_cast<double>(cf) / total_tokens;
        }
    } else {
        _doc_token_strs.resize(num_docs);
        _doc_term_freqs.resize(num_docs);

        luisa::vector<luisa::string> terms_by_id(_term_to_id.size());
        for (const auto &[term, tid] : _term_to_id) {
            terms_by_id[tid] = term;
        }

        luisa::fiber::parallel(static_cast<uint32_t>(num_terms), [&](uint32_t tid) noexcept {
            int start = _posting_docs_ptr[tid];
            int end = _posting_docs_ptr[tid + 1];
            int cf = 0;
            for (int i = start; i < end; ++i) cf += _posting_tfs_data[i];
            _term_collection_freqs[tid] = cf;
        });
        for (size_t tid = 0; tid < terms_by_id.size(); ++tid) {
            int cf = _term_collection_freqs[tid];
            if (total_tokens > 0) _collection_lm_cache[terms_by_id[tid]] = static_cast<double>(cf) / total_tokens;
            int start = _posting_docs_ptr[tid];
            int end = _posting_docs_ptr[tid + 1];
            for (int i = start; i < end; ++i) {
                int d = _posting_docs_data[i];
                int tf = _posting_tfs_data[i];
                if (d < static_cast<int>(num_docs)) {
                    if (!_doc_token_strs[d].has_value()) _doc_token_strs[d].emplace();
                    _doc_token_strs[d]->insert(terms_by_id[tid]);
                    _doc_term_freqs[d][terms_by_id[tid]] = tf;
                }
            }
        }
    }
    _finalized = true;
}

double InvertedIndex::collection_lm(luisa::string_view term) const {
    auto it = _collection_lm_cache.find(luisa::string{term});
    if (it != _collection_lm_cache.end()) return it->second;
    return 0.0;
}

int InvertedIndex::term_collection_freq(int tid) const {
    if (tid >= 0 && tid < static_cast<int>(_term_collection_freqs.size())) return _term_collection_freqs[tid];
    return 0;
}

const luisa::unordered_set<luisa::string> &InvertedIndex::doc_token_set(int doc_id) const {
    if (doc_id >= 0 && doc_id < static_cast<int>(_doc_token_strs.size()) && _doc_token_strs[doc_id].has_value()) {
        return *_doc_token_strs[doc_id];
    }
    static const luisa::unordered_set<luisa::string> empty;
    return empty;
}

}// namespace tokenize
