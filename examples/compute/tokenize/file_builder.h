#pragma once

#include "ngram_tokenizer.h"
#include "inverted_index.h"
#include "searcher.h"
#include "simhash.h"
#include <luisa/core/stl.h>
#include <luisa/core/fiber.h>

namespace tokenize {

struct FileBuilderResult {
    int doc_id = 0;
    double score = 0.0;
    luisa::string path;
    int line_index = 0;
};

class FileBuilder {
public:
    FileBuilder(luisa::vector<luisa::filesystem::path> paths,
                luisa::filesystem::path output_path,
                int n = 2,
                double k1 = 1.2,
                double b = 0.75);

    [[nodiscard]] luisa::vector<FileBuilderResult> search(
        luisa::string_view keywords,
        int top_k = 5,
        bool diversify = false,
        double diversity_lambda = 0.5,
        bool use_spelling = true,
        bool use_stemming = true,
        bool use_string_similarity = false,
        bool use_adaptive_scoring = false);

    void update();

    [[nodiscard]] const InvertedIndex &index() const noexcept { return *_index; }
    [[nodiscard]] bool empty() const noexcept { return !_index || _index->N() == 0; }

private:
    struct LineEntry {
        luisa::string rel;
        int line_idx = 0;
        luisa::string content;
        luisa::vector<luisa::string> tokens;
        uint64_t simhash = 0;
        luisa::string fingerprint;
    };

    struct DocInfo {
        luisa::string path;
        int line_index = 0;
        luisa::string content;
    };

    void _scan_and_build();
    void _build_index(const luisa::vector<LineEntry> &lines);
    void _save_cache();
    bool _load_cache();
    bool _cache_valid() const;

    [[nodiscard]] luisa::string _hash_file(const luisa::filesystem::path &path) const;
    [[nodiscard]] bool _is_text_file(const luisa::filesystem::path &path) const;
    [[nodiscard]] luisa::string _compute_fingerprint(luisa::string_view text) const;

    luisa::vector<luisa::filesystem::path> _paths;
    luisa::filesystem::path _output_path;
    luisa::filesystem::path _cache_path;
    luisa::filesystem::path _index_path;
    int _n;
    double _k1;
    double _b;

    luisa::unordered_map<luisa::string, luisa::string> _file_hashes;
    luisa::vector<DocInfo> _doc_info;
    luisa::unique_ptr<InvertedIndex> _index;
    luisa::unique_ptr<Searcher> _searcher;
    NgramTokenizer _tokenizer;
};

}// namespace tokenize
