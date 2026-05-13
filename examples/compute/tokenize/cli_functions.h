#pragma once

#include "file_builder.h"
#include <luisa/core/stl.h>
#include <luisa/core/stl/vector.h>
#include <luisa/vstl/hash_map.h>
#include <luisa/vstl/md5.h>
#include <luisa/vstl/string_builder.h>
#include <charconv>
#include <cstdlib>

namespace {
inline bool parse_u64(const luisa::string &s, uint64_t &out) noexcept {
    auto r = std::from_chars(s.data(), s.data() + s.size(), out);
    return r.ec == std::errc{} && r.ptr == s.data() + s.size();
}
inline bool parse_i32(const luisa::string &s, int &out) noexcept {
    auto r = std::from_chars(s.data(), s.data() + s.size(), out);
    return r.ec == std::errc{} && r.ptr == s.data() + s.size();
}
inline bool parse_f64(const luisa::string &s, double &out) noexcept {
    char *end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return end == s.c_str() + s.size();
}
}// namespace

namespace cli {

using ArgumentList = luisa::vector<luisa::string>;

inline vstd::HashMap<luisa::string, tokenize::FileBuilder> &builders() {
    static vstd::HashMap<luisa::string, tokenize::FileBuilder> instance;
    return instance;
}

inline vstd::HashMap<uint64_t> &builder_handles() {
    static vstd::HashMap<uint64_t> instance;
    return instance;
}

inline luisa::filesystem::path &global_cache_path() {
    static luisa::filesystem::path instance;
    return instance;
}

inline luisa::filesystem::path &global_index_path() {
    static luisa::filesystem::path instance;
    return instance;
}

inline luisa::string make_builder_key(const luisa::vector<luisa::filesystem::path> &paths) {
    luisa::vector<luisa::string> canonicals;
    canonicals.reserve(paths.size());
    std::error_code ec;
    for (auto &p : paths) {
        auto cp = luisa::filesystem::canonical(p, ec);
        if (ec) cp = luisa::filesystem::weakly_canonical(p, ec);
        if (ec) cp = p;
        canonicals.push_back(luisa::to_string(cp));
    }
    luisa::sort(canonicals.begin(), canonicals.end());
    vstd::StringBuilder sb;
    for (size_t i = 0; i < canonicals.size(); ++i) {
        if (i > 0) sb.append('\0');
        sb.append(canonicals[i]);
    }
    return luisa::string{sb.view()};
}

inline luisa::filesystem::path make_output_path(const luisa::string &key) {
    vstd::MD5 md5{vstd::span<uint8_t const>(
        reinterpret_cast<uint8_t const *>(key.data()), key.size())};
    auto hash_str = md5.to_string(false);
    return luisa::filesystem::temp_directory_path() / ("fb_" + hash_str);
}

inline luisa::string cmd_add_builder(ArgumentList args) {
    if (args.empty()) return luisa::string{"error: missing paths"};

    int n = 2;
    double k1 = 1.2;
    double b = 0.75;
    size_t path_count = args.size();
    if (args.size() >= 4) {
        if (parse_i32(args[args.size() - 3], n) &&
            parse_f64(args[args.size() - 2], k1) &&
            parse_f64(args[args.size() - 1], b)) {
            path_count = args.size() - 3;
        }
    }

    if (path_count == 0) return luisa::string{"error: empty paths"};

    luisa::vector<luisa::filesystem::path> paths;
    paths.reserve(path_count);
    for (size_t i = 0; i < path_count; ++i) {
        paths.emplace_back(args[i]);
    }

    auto key = make_builder_key(paths);
    auto &g_builders = builders();
    auto idx_builder = g_builders.find(key);
    if (!idx_builder) {
        auto out_path = make_output_path(key);
        auto cache_path = global_cache_path();
        auto index_path = global_index_path();
        idx_builder = g_builders.emplace(
            key, std::move(paths), std::move(out_path),
            std::move(cache_path), std::move(index_path), n, k1, b);
    }

    auto *builder = &idx_builder.value();
    uint64_t handle = reinterpret_cast<uint64_t>(builder);
    auto &handles = builder_handles();
    handles.emplace(handle);
    return luisa::string{std::to_string(handle)};
}

inline luisa::string cmd_remove_builder(ArgumentList args) {
    if (args.empty()) return luisa::string{"error: missing handle"};
    uint64_t handle = 0;
    if (!parse_u64(args[0], handle)) {
        return luisa::string{"error: invalid handle"};
    }
    auto &handles = builder_handles();
    auto idx = handles.find(handle);
    if (!idx) {
        return luisa::string{"error: builder not found"};
    }
    auto *target = reinterpret_cast<tokenize::FileBuilder *>(handle);
    auto &g_builders = builders();
    for (auto &kv : g_builders) {
        if (&kv.second == target) {
            g_builders.remove(kv.first);
            handles.remove(handle);
            return luisa::string{"ok"};
        }
    }
    return luisa::string{"error: builder not found"};
}

inline luisa::string cmd_search(ArgumentList args) {
    if (args.size() < 2) return luisa::string{"error: missing handle or keywords"};
    uint64_t handle = 0;
    if (!parse_u64(args[0], handle)) {
        return luisa::string{"error: invalid handle"};
    }
    auto &handles = builder_handles();
    auto idx = handles.find(handle);
    if (!idx) {
        return luisa::string{"error: builder not found"};
    }
    auto *builder = reinterpret_cast<tokenize::FileBuilder *>(handle);

    luisa::string_view keywords = args[1];
    int top_k = 5;
    bool diversify = false;
    double diversity_lambda = 0.5;
    bool use_spelling = true;
    bool use_stemming = true;
    bool use_string_similarity = false;
    bool use_adaptive_scoring = false;

    if (args.size() > 2 && !parse_i32(args[2], top_k)) {
        return luisa::string{"error: invalid argument format"};
    }
    if (args.size() > 3) diversify = (args[3] == "true");
    if (args.size() > 4 && !parse_f64(args[4], diversity_lambda)) {
        return luisa::string{"error: invalid argument format"};
    }
    if (args.size() > 5) use_spelling = (args[5] == "true");
    if (args.size() > 6) use_stemming = (args[6] == "true");
    if (args.size() > 7) use_string_similarity = (args[7] == "true");
    if (args.size() > 8) use_adaptive_scoring = (args[8] == "true");

    auto results = builder->search(
        keywords, top_k, diversify, diversity_lambda,
        use_spelling, use_stemming, use_string_similarity, use_adaptive_scoring);

    vstd::StringBuilder sb;
    sb.append(luisa::format("result count: {}\n", results.size()));
    for (auto &r : results) {
        sb.append("[score=");
        sb << luisa::format("{:.2f}", r.score);
        sb.append(", path=");
        sb.append(r.path);
        sb.append(", line=");
        vstd::to_string(r.line_index, sb);
        sb.append("]\n");
    }
    return luisa::string{sb.view()};
}

}// namespace cli
