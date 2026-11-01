// Host-side retrieval index implementation (retrieval plan phase 1): canonical
// path ownership, deterministic discovery/ranking, and transactional incremental
// updates with per-file failure isolation. Phase H1 adds hybrid dense + sparse
// lexical scoring (fused 0.7 / 0.3 by default) on top of the unchanged dense
// path. RAG growth (report.md §2-§6) adds chunked document ingestion with byte
// ranges, memory-only documents, corpus snapshots, per-query ranking controls
// (weights / metadata filter / MMR / ColBERT), progress + cancellation and
// statistics counters.

#include "retrieval_index.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <system_error>

#include <yyjson.h>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace luisa::embed {
namespace {

[[nodiscard]] RetrievalStatus make_error(RetrievalErrorCode code, luisa::string message,
                                         RetrievalSubCode sub = RetrievalSubCode::none) {
    RetrievalStatus status;
    status.code = code;
    status.message = std::move(message);
    status.sub_code = sub;
    return status;
}

[[nodiscard]] luisa::string ascii_lower(luisa::string_view input) {
    luisa::string result;
    result.reserve(input.size());
    for (const char c : input) {
        const auto u = static_cast<unsigned char>(c);
        result.push_back(static_cast<char>(u >= 'A' && u <= 'Z' ? u + ('a' - 'A') : u));
    }
    return result;
}

[[nodiscard]] bool valid_utf8(luisa::string_view text) noexcept {
    size_t i = 0u;
    while (i < text.size()) {
        const auto first = static_cast<uint8_t>(text[i]);
        if (first <= 0x7fu) {
            i++;
            continue;
        }
        size_t continuation_count = 0u;
        uint32_t code_point = 0u;
        uint32_t minimum = 0u;
        if (first >= 0xc2u && first <= 0xdfu) {
            continuation_count = 1u;
            code_point = first & 0x1fu;
            minimum = 0x80u;
        } else if (first >= 0xe0u && first <= 0xefu) {
            continuation_count = 2u;
            code_point = first & 0x0fu;
            minimum = 0x800u;
        } else if (first >= 0xf0u && first <= 0xf4u) {
            continuation_count = 3u;
            code_point = first & 0x07u;
            minimum = 0x10000u;
        } else {
            return false;
        }
        if (i + continuation_count >= text.size()) {
            return false;
        }
        for (size_t j = 1u; j <= continuation_count; j++) {
            const auto next = static_cast<uint8_t>(text[i + j]);
            if ((next & 0xc0u) != 0x80u) {
                return false;
            }
            code_point = (code_point << 6u) | (next & 0x3fu);
        }
        if (code_point < minimum || code_point > 0x10ffffu ||
            (code_point >= 0xd800u && code_point <= 0xdfffu)) {
            return false;
        }
        i += continuation_count + 1u;
    }
    return true;
}

[[nodiscard]] bool blank_text(luisa::string_view text) noexcept {
    return std::ranges::all_of(text, [](const char c) {
        return std::isspace(static_cast<unsigned char>(c)) != 0;
    });
}

[[nodiscard]] bool file_not_found(const std::error_code &ec) noexcept {
    if (!ec) {
        return false;
    }
    if (ec == std::errc::no_such_file_or_directory) {
        return true;
    }
#ifdef _WIN32
    return ec.value() == ERROR_FILE_NOT_FOUND || ec.value() == ERROR_PATH_NOT_FOUND;
#else
    return false;
#endif
}

[[nodiscard]] bool cancelled(const RetrievalProgress *progress) noexcept {
    return progress != nullptr && progress->cancel_flag != nullptr &&
           *progress->cancel_flag != 0u;
}

void report_progress(const RetrievalProgress *progress, uint64_t done, uint64_t total) noexcept {
    if (progress != nullptr && progress->on_progress != nullptr) {
        progress->on_progress(progress->user_data, done, total);
    }
}

// --- metadata JSON validation -------------------------------------------------

[[nodiscard]] RetrievalStatus validate_json_object(luisa::string_view json,
                                                   luisa::string_view what) {
    if (json.empty()) {
        return {};
    }
    if (!valid_utf8(json)) {
        return make_error(RetrievalErrorCode::invalid_argument,
                          luisa::format("{} must be valid UTF-8", what));
    }
    yyjson_read_err error{};
    auto *doc = yyjson_read_opts(
        const_cast<char *>(json.data()),
        json.size(), 0, nullptr, &error);
    if (doc == nullptr) {
        return make_error(RetrievalErrorCode::invalid_argument,
                          luisa::format("{} is not valid JSON: {}", what, error.msg));
    }
    const auto *root = yyjson_doc_get_root(doc);
    const auto ok = yyjson_is_obj(root);
    yyjson_doc_free(doc);
    if (!ok) {
        return make_error(RetrievalErrorCode::invalid_argument,
                          luisa::format("{} must be a JSON object", what));
    }
    return {};
}

// --- metadata filter (report.md §4) -------------------------------------------

[[nodiscard]] bool json_scalar_equal(const yyjson_val *lhs,
                                     const RetrievalIndex::Filter &leaf) noexcept {
    if (leaf.is_number) {
        if (!yyjson_is_num(lhs)) { return false; }
        return yyjson_get_num(lhs) == leaf.number;
    }
    if (yyjson_is_str(lhs)) {
        const luisa::string_view str{yyjson_get_str(lhs), yyjson_get_len(lhs)};
        return str == luisa::string_view{leaf.text};
    }
    if (yyjson_is_bool(lhs)) {
        return (yyjson_get_bool(lhs) ? "true" : "false") == luisa::string_view{leaf.text};
    }
    return false;
}

[[nodiscard]] bool json_scalar_compare(const yyjson_val *lhs,
                                       const RetrievalIndex::Filter &leaf) noexcept {
    if (!yyjson_is_num(lhs)) { return false; }
    const auto value = yyjson_get_num(lhs);
    switch (leaf.op) {
        case RetrievalIndex::Filter::Op::gte: return value >= leaf.number;
        case RetrievalIndex::Filter::Op::lte: return value <= leaf.number;
        case RetrievalIndex::Filter::Op::gt: return value > leaf.number;
        case RetrievalIndex::Filter::Op::lt: return value < leaf.number;
        default: return false;
    }
}

// Parses a {"key": {"sub": value}} comparison object into ANDed leaves.
[[nodiscard]] RetrievalStatus parse_comparison(const char *key, size_t key_len,
                                               const yyjson_val *object,
                                               RetrievalIndex::Filter &and_node) {
    size_t idx = 0u;
    size_t max = 0u;
    const yyjson_val *sub_key = nullptr;
    const yyjson_val *value = nullptr;
    yyjson_obj_foreach(object, idx, max, sub_key, value) {
        const luisa::string_view op{yyjson_get_str(sub_key), yyjson_get_len(sub_key)};
        auto leaf = luisa::make_unique<RetrievalIndex::Filter>();
        leaf->key = luisa::string{key, key_len};
        if (op == "$eq") {
            leaf->op = RetrievalIndex::Filter::Op::eq;
        } else if (op == "$gte") {
            leaf->op = RetrievalIndex::Filter::Op::gte;
        } else if (op == "$lte") {
            leaf->op = RetrievalIndex::Filter::Op::lte;
        } else if (op == "$gt") {
            leaf->op = RetrievalIndex::Filter::Op::gt;
        } else if (op == "$lt") {
            leaf->op = RetrievalIndex::Filter::Op::lt;
        } else if (op == "$in") {
            leaf->op = RetrievalIndex::Filter::Op::in_list;
            if (!yyjson_is_arr(value)) {
                return make_error(RetrievalErrorCode::invalid_argument,
                                  "$in expects an array value");
            }
            size_t i = 0u;
            size_t imax = 0u;
            const yyjson_val *item = nullptr;
            yyjson_arr_foreach(value, i, imax, item) {
                if (yyjson_is_num(item)) {
                    leaf->numbers.push_back(yyjson_get_num(item));
                } else if (yyjson_is_str(item)) {
                    leaf->texts.emplace_back(yyjson_get_str(item), yyjson_get_len(item));
                } else if (yyjson_is_bool(item)) {
                    leaf->texts.emplace_back(yyjson_get_bool(item) ? "true" : "false");
                } else {
                    return make_error(RetrievalErrorCode::invalid_argument,
                                      "$in accepts only string/number/bool array items");
                }
            }
            and_node.children.push_back(std::move(leaf));
            continue;
        } else {
            return make_error(RetrievalErrorCode::invalid_argument,
                              luisa::format("unsupported filter operator '{}'", op));
        }
        if (leaf->op == RetrievalIndex::Filter::Op::eq) {
            if (yyjson_is_num(value)) {
                leaf->is_number = true;
                leaf->number = yyjson_get_num(value);
            } else if (yyjson_is_str(value)) {
                leaf->text = luisa::string{yyjson_get_str(value), yyjson_get_len(value)};
            } else if (yyjson_is_bool(value)) {
                leaf->text = yyjson_get_bool(value) ? "true" : "false";
            } else {
                return make_error(RetrievalErrorCode::invalid_argument,
                                  "filter comparison expects a scalar value");
            }
        } else {
            if (!yyjson_is_num(value)) {
                return make_error(RetrievalErrorCode::invalid_argument,
                                  "$gte/$lte/$gt/$lt expect a numeric value");
            }
            leaf->is_number = true;
            leaf->number = yyjson_get_num(value);
        }
        and_node.children.push_back(std::move(leaf));
    }
    return {};
}

[[nodiscard]] RetrievalStatus parse_filter_node(const yyjson_val *node,
                                                luisa::unique_ptr<RetrievalIndex::Filter> &out) {
    if (!yyjson_is_obj(node)) {
        return make_error(RetrievalErrorCode::invalid_argument,
                          "filter must be a JSON object");
    }
    auto root = luisa::make_unique<RetrievalIndex::Filter>();
    root->op = RetrievalIndex::Filter::Op::logical_and;
    size_t idx = 0u;
    size_t max = 0u;
    const yyjson_val *key = nullptr;
    const yyjson_val *value = nullptr;
    yyjson_obj_foreach(node, idx, max, key, value) {
        const luisa::string_view name{yyjson_get_str(key), yyjson_get_len(key)};
        if (name == "$and" || name == "$or") {
            if (!yyjson_is_arr(value)) {
                return make_error(RetrievalErrorCode::invalid_argument,
                                  luisa::format("{} expects an array", name));
            }
            auto logical = luisa::make_unique<RetrievalIndex::Filter>();
            logical->op = name == "$and" ? RetrievalIndex::Filter::Op::logical_and
                                         : RetrievalIndex::Filter::Op::logical_or;
            size_t i = 0u;
            size_t imax = 0u;
            const yyjson_val *child = nullptr;
            yyjson_arr_foreach(value, i, imax, child) {
                luisa::unique_ptr<RetrievalIndex::Filter> parsed;
                auto status = parse_filter_node(child, parsed);
                if (!status.ok()) { return status; }
                logical->children.push_back(std::move(parsed));
            }
            if (logical->children.empty()) {
                return make_error(RetrievalErrorCode::invalid_argument,
                                  luisa::format("{} must not be empty", name));
            }
            root->children.push_back(std::move(logical));
            continue;
        }
        if (!name.empty() && name.front() == '$') {
            return make_error(RetrievalErrorCode::invalid_argument,
                              luisa::format("unsupported filter key '{}'", name));
        }
        if (yyjson_is_obj(value)) {
            auto status = parse_comparison(yyjson_get_str(key), yyjson_get_len(key), value, *root);
            if (!status.ok()) { return status; }
            continue;
        }
        auto leaf = luisa::make_unique<RetrievalIndex::Filter>();
        leaf->op = RetrievalIndex::Filter::Op::eq;
        leaf->key = luisa::string{yyjson_get_str(key), yyjson_get_len(key)};
        if (yyjson_is_num(value)) {
            leaf->is_number = true;
            leaf->number = yyjson_get_num(value);
        } else if (yyjson_is_str(value)) {
            leaf->text = luisa::string{yyjson_get_str(value), yyjson_get_len(value)};
        } else if (yyjson_is_bool(value)) {
            leaf->text = yyjson_get_bool(value) ? "true" : "false";
        } else {
            return make_error(RetrievalErrorCode::invalid_argument,
                              "filter equality expects a scalar value");
        }
        root->children.push_back(std::move(leaf));
    }
    out = std::move(root);
    return {};
}

}// namespace

// --- static helpers ------------------------------------------------------------

RetrievalIndex::RetrievalIndex(RetrievalEncodeCallback encode) noexcept
    : _encode_callback{std::move(encode)} {}

RetrievalIndex::RetrievalIndex(RetrievalHybridEncodeCallback encode) noexcept
    : _hybrid_encode_callback{std::move(encode)}, _hybrid{true} {}

luisa::vector<luisa::string> RetrievalIndex::default_extensions() {
    return {".txt", ".md"};
}

RetrievalStatus RetrievalIndex::path_from_utf8(
    luisa::string_view utf8,
    luisa::filesystem::path &path) {
    if (utf8.empty()) {
        return make_error(RetrievalErrorCode::invalid_argument, "path must not be empty");
    }
    if (utf8.find('\0') != luisa::string_view::npos || !valid_utf8(utf8)) {
        return make_error(RetrievalErrorCode::invalid_argument,
                          "path must be valid UTF-8 without embedded NUL bytes");
    }
#ifdef _WIN32
    if (utf8.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return make_error(RetrievalErrorCode::invalid_argument, "path is too long");
    }
    const auto input_length = static_cast<int>(utf8.size());
    const auto output_length = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), input_length, nullptr, 0);
    if (output_length <= 0) {
        return make_error(RetrievalErrorCode::invalid_argument,
                          "cannot convert UTF-8 path to a Windows path");
    }
    luisa::wstring wide;
    wide.resize(static_cast<size_t>(output_length));
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                            utf8.data(), input_length, wide.data(), output_length) == 0) {
        return make_error(RetrievalErrorCode::invalid_argument,
                          "cannot convert UTF-8 path to a Windows path");
    }
    path = luisa::filesystem::path{wide};
#else
    path = luisa::filesystem::path{utf8};
#endif
    return {};
}

RetrievalStatus RetrievalIndex::validate_metadata_json(
    luisa::string_view json,
    luisa::string_view what) {
    return validate_json_object(json, what);
}

RetrievalStatus RetrievalIndex::normalize_extensions(
    luisa::span<const luisa::string> input,
    luisa::vector<luisa::string> &normalized) {
    if (input.empty()) {
        return make_error(RetrievalErrorCode::invalid_argument,
                          "extension list must not be empty");
    }
    luisa::vector<luisa::string> result;
    result.reserve(input.size());
    for (const auto &raw : input) {
        if (raw.empty()) {
            return make_error(RetrievalErrorCode::invalid_argument,
                              "extension list contains an empty entry");
        }
        auto extension = ascii_lower(raw);
        if (extension.front() != '.') {
            extension.insert(extension.begin(), '.');
        }
        if (extension.size() <= 1u) {
            return make_error(RetrievalErrorCode::invalid_argument,
                              luisa::format("malformed extension rule '{}'", raw));
        }
        for (size_t i = 0u; i < extension.size(); i++) {
            const auto c = static_cast<unsigned char>(extension[i]);
            const bool invalid = c == '/' || c == '\\' || c == ':' || c == '*' ||
                                 c == '?' || c == ',' || (c == '.' && i > 0u) ||
                                 std::isspace(c) != 0 || std::iscntrl(c) != 0;
            if (invalid) {
                return make_error(RetrievalErrorCode::invalid_argument,
                                  luisa::format("malformed extension rule '{}'", raw));
            }
        }
        result.push_back(std::move(extension));
    }
    luisa::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    normalized = std::move(result);
    return {};
}

RetrievalStatus RetrievalIndex::parse_extension_list(
    luisa::string_view comma_separated,
    luisa::vector<luisa::string> &normalized) {
    if (comma_separated.empty()) {
        return make_error(RetrievalErrorCode::invalid_argument,
                          "extension list must not be empty");
    }
    luisa::vector<luisa::string> entries;
    size_t begin = 0u;
    while (begin <= comma_separated.size()) {
        const auto comma = comma_separated.find(',', begin);
        const auto end = comma == luisa::string_view::npos ? comma_separated.size() : comma;
        entries.emplace_back(comma_separated.substr(begin, end - begin));
        if (comma == luisa::string_view::npos) {
            break;
        }
        begin = comma + 1u;
    }
    return normalize_extensions(
        luisa::span<const luisa::string>{entries.data(), entries.size()}, normalized);
}

RetrievalStatus RetrievalIndex::parse_filter(
    luisa::string_view json,
    luisa::unique_ptr<Filter> &out) {
    if (!valid_utf8(json)) {
        return make_error(RetrievalErrorCode::invalid_argument,
                          "filter must be valid UTF-8");
    }
    yyjson_read_err error{};
    auto *doc = yyjson_read_opts(
        const_cast<char *>(json.data()),
        json.size(), 0, nullptr, &error);
    if (doc == nullptr) {
        return make_error(RetrievalErrorCode::invalid_argument,
                          luisa::format("filter is not valid JSON: {}", error.msg));
    }
    auto status = parse_filter_node(yyjson_doc_get_root(doc), out);
    yyjson_doc_free(doc);
    return status;
}

bool RetrievalIndex::Filter::matches(luisa::string_view metadata_json) const {
    switch (op) {
        case Op::logical_and: {
            for (const auto &child : children) {
                if (!child->matches(metadata_json)) { return false; }
            }
            return true;
        }
        case Op::logical_or: {
            for (const auto &child : children) {
                if (child->matches(metadata_json)) { return true; }
            }
            return false;
        }
        case Op::eq:
        case Op::gte:
        case Op::lte:
        case Op::gt:
        case Op::lt:
        case Op::in_list:
            break;
    }
    if (metadata_json.empty()) { return false; }
    yyjson_read_err error{};
    auto *doc = yyjson_read_opts(
        const_cast<char *>(metadata_json.data()),
        metadata_json.size(), 0, nullptr, &error);
    if (doc == nullptr) { return false; }
    const auto *root = yyjson_doc_get_root(doc);
    const auto *value = yyjson_is_obj(root) ? yyjson_obj_getn(root, key.data(), key.size()) : nullptr;
    bool matched = false;
    if (value != nullptr) {
        switch (op) {
            case Op::eq: matched = json_scalar_equal(value, *this); break;
            case Op::gte:
            case Op::lte:
            case Op::gt:
            case Op::lt: matched = json_scalar_compare(value, *this); break;
            case Op::in_list: {
                if (is_number) {
                    matched = std::ranges::find(numbers, yyjson_get_num(value)) != numbers.end();
                } else {
                    matched = std::ranges::find(
                        texts, luisa::string{yyjson_get_str(value), yyjson_get_len(value)}) != texts.end();
                }
                break;
            }
            default: break;
        }
    }
    yyjson_doc_free(doc);
    return matched;
}

luisa::string RetrievalIndex::_path_key(const luisa::filesystem::path &path) {
    const auto utf8 = path.generic_u8string();
    return {reinterpret_cast<const char *>(utf8.data()), utf8.size()};
}

luisa::string RetrievalIndex::_display_path(const luisa::filesystem::path &path) {
    return _path_key(path);
}

RetrievalStatus RetrievalIndex::_normalize_path(
    const luisa::filesystem::path &input,
    bool require_existing,
    luisa::filesystem::path &display_path,
    luisa::filesystem::path &comparison_path,
    luisa::string &comparison_key) {
    if (input.empty()) {
        return make_error(RetrievalErrorCode::invalid_argument, "path must not be empty");
    }
    std::error_code ec;
    luisa::filesystem::path normalized;
    if (require_existing) {
        normalized = luisa::filesystem::canonical(input, ec);
    } else {
        auto absolute = luisa::filesystem::absolute(input, ec);
        if (!ec) {
            normalized = luisa::filesystem::weakly_canonical(absolute, ec);
        }
    }
    if (ec || normalized.empty()) {
        return make_error(RetrievalErrorCode::invalid_argument,
                          luisa::format("cannot normalize path '{}': {}",
                                        _display_path(input), ec.message()));
    }
    display_path = normalized.lexically_normal();
#ifdef _WIN32
    auto wide = display_path.wstring();
    if (wide.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return make_error(RetrievalErrorCode::invalid_argument, "path is too long");
    }
    if (!wide.empty()) {
        const auto length = static_cast<int>(wide.size());
        if (LCMapStringW(LOCALE_INVARIANT, LCMAP_LOWERCASE,
                         wide.data(), length, wide.data(), length) == 0) {
            return make_error(RetrievalErrorCode::invalid_argument,
                              "cannot apply Windows case folding to path");
        }
    }
    comparison_path = luisa::filesystem::path{wide};
#else
    comparison_path = display_path;
#endif
    comparison_key = _path_key(comparison_path);
    return {};
}

bool RetrievalIndex::_is_within(
    const luisa::filesystem::path &path,
    const luisa::filesystem::path &root) noexcept {
    const auto relative = path.lexically_relative(root);
    if (relative.empty() || relative.is_absolute()) {
        return false;
    }
    return std::ranges::all_of(relative, [](const auto &component) {
        return component != "..";
    });
}

bool RetrievalIndex::_extension_admitted(
    const luisa::filesystem::path &path,
    luisa::span<const luisa::string> extensions) {
    const auto raw_extension = path.extension().u8string();
    const luisa::string_view extension_view{
        reinterpret_cast<const char *>(raw_extension.data()), raw_extension.size()};
    const auto extension = ascii_lower(extension_view);
    return luisa::binary_search(extensions.begin(), extensions.end(), extension);
}

RetrievalIndex::DiscoveryResult RetrievalIndex::_discover(
    const luisa::filesystem::path &root,
    luisa::span<const luisa::string> extensions) {
    DiscoveryResult result;
    std::error_code ec;
    luisa::filesystem::recursive_directory_iterator iterator{
        root, luisa::filesystem::directory_options::skip_permission_denied, ec};
    const luisa::filesystem::recursive_directory_iterator end;
    if (ec) {
        result.failed = 1u;
        return result;
    }
    while (iterator != end) {
        const auto entry_path = iterator->path();
        const auto status = iterator->symlink_status(ec);
        if (ec) {
            result.failed++;
            ec.clear();
        } else if (luisa::filesystem::is_regular_file(status)) {
            if (_extension_admitted(entry_path, extensions)) {
                auto canonical = luisa::filesystem::canonical(entry_path, ec);
                if (ec) {
                    result.failed++;
                    ec.clear();
                } else {
                    result.files.push_back(std::move(canonical));
                }
            } else {
                result.skipped++;
            }
        }
        iterator.increment(ec);
        if (ec) {
            result.failed++;
            break;
        }
    }
    luisa::sort(result.files.begin(), result.files.end(), [](const auto &a, const auto &b) {
        return _path_key(a) < _path_key(b);
    });
    result.files.erase(
        std::unique(result.files.begin(), result.files.end(), [](const auto &a, const auto &b) {
            return _path_key(a) == _path_key(b);
        }),
        result.files.end());
    return result;
}

RetrievalStatus RetrievalIndex::_read_text_file(
    const luisa::filesystem::path &path,
    luisa::string &text) {
    std::ifstream stream{path, std::ios::binary};
    if (!stream.is_open()) {
        return make_error(RetrievalErrorCode::operational,
                          luisa::format("cannot open '{}'", _display_path(path)),
                          RetrievalSubCode::io);
    }
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    if (end < 0 || static_cast<uint64_t>(end) > std::numeric_limits<size_t>::max() ||
        static_cast<uint64_t>(end) > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        return make_error(RetrievalErrorCode::operational,
                          luisa::format("cannot determine size of '{}'", _display_path(path)),
                          RetrievalSubCode::io);
    }
    luisa::string contents;
    contents.resize(static_cast<size_t>(end));
    stream.seekg(0, std::ios::beg);
    if (!contents.empty()) {
        stream.read(contents.data(), static_cast<std::streamsize>(contents.size()));
        if (!stream) {
            return make_error(RetrievalErrorCode::operational,
                              luisa::format("cannot read '{}'", _display_path(path)),
                              RetrievalSubCode::io);
        }
    }
    if (!valid_utf8(contents)) {
        return make_error(RetrievalErrorCode::operational,
                          luisa::format("'{}' is not valid UTF-8", _display_path(path)),
                          RetrievalSubCode::io);
    }
    text = std::move(contents);
    return {};
}

RetrievalIndex::RootRecord *RetrievalIndex::_find_owning_root(
    const luisa::filesystem::path &comparison_path) {
    for (auto &entry : _roots) {
        if (_is_within(comparison_path, entry.second.comparison_path)) {
            return &entry.second;
        }
    }
    return nullptr;
}

const RetrievalIndex::RootRecord *RetrievalIndex::_find_owning_root(
    const luisa::filesystem::path &comparison_path) const {
    for (const auto &entry : _roots) {
        if (_is_within(comparison_path, entry.second.comparison_path)) {
            return &entry.second;
        }
    }
    return nullptr;
}

RetrievalStatus RetrievalIndex::canonicalize_sparse(
    luisa::span<const std::pair<uint32_t, float>> input,
    luisa::vector<std::pair<uint32_t, float>> &output) {
    luisa::vector<std::pair<uint32_t, float>> entries;
    entries.reserve(input.size());
    for (const auto &entry : input) {
        const auto weight = entry.second;
        if (!std::isfinite(weight)) {
            return make_error(RetrievalErrorCode::operational,
                              luisa::format("sparse embedding contains a non-finite weight for token {}",
                                            entry.first),
                              RetrievalSubCode::encode);
        }
        if (weight < 0.0f) {
            return make_error(RetrievalErrorCode::operational,
                              luisa::format("sparse embedding contains a negative weight {} for token {}",
                                            weight, entry.first),
                              RetrievalSubCode::encode);
        }
        if (weight == 0.0f) {
            continue;// exact zeros carry no lexical evidence
        }
        entries.emplace_back(entry.first, weight);
    }
    // Sort by token id (weight as a tie-break so duplicate merging below is
    // order-canonical, hence deterministic).
    luisa::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    });
    luisa::vector<std::pair<uint32_t, float>> merged;
    merged.reserve(entries.size());
    double squared_norm = 0.0;
    size_t index = 0u;
    while (index < entries.size()) {
        const auto token = entries[index].first;
        double sum = 0.0;
        size_t next = index;
        while (next < entries.size() && entries[next].first == token) {
            sum += static_cast<double>(entries[next].second);
            next++;
        }
        if (!std::isfinite(sum)) {
            return make_error(RetrievalErrorCode::operational,
                              luisa::format("sparse embedding weight sum for token {} is not finite",
                                            token),
                              RetrievalSubCode::encode);
        }
        const auto combined = static_cast<float>(sum);
        if (!std::isfinite(combined)) {
            return make_error(RetrievalErrorCode::operational,
                              luisa::format("sparse embedding weight sum for token {} is not representable",
                                            token),
                              RetrievalSubCode::encode);
        }
        if (combined > 0.0f) {// also drops sums that underflowed to zero
            merged.emplace_back(token, combined);
            squared_norm += static_cast<double>(combined) * static_cast<double>(combined);
        }
        index = next;
    }
    if (merged.empty()) {
        // Empty / all-zero input: a valid sparse vector with norm zero.
        output.clear();
        return {};
    }
    if (!std::isfinite(squared_norm) || !(squared_norm > 0.0)) {
        return make_error(RetrievalErrorCode::operational,
                          "sparse embedding norm is not finite or not positive",
                          RetrievalSubCode::encode);
    }
    const auto scale = 1.0 / std::sqrt(squared_norm);
    for (auto &entry : merged) {
        entry.second = static_cast<float>(static_cast<double>(entry.second) * scale);
    }
    output = std::move(merged);
    return {};
}

double RetrievalIndex::_sparse_dot(
    luisa::span<const std::pair<uint32_t, float>> lhs,
    luisa::span<const std::pair<uint32_t, float>> rhs) noexcept {
    double dot = 0.0;
    size_t i = 0u;
    size_t j = 0u;
    while (i < lhs.size() && j < rhs.size()) {
        if (lhs[i].first < rhs[j].first) {
            i++;
        } else if (rhs[j].first < lhs[i].first) {
            j++;
        } else {
            dot += static_cast<double>(lhs[i].second) *
                   static_cast<double>(rhs[j].second);
            i++;
            j++;
        }
    }
    return dot;
}

double RetrievalIndex::_colbert_maxsim(
    luisa::span<const float> query,
    luisa::span<const float> document,
    uint32_t dimension) noexcept {
    if (dimension == 0u) { return 0.0; }
    const auto query_rows = query.size() / dimension;
    const auto document_rows = document.size() / dimension;
    if (query_rows == 0u || document_rows == 0u) { return 0.0; }
    double sum = 0.0;
    for (size_t r = 0u; r < query_rows; r++) {
        double best = -1.0;
        const auto *qrow = query.data() + r * dimension;
        for (size_t d = 0u; d < document_rows; d++) {
            const auto *drow = document.data() + d * dimension;
            double dot = 0.0;
            for (size_t k = 0u; k < dimension; k++) {
                dot += static_cast<double>(qrow[k]) * static_cast<double>(drow[k]);
            }
            if (dot > best) { best = dot; }
        }
        sum += best;
    }
    return sum / static_cast<double>(query_rows);
}

RetrievalStatus RetrievalIndex::_validate_dense(
    luisa::span<const float> dense) const {
    if (dense.empty()) {
        return make_error(RetrievalErrorCode::operational,
                          "encoder returned an empty dense embedding",
                          RetrievalSubCode::encode);
    }
    if (_dimension != 0u && dense.size() != _dimension) {
        return make_error(RetrievalErrorCode::operational,
                          luisa::format("embedding dimension {} does not match index dimension {}",
                                        dense.size(), _dimension),
                          RetrievalSubCode::encode);
    }
    for (const auto value : dense) {
        if (!std::isfinite(value)) {
            return make_error(RetrievalErrorCode::operational,
                              "encoder returned a non-finite dense embedding",
                              RetrievalSubCode::encode);
        }
    }
    return {};
}

RetrievalStatus RetrievalIndex::_encode(
    luisa::string_view text,
    uint32_t max_seq_len,
    bool want_colbert,
    RetrievalEmbedding &embedding) {
    RetrievalEncodeRequest request;
    request.max_seq_len = max_seq_len;
    request.want_colbert = want_colbert;
    if (_hybrid) {
        if (!_hybrid_encode_callback) {
            return make_error(RetrievalErrorCode::operational,
                              "no embedding encoder is configured",
                              RetrievalSubCode::encode);
        }
        auto candidate = _hybrid_encode_callback(text, request);
        auto status = _validate_dense(
            luisa::span<const float>{candidate.dense.data(), candidate.dense.size()});
        if (!status.ok()) {
            return status;
        }
        luisa::vector<std::pair<uint32_t, float>> sparse;
        status = canonicalize_sparse(
            luisa::span<const std::pair<uint32_t, float>>{candidate.sparse.data(),
                                                          candidate.sparse.size()},
            sparse);
        if (!status.ok()) {
            // Negative / non-finite sparse weights are recoverable encoder errors:
            // the caller keeps the previous document (see _ingest_document).
            return status;
        }
        embedding.dense = std::move(candidate.dense);
        embedding.sparse = std::move(sparse);
        embedding.colbert = _want_colbert ? std::move(candidate.colbert) : luisa::vector<float>{};
        embedding.truncated = candidate.truncated;
        return {};
    }
    if (!_encode_callback) {
        return make_error(RetrievalErrorCode::operational,
                          "no embedding encoder is configured",
                          RetrievalSubCode::encode);
    }
    auto candidate = _encode_callback(text, request);
    auto status = _validate_dense(
        luisa::span<const float>{candidate.data(), candidate.size()});
    if (!status.ok()) {
        return status;
    }
    embedding.dense = std::move(candidate);
    embedding.sparse.clear();
    embedding.colbert.clear();
    embedding.truncated = false;
    return {};
}

void RetrievalIndex::_record_encode(double elapsed_ms, bool cache_hit) noexcept {
    if (cache_hit) {
        _cache_hits++;
        return;
    }
    _encode_count++;
    _encode_time_total_ms += elapsed_ms;
    _cache_misses++;
}

luisa::string RetrievalIndex::_chunk_key(
    const luisa::string &base_key,
    uint32_t chunk_index) {
    return luisa::format("{}\x1f{}", base_key, chunk_index);
}

luisa::string RetrievalIndex::_chunk_variant(
    uint32_t chunk_index,
    const ChunkOptions &options) {
    return luisa::format("{}:{}:{}:{}", chunk_index,
                         options.chunk_size, options.chunk_overlap,
                         static_cast<uint32_t>(options.by));
}

void RetrievalIndex::_drop_document(luisa::string_view doc_id) noexcept {
    const auto it = _doc_chunks.find(doc_id);
    if (it == _doc_chunks.end()) { return; }
    for (const auto &key : it->second) {
        _documents.erase(key);
    }
    _doc_chunks.erase(it);
    if (_documents.empty()) {
        _dimension = 0u;
    }
}

RetrievalStatus RetrievalIndex::_ingest_document(
    luisa::string doc_id,
    const luisa::filesystem::path &path,
    const luisa::filesystem::path &comparison_path,
    const luisa::string &root_key,
    luisa::string_view text,
    luisa::string_view metadata_json,
    const ChunkOptions &options,
    luisa::vector<luisa::string> &removed_keys,
    luisa::vector<luisa::string> &added_keys,
    size_t &chunks_unchanged,
    size_t &served_count,
    const RetrievalProgress *progress) {
    auto status = validate_json_object(metadata_json, "metadata_json");
    if (!status.ok()) { return status; }
    const bool chunked = options.chunk_size != 0u && options.by != ChunkOptions::By::off;
    const auto chunks = chunk_text(text, options);
    const luisa::string base_key = path.empty() ?
                                       luisa::format("doc\x1f{}", doc_id) :
                                       _path_key(comparison_path);

    // Idempotent re-ingestion: same text, metadata and chunk options leaves
    // the document untouched (reported via chunks_unchanged).
    const auto existing = _doc_chunks.find(doc_id);
    if (existing != _doc_chunks.end() && !existing->second.empty()) {
        const auto record = _documents.find(existing->second.front());
        if (record != _documents.end() && record->second.text == text &&
            record->second.metadata_json == metadata_json &&
            record->second.chunk_options == options) {
            chunks_unchanged = chunks.size();
            return {};
        }
    }

    // Encode every chunk into pending records BEFORE touching the index, so a
    // failure or a cancellation leaves the previous document fully intact
    // (per-document transaction, same contract as _apply_file).
    served_count = 0u;
    luisa::vector<DocumentRecord> pending;
    pending.reserve(chunks.size());
    for (size_t i = 0u; i < chunks.size(); i++) {
        if (cancelled(progress)) {
            return make_error(RetrievalErrorCode::operational, "cancelled",
                              RetrievalSubCode::cancelled);
        }
        const auto body = chunks[i].text(text);
        RetrievalEmbedding embedding;
        bool served_from_cache = false;
        if (_embed_cache != nullptr && !path.empty()) {
            const luisa::string variant = chunked ? _chunk_variant(
                static_cast<uint32_t>(i), options) : luisa::string{};
            luisa::vector<float> dense;
            if (_embed_cache->load_variant(path, text, variant, dense) &&
                (_dimension == 0u || dense.size() == _dimension) &&
                std::ranges::all_of(dense, [](const float v) { return std::isfinite(v); })) {
                if (!_hybrid) {
                    embedding.dense = std::move(dense);
                    served_from_cache = true;
                } else {
                    luisa::vector<std::pair<uint32_t, float>> sparse;
                    luisa::vector<std::pair<uint32_t, float>> canonical;
                    const bool sparse_hit =
                        _embed_cache->load_sparse_variant(path, text, variant, sparse) &&
                        canonicalize_sparse(
                            luisa::span<const std::pair<uint32_t, float>>{sparse.data(),
                                                                         sparse.size()},
                            canonical)
                            .ok();
                    // ColBERT matrices are never cached: an opted-in index
                    // re-encodes rather than serving a colbert-less hit.
                    if (sparse_hit && !_want_colbert) {
                        embedding.dense = std::move(dense);
                        embedding.sparse = std::move(canonical);
                        served_from_cache = true;
                    }
                }
            }
            _record_encode(0.0, served_from_cache);
            if (served_from_cache) { served_count++; }
        }
        if (!served_from_cache) {
            const auto began = std::chrono::steady_clock::now();
            embedding = RetrievalEmbedding{};
            status = _encode(body, options.max_seq_len, _want_colbert, embedding);
            const auto elapsed = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - began)
                                     .count();
            if (_embed_cache != nullptr && !path.empty()) {
                _record_encode(elapsed, false);
            } else {
                _encode_count++;
                _encode_time_total_ms += elapsed;
            }
            if (!status.ok()) {
                return status;
            }
            if (_embed_cache != nullptr && !path.empty()) {
                const luisa::string variant = chunked ? _chunk_variant(
                    static_cast<uint32_t>(i), options) : luisa::string{};
                _embed_cache->store_variant(
                    path, text, variant,
                    luisa::span<const float>{embedding.dense.data(), embedding.dense.size()});
                if (_hybrid) {
                    _embed_cache->store_sparse_variant(
                        path, text, variant,
                        luisa::span<const std::pair<uint32_t, float>>{embedding.sparse.data(),
                                                                     embedding.sparse.size()});
                }
            }
        }
        DocumentRecord record;
        record.path = path;
        record.comparison_path = comparison_path;
        record.root_key = root_key;
        record.display_path = path.empty() ? luisa::string{} : _display_path(path);
        record.doc_id = doc_id;
        record.metadata_json = luisa::string{metadata_json};
        record.chunk_index = static_cast<uint32_t>(i);
        record.byte_offset = chunks[i].byte_offset;
        record.byte_length = chunks[i].byte_length;
        record.embedding = std::move(embedding);
        if (i == 0u) {
            // Only chunk 0 carries the full text / options (RAM bound).
            record.text = luisa::string{text};
            record.chunk_options = options;
        }
        pending.push_back(std::move(record));
        report_progress(progress, i + 1u, chunks.size());
    }

    // Transaction point: swap the document's chunk records.
    if (existing != _doc_chunks.end()) {
        removed_keys = existing->second;
    }
    _drop_document(doc_id);
    added_keys.reserve(pending.size());
    for (size_t i = 0u; i < pending.size(); i++) {
        luisa::string key = chunked ? _chunk_key(base_key, static_cast<uint32_t>(i)) : base_key;
        added_keys.push_back(key);
        _documents.emplace(std::move(key), std::move(pending[i]));
    }
    _doc_chunks[doc_id] = added_keys;
    if (_dimension == 0u) {
        _dimension = _documents.begin()->second.embedding.dense.size();
    }
    return {};
}

RetrievalStatus RetrievalIndex::_apply_file(
    const luisa::string &root_key,
    const luisa::filesystem::path &file,
    FileOutcome &outcome,
    bool &cache_hit,
    const RetrievalProgress *progress) {
    cache_hit = false;
    luisa::filesystem::path display;
    luisa::filesystem::path comparison;
    luisa::string key;
    auto status = _normalize_path(file, true, display, comparison, key);
    if (!status.ok()) {
        return status;
    }
    const auto root = _roots.find(root_key);
    const ChunkOptions options = root != _roots.end() ? root->second.chunk_options
                                                      : ChunkOptions::whole_file();
    luisa::string contents;
    status = _read_text_file(display, contents);
    if (!status.ok()) {
        return status;
    }
    const auto doc_id = _display_path(display);
    const bool existed = _doc_chunks.find(doc_id) != _doc_chunks.end();
    luisa::vector<luisa::string> removed_keys;
    luisa::vector<luisa::string> added_keys;
    size_t chunks_unchanged = 0u;
    size_t served_count = 0u;
    status = _ingest_document(doc_id, display, comparison, root_key, contents,
                              luisa::string_view{}, options, removed_keys, added_keys,
                              chunks_unchanged, served_count, progress);
    if (!status.ok()) {
        return status;
    }
    if (!added_keys.empty() || !removed_keys.empty()) {
        outcome = existed ? FileOutcome::updated : FileOutcome::added;
    } else {
        outcome = FileOutcome::unchanged;
    }
    // cache_hit reports a fully cache-served document.
    const auto chunk_total = chunks_unchanged > 0u ? chunks_unchanged : added_keys.size();
    cache_hit = chunk_total > 0u && served_count == chunk_total;
    return {};
}

// --- directory load --------------------------------------------------------------

RetrievalLoadResult RetrievalIndex::load(const luisa::filesystem::path &root) {
    const auto extensions = default_extensions();
    return load(root, luisa::span<const luisa::string>{extensions.data(), extensions.size()});
}

RetrievalLoadResult RetrievalIndex::load(
    const luisa::filesystem::path &root,
    luisa::span<const luisa::string> extensions) {
    return load(root, extensions, ChunkOptions::whole_file(), nullptr);
}

RetrievalLoadResult RetrievalIndex::load(
    const luisa::filesystem::path &root,
    luisa::span<const luisa::string> extensions,
    const ChunkOptions &options,
    const RetrievalProgress *progress) {
    RetrievalLoadResult result;
    auto status = normalize_extensions(extensions, result.extensions);
    if (!status.ok()) {
        result.status = std::move(status);
        result.total_documents = _documents.size();
        return result;
    }
    luisa::filesystem::path display;
    luisa::filesystem::path comparison;
    luisa::string key;
    status = _normalize_path(root, true, display, comparison, key);
    if (!status.ok()) {
        result.status = std::move(status);
        result.total_documents = _documents.size();
        return result;
    }
    std::error_code ec;
    if (!luisa::filesystem::is_directory(display, ec) || ec) {
        result.status = make_error(RetrievalErrorCode::invalid_argument,
                                   luisa::format("load root '{}' is not a directory", _display_path(display)));
        result.total_documents = _documents.size();
        return result;
    }
    if (_roots.find(key) != _roots.end()) {
        result.status = make_error(RetrievalErrorCode::invalid_argument,
                                   luisa::format("root '{}' is already loaded", _display_path(display)));
        result.total_documents = _documents.size();
        return result;
    }
    for (const auto &entry : _roots) {
        if (_is_within(comparison, entry.second.comparison_path) ||
            _is_within(entry.second.comparison_path, comparison)) {
            result.status = make_error(
                RetrievalErrorCode::invalid_argument,
                luisa::format("root '{}' overlaps loaded root '{}'",
                              _display_path(display), _display_path(entry.second.path)));
            result.total_documents = _documents.size();
            return result;
        }
    }
    RootRecord record;
    record.path = display;
    record.comparison_path = comparison;
    record.extensions = result.extensions;
    record.chunk_options = options;
    _roots.emplace(key, std::move(record));
    result.root = _display_path(display);

    if (_embed_cache != nullptr) {
        _embed_cache->begin_directory(
            display,
            luisa::span<const luisa::string>{result.extensions.data(), result.extensions.size()});
    }
    auto discovery = _discover(display, result.extensions);
    result.skipped = discovery.skipped;
    result.failed = discovery.failed;
    uint64_t done = 0u;
    for (const auto &file : discovery.files) {
        if (cancelled(progress)) {
            result.cancelled = true;
            result.status = make_error(RetrievalErrorCode::operational, "cancelled",
                                       RetrievalSubCode::cancelled);
            // Roll back the root registration (the documents applied so far
            // stay tracked and queryable): a follow-up load without the flag
            // then resumes the directory, reporting the already-ingested
            // files as unchanged instead of "root already loaded".
            _roots.erase(key);
            break;
        }
        FileOutcome outcome = FileOutcome::added;
        bool cache_hit = false;
        status = _apply_file(key, file, outcome, cache_hit, progress);
        if (!status.ok()) {
            result.failed++;
            LUISA_WARNING("retrieval load skipped '{}': {}", _display_path(file), status.message);
            done++;
            continue;
        }
        if (outcome == FileOutcome::added) {
            result.added++;
        }
        if (cache_hit) {
            result.cached++;
        }
        done++;
        report_progress(progress, done, discovery.files.size());
    }
    if (_embed_cache != nullptr) {
        _embed_cache->end_directory();
    }
    result.total_documents = _documents.size();
    return result;
}

RetrievalUnloadResult RetrievalIndex::unload(const luisa::filesystem::path &root) {
    RetrievalUnloadResult result;
    luisa::filesystem::path display;
    luisa::filesystem::path comparison;
    luisa::string key;
    auto status = _normalize_path(root, false, display, comparison, key);
    if (!status.ok()) {
        result.status = std::move(status);
        result.total_documents = _documents.size();
        return result;
    }
    const auto root_it = _roots.find(key);
    if (root_it == _roots.end()) {
        result.status = make_error(RetrievalErrorCode::not_found,
                                   luisa::format("root '{}' is not loaded", _display_path(display)));
        result.total_documents = _documents.size();
        return result;
    }
    result.root = _display_path(root_it->second.path);
    luisa::vector<luisa::string> remove_keys;
    for (const auto &entry : _documents) {
        if (entry.second.root_key == key) {
            remove_keys.push_back(entry.first);
        }
    }
    for (const auto &document_key : remove_keys) {
        const auto record = _documents.find(document_key);
        if (record != _documents.end()) {
            _doc_chunks.erase(record->second.doc_id);
        }
        _documents.erase(document_key);
    }
    result.removed_documents = remove_keys.size();
    _roots.erase(key);
    if (_documents.empty()) {
        _dimension = 0u;
    }
    result.total_documents = _documents.size();
    return result;
}

// --- in-memory documents (report.md §2) -----------------------------------------

RetrievalDocumentResult RetrievalIndex::add_document(
    luisa::string doc_id,
    luisa::string_view text,
    luisa::string_view metadata_json,
    const ChunkOptions &options,
    const RetrievalProgress *progress) {
    RetrievalDocumentResult result;
    result.doc_id = doc_id;
    if (doc_id.empty()) {
        result.status = make_error(RetrievalErrorCode::invalid_argument,
                                   "doc_id must not be empty");
        return result;
    }
    if (doc_id.find('\0') != luisa::string::npos || !valid_utf8(doc_id)) {
        result.status = make_error(RetrievalErrorCode::invalid_argument,
                                   "doc_id must be valid UTF-8 without embedded NUL bytes");
        return result;
    }
    if (!text.empty() && !valid_utf8(text)) {
        result.status = make_error(RetrievalErrorCode::invalid_argument,
                                   "document text must be valid UTF-8");
        return result;
    }
    luisa::vector<luisa::string> removed_keys;
    luisa::vector<luisa::string> added_keys;
    size_t chunks_unchanged = 0u;
    size_t served_count = 0u;
    auto status = _ingest_document(luisa::string{doc_id}, {}, {}, {}, text, metadata_json,
                                   options, removed_keys, added_keys, chunks_unchanged,
                                   served_count, progress);
    result.status = std::move(status);
    result.chunks_added = added_keys.size();
    result.chunks_removed = removed_keys.size();
    result.chunks_unchanged = chunks_unchanged;
    result.total_documents = _documents.size();
    result.total_docs = _doc_chunks.size();
    return result;
}

RetrievalDocumentResult RetrievalIndex::remove_document(luisa::string_view doc_id) {
    RetrievalDocumentResult result;
    result.doc_id = luisa::string{doc_id};
    const auto doc = _doc_chunks.find(doc_id);
    if (doc == _doc_chunks.end()) {
        result.status = make_error(RetrievalErrorCode::not_found,
                                   luisa::format("document '{}' is not indexed", doc_id));
        result.total_documents = _documents.size();
        result.total_docs = _doc_chunks.size();
        return result;
    }
    result.chunks_removed = doc->second.size();
    _drop_document(doc_id);
    result.total_documents = _documents.size();
    result.total_docs = _doc_chunks.size();
    return result;
}

// --- update ----------------------------------------------------------------------

RetrievalUpdateResult RetrievalIndex::_update_file(
    const luisa::filesystem::path &display_path,
    const luisa::string &comparison_key,
    RootRecord &root,
    const luisa::string &root_key) {
    RetrievalUpdateResult result;
    result.path = _display_path(display_path);
    if (!_extension_admitted(display_path, root.extensions)) {
        result.status = make_error(RetrievalErrorCode::invalid_argument,
                                   luisa::format("file '{}' does not match the owning root's extension rule",
                                                 result.path));
        result.total_documents = _documents.size();
        return result;
    }
    auto existing = _documents.find(comparison_key);
    std::error_code ec;
    const bool regular_file = luisa::filesystem::is_regular_file(display_path, ec);
    if (ec && !file_not_found(ec)) {
        if (existing != _documents.end()) {
            result.failed = 1u;
            LUISA_WARNING("retrieval update could not stat '{}': {}", result.path, ec.message());
        } else {
            result.status = make_error(RetrievalErrorCode::operational,
                                       luisa::format("cannot inspect '{}': {}", result.path, ec.message()),
                                       RetrievalSubCode::io);
        }
        result.total_documents = _documents.size();
        return result;
    }
    if (!regular_file || file_not_found(ec)) {
        if (existing != _documents.end()) {
            const auto doc_id = existing->second.doc_id;
            result.removed = _documents.erase(comparison_key);
            _doc_chunks.erase(doc_id);
            if (_documents.empty()) {
                _dimension = 0u;
            }
        } else {
            result.status = make_error(RetrievalErrorCode::not_found,
                                       luisa::format("file '{}' does not exist or is not a regular file",
                                                     result.path));
        }
        result.total_documents = _documents.size();
        return result;
    }
    FileOutcome outcome = FileOutcome::unchanged;
    if (_embed_cache != nullptr) {
        _embed_cache->begin_directory(
            display_path,
            luisa::span<const luisa::string>{root.extensions.data(), root.extensions.size()});
    }
    bool cache_hit = false;
    auto status = _apply_file(root_key, display_path, outcome, cache_hit, nullptr);
    if (_embed_cache != nullptr) {
        _embed_cache->end_directory();
    }
    if (!status.ok()) {
        result.failed = 1u;
        LUISA_WARNING("retrieval update failed for '{}': {}", result.path, status.message);
    } else {
        switch (outcome) {
            case FileOutcome::added: result.added = 1u; break;
            case FileOutcome::updated: result.updated = 1u; break;
            case FileOutcome::unchanged: result.unchanged = 1u; break;
        }
        if (cache_hit) {
            result.cached = 1u;
        }
    }
    result.total_documents = _documents.size();
    return result;
}

RetrievalUpdateResult RetrievalIndex::_update_directory(
    const luisa::filesystem::path &display_path,
    const luisa::filesystem::path &comparison_path,
    RootRecord &root,
    const luisa::string &root_key) {
    RetrievalUpdateResult result;
    result.path = _display_path(display_path);
    if (_embed_cache != nullptr) {
        _embed_cache->begin_directory(
            display_path,
            luisa::span<const luisa::string>{root.extensions.data(), root.extensions.size()});
    }
    DiscoveryResult discovery;
    std::error_code ec;
    const bool is_directory = luisa::filesystem::is_directory(display_path, ec);
    if (ec && !file_not_found(ec)) {
        result.failed = 1u;
    } else if (is_directory) {
        discovery = _discover(display_path, root.extensions);
        result.failed = discovery.failed;
    }
    for (const auto &file : discovery.files) {
        FileOutcome outcome = FileOutcome::unchanged;
        bool cache_hit = false;
        auto status = _apply_file(root_key, file, outcome, cache_hit, nullptr);
        if (!status.ok()) {
            result.failed++;
            LUISA_WARNING("retrieval update failed for '{}': {}", _display_path(file), status.message);
            continue;
        }
        switch (outcome) {
            case FileOutcome::added: result.added++; break;
            case FileOutcome::updated: result.updated++; break;
            case FileOutcome::unchanged: result.unchanged++; break;
        }
        if (cache_hit) {
            result.cached++;
        }
    }

    luisa::vector<luisa::string> remove_keys;
    for (const auto &entry : _documents) {
        const auto &document = entry.second;
        if (document.root_key != root_key ||
            !_is_within(document.comparison_path, comparison_path)) {
            continue;
        }
        std::error_code ec;
        const bool still_regular = luisa::filesystem::is_regular_file(document.path, ec);
        if (ec && !file_not_found(ec)) {
            result.failed++;
        } else if (file_not_found(ec) || !still_regular || !_extension_admitted(document.path, root.extensions)) {
            remove_keys.push_back(entry.first);
        }
    }
    for (const auto &key : remove_keys) {
        const auto record = _documents.find(key);
        if (record != _documents.end()) {
            _doc_chunks.erase(record->second.doc_id);
        }
        _documents.erase(key);
    }
    result.removed = remove_keys.size();
    if (_documents.empty()) {
        _dimension = 0u;
    }
    if (_embed_cache != nullptr) {
        _embed_cache->end_directory();
    }
    result.total_documents = _documents.size();
    return result;
}

RetrievalUpdateResult RetrievalIndex::update(const luisa::filesystem::path &path) {
    RetrievalUpdateResult result;
    luisa::filesystem::path display;
    luisa::filesystem::path comparison;
    luisa::string key;
    auto status = _normalize_path(path, false, display, comparison, key);
    if (!status.ok()) {
        result.status = std::move(status);
        result.total_documents = _documents.size();
        return result;
    }
    auto *root = _find_owning_root(comparison);
    if (root == nullptr) {
        result.status = make_error(RetrievalErrorCode::not_found,
                                   luisa::format("path '{}' is outside every loaded root",
                                                 _display_path(display)));
        result.total_documents = _documents.size();
        return result;
    }
    luisa::string root_key;
    for (const auto &entry : _roots) {
        if (&entry.second == root) {
            root_key = entry.first;
            break;
        }
    }

    std::error_code ec;
    const auto file_status = luisa::filesystem::status(display, ec);
    const bool exists = !ec && luisa::filesystem::exists(file_status);
    if (exists && luisa::filesystem::is_regular_file(file_status)) {
        return _update_file(display, key, *root, root_key);
    }
    if (exists && luisa::filesystem::is_directory(file_status)) {
        return _update_directory(display, comparison, *root, root_key);
    }
    if (!exists) {
        if (comparison == root->comparison_path) {
            return _update_directory(display, comparison, *root, root_key);
        }
        if (_documents.find(key) != _documents.end()) {
            return _update_file(display, key, *root, root_key);
        }
        bool has_tracked_descendant = false;
        for (const auto &entry : _documents) {
            if (entry.second.root_key == root_key &&
                _is_within(entry.second.comparison_path, comparison)) {
                has_tracked_descendant = true;
                break;
            }
        }
        if (has_tracked_descendant) {
            return _update_directory(display, comparison, *root, root_key);
        }
    }
    result.status = make_error(RetrievalErrorCode::not_found,
                               luisa::format("update path '{}' is not a file, directory, or tracked deletion",
                                             _display_path(display)));
    result.total_documents = _documents.size();
    return result;
}

// --- query -----------------------------------------------------------------------

RetrievalQueryResult RetrievalIndex::query(luisa::string_view text, uint32_t top_k) {
    return query(text, top_k, nullptr);
}

RetrievalQueryResult RetrievalIndex::query(
    luisa::string_view text,
    uint32_t top_k,
    const RetrievalQueryOptions *options) {
    RetrievalQueryResult result;
    const RetrievalQueryOptions defaults;
    const RetrievalQueryOptions &opt = options == nullptr ? defaults : *options;
    if (text.empty() || blank_text(text)) {
        result.status = make_error(RetrievalErrorCode::invalid_argument,
                                   "query text must not be empty or blank");
        return result;
    }
    if (top_k == 0u || top_k > kRetrievalMaxTopK) {
        result.status = make_error(
            RetrievalErrorCode::invalid_argument,
            luisa::format("top_k must be in [1, {}]", kRetrievalMaxTopK));
        return result;
    }
    if (opt.diversity < 0.0 || opt.diversity > 1.0) {
        result.status = make_error(RetrievalErrorCode::invalid_argument,
                                   "diversity must be in [0, 1]");
        return result;
    }
    if (_documents.empty()) {
        LUISA_WARNING("retrieval query rejected: the document corpus is empty");
        result.status = make_error(RetrievalErrorCode::operational,
                                   "cannot query an empty document corpus",
                                   RetrievalSubCode::index);
        return result;
    }
    double dense_weight = kRetrievalDenseWeight;
    double sparse_weight = kRetrievalSparseWeight;
    if (std::isfinite(opt.dense_weight) && opt.dense_weight >= 0.0) {
        dense_weight = opt.dense_weight;
    }
    if (std::isfinite(opt.sparse_weight) && opt.sparse_weight >= 0.0) {
        sparse_weight = opt.sparse_weight;
    }
    luisa::unique_ptr<Filter> filter;
    if (!opt.filter_json.empty()) {
        auto status = parse_filter(opt.filter_json, filter);
        if (!status.ok()) {
            result.status = std::move(status);
            return result;
        }
    }
    if (opt.want_colbert && !_want_colbert) {
        result.status = make_error(
            RetrievalErrorCode::operational,
            "ColBERT re-ranking requires the EER_CREATE_WANT_COLBERT create flag",
            RetrievalSubCode::index);
        return result;
    }
    // Instruction prefix (report.md §4): prepended verbatim when requested;
    // NULL/empty preserves today's exact query encoding.
    const luisa::string prefixed = opt.instruction_prefix.empty() ?
                                       luisa::string{} :
                                       luisa::format("{}{}", opt.instruction_prefix, text);
    const luisa::string_view query_text = prefixed.empty() ? text
                                                           : luisa::string_view{prefixed};
    RetrievalEmbedding query_embedding;
    auto status = _encode(query_text, opt.max_seq_len, opt.want_colbert, query_embedding);
    if (!status.ok()) {
        result.status = std::move(status);
        return result;
    }
    result.truncated = opt.report_truncation && query_embedding.truncated;

    struct Scored {
        RetrievalHit hit;
        // Pointers into the chunk record's dense vector (stable: the service
        // holds the index lock for the whole query); used by MMR.
        const float *dense = nullptr;
        size_t dense_size = 0u;
    };
    luisa::vector<Scored> scored;
    scored.reserve(_documents.size());
    for (const auto &entry : _documents) {
        const auto &document = entry.second;
        if (document.embedding.dense.size() != query_embedding.dense.size()) {
            result.status = make_error(RetrievalErrorCode::operational,
                                       "index contains inconsistent embedding dimensions",
                                       RetrievalSubCode::index);
            return result;
        }
        if (filter != nullptr && !filter->matches(document.metadata_json)) {
            continue;// applied BEFORE top-k selection (report.md §4)
        }
        double dense_score = 0.0;
        for (size_t i = 0u; i < query_embedding.dense.size(); i++) {
            dense_score += static_cast<double>(query_embedding.dense[i]) *
                           static_cast<double>(document.embedding.dense[i]);
        }
        double sparse_score = 0.0;
        if (_hybrid) {
            // Sparse term: two-pointer cosine over canonical vectors (both norms
            // are 1), fused with the dense dot per phase H1 / per-query weights.
            sparse_score = _sparse_dot(
                luisa::span<const std::pair<uint32_t, float>>{query_embedding.sparse.data(),
                                                              query_embedding.sparse.size()},
                luisa::span<const std::pair<uint32_t, float>>{document.embedding.sparse.data(),
                                                              document.embedding.sparse.size()});
        }
        Scored item;
        item.hit.path = document.display_path;
        item.hit.doc_id = document.doc_id;
        item.hit.dense_score = dense_score;
        item.hit.sparse_score = _hybrid ? sparse_score : 0.0;
        item.hit.score = _hybrid ? dense_weight * dense_score + sparse_weight * sparse_score
                                 : dense_score;
        item.hit.chunk_index = document.chunk_index;
        item.hit.byte_offset = document.byte_offset;
        item.hit.byte_length = document.byte_length;
        item.dense = document.embedding.dense.data();
        item.dense_size = document.embedding.dense.size();
        scored.push_back(std::move(item));
    }
    luisa::sort(scored.begin(), scored.end(), [](const auto &a, const auto &b) {
        if (a.hit.score != b.hit.score) {
            return a.hit.score > b.hit.score;
        }
        if (a.hit.doc_id != b.hit.doc_id) {
            return a.hit.doc_id < b.hit.doc_id;
        }
        return a.hit.chunk_index < b.hit.chunk_index;
    });

    const size_t pool_size = scored.size() < static_cast<size_t>(std::max(top_k * 4u, top_k + 20u)) ?
                                 scored.size() :
                                 static_cast<size_t>(std::max(top_k * 4u, top_k + 20u));
    const size_t wanted = std::min(static_cast<size_t>(top_k), scored.size());

    if (opt.want_colbert) {
        // ColBERT MaxSim re-ranking (report.md §5): the fused dense/sparse list
        // is the shortlist; token-level late interaction decides the order.
        if (query_embedding.colbert.empty()) {
            result.status = make_error(RetrievalErrorCode::operational,
                                       "the query produced an empty ColBERT encoding",
                                       RetrievalSubCode::encode);
            return result;
        }
        luisa::vector<Scored> pool{scored.begin(), scored.begin() + static_cast<ptrdiff_t>(pool_size)};
        for (size_t i = 0u; i < pool.size(); i++) {
            // Look the chunk record back up by identity for its colbert matrix.
            double best = 0.0;
            bool found = false;
            bool has_colbert = false;
            for (const auto &entry : _documents) {
                const auto &document = entry.second;
                if (document.doc_id != pool[i].hit.doc_id ||
                    document.chunk_index != pool[i].hit.chunk_index) {
                    continue;
                }
                if (document.embedding.colbert.empty()) {
                    // Degrade instead of failing the whole query: a snapshot
                    // written before ColBERT serialization, or a corpus
                    // ingested without EER_CREATE_WANT_COLBERT, has no matrix
                    // for this record; it keeps its fused dense/sparse score.
                    found = true;// best stays 0.0, the fused score is kept
                    break;
                }
                best = _colbert_maxsim(
                    luisa::span<const float>{query_embedding.colbert.data(),
                                             query_embedding.colbert.size()},
                    luisa::span<const float>{document.embedding.colbert.data(),
                                             document.embedding.colbert.size()},
                    static_cast<uint32_t>(query_embedding.dense.size()));
                found = true;
                has_colbert = true;
                break;
            }
            if (!found) {
                result.status = make_error(RetrievalErrorCode::operational,
                                           "index changed while ranking",
                                           RetrievalSubCode::index);
                return result;
            }
            pool[i].hit.colbert_score = has_colbert ? best : 0.0;
            if (has_colbert) { pool[i].hit.score = best; }
        }
        luisa::sort(pool.begin(), pool.end(), [](const auto &a, const auto &b) {
            if (a.hit.score != b.hit.score) {
                return a.hit.score > b.hit.score;
            }
            if (a.hit.doc_id != b.hit.doc_id) {
                return a.hit.doc_id < b.hit.doc_id;
            }
            return a.hit.chunk_index < b.hit.chunk_index;
        });
        for (size_t i = 0u; i < wanted && i < pool.size(); i++) {
            result.hits.push_back(std::move(pool[i].hit));
        }
        return result;
    }

    if (opt.diversity > 0.0) {
        // MMR re-ranking (report.md §4): relevance stays (1 - d) * fused score,
        // novelty is d * (1 - max cosine against already-selected passages).
        luisa::vector<Scored> pool{scored.begin(), scored.begin() + static_cast<ptrdiff_t>(pool_size)};
        luisa::vector<bool> selected(pool.size(), false);
        result.hits.reserve(wanted);
        for (size_t pick = 0u; pick < wanted; pick++) {
            std::ptrdiff_t best_index = -1;
            double best_value = -std::numeric_limits<double>::max();
            for (size_t i = 0u; i < pool.size(); i++) {
                if (selected[i]) { continue; }
                double mmr = (1.0 - opt.diversity) * pool[i].hit.score;
                if (pick > 0u) {
                    double max_cosine = -1.0;
                    for (size_t j = 0u; j < pool.size(); j++) {
                        if (!selected[j]) { continue; }
                        // Dense cosine between the two passages (L2-normalized
                        // embeddings: the dot product IS the cosine).
                        const auto n = std::min(pool[i].dense_size, pool[j].dense_size);
                        double cosine = 0.0;
                        for (size_t k = 0u; k < n; k++) {
                            cosine += static_cast<double>(pool[i].dense[k]) *
                                      static_cast<double>(pool[j].dense[k]);
                        }
                        if (cosine > max_cosine) { max_cosine = cosine; }
                    }
                    mmr += opt.diversity * (1.0 - max_cosine);
                }
                if (mmr > best_value) {
                    best_value = mmr;
                    best_index = static_cast<std::ptrdiff_t>(i);
                }
            }
            if (best_index < 0) { break; }
            selected[static_cast<size_t>(best_index)] = true;
            result.hits.push_back(std::move(pool[static_cast<size_t>(best_index)].hit));
        }
        return result;
    }

    for (size_t i = 0u; i < wanted; i++) {
        result.hits.push_back(std::move(scored[i].hit));
    }
    return result;
}

// --- statistics (report.md §6) ----------------------------------------------------

RetrievalStats RetrievalIndex::stats() const noexcept {
    RetrievalStats result;
    result.document_count = _doc_chunks.size();
    result.chunk_count = _documents.size();
    result.dimension = static_cast<uint32_t>(_dimension);
    result.hybrid = _hybrid;
    result.cache_hits = _cache_hits;
    result.cache_misses = _cache_misses;
    const auto consulted = _cache_hits + _cache_misses;
    result.cache_hit_ratio = consulted == 0u ? 0.0
                                             : static_cast<double>(_cache_hits) /
                                                   static_cast<double>(consulted);
    result.encode_count = _encode_count;
    result.mean_encode_ms = _encode_count == 0u ? 0.0
                                                : _encode_time_total_ms /
                                                      static_cast<double>(_encode_count);
    return result;
}

// --- snapshots (report.md §3) ------------------------------------------------------

namespace {

constexpr uint32_t kSnapshotVersion = 2u;

[[nodiscard]] luisa::string read_binary_file(const luisa::filesystem::path &path,
                                             bool &ok) {
    std::ifstream stream{path, std::ios::binary};
    if (!stream.is_open()) {
        ok = false;
        return {};
    }
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    if (end < 0) {
        ok = false;
        return {};
    }
    luisa::string contents;
    contents.resize(static_cast<size_t>(end));
    stream.seekg(0, std::ios::beg);
    if (!contents.empty()) {
        stream.read(contents.data(), static_cast<std::streamsize>(contents.size()));
        if (!stream) {
            ok = false;
            return {};
        }
    }
    ok = true;
    return contents;
}

[[nodiscard]] bool write_binary_file(const luisa::filesystem::path &path,
                                     luisa::span<const std::byte> data) {
    std::ofstream stream{path, std::ios::binary | std::ios::trunc};
    if (!stream.is_open()) { return false; }
    if (!data.empty()) {
        stream.write(reinterpret_cast<const char *>(data.data()),
                     static_cast<std::streamsize>(data.size_bytes()));
    }
    stream.close();
    return static_cast<bool>(stream);
}

void add_string(yyjson_mut_doc *doc, yyjson_mut_val *object,
                const char *key, luisa::string_view value) {
    // yyjson_mut_strn() requires a non-null pointer even for len == 0, and an
    // empty string_view's data() may be null: route empties through the
    // NUL-terminated constructor.
    // Copying variant: several callers hand in temporaries (display paths),
    // and yyjson_mut_strn would keep a dangling pointer.
    auto *string_value = value.empty() ? yyjson_mut_str(doc, "")
                                       : yyjson_mut_strncpy(doc, value.data(), value.size());
    yyjson_mut_obj_add(object, yyjson_mut_str(doc, key), string_value);
}

}// namespace

RetrievalStatus RetrievalIndex::save(const luisa::filesystem::path &path) {
    if (path.empty()) {
        return make_error(RetrievalErrorCode::invalid_argument, "snapshot path must not be empty");
    }
    // Deterministic order: (doc_id, chunk_index).
    luisa::vector<const DocumentRecord *> documents;
    documents.reserve(_documents.size());
    for (const auto &entry : _documents) {
        documents.push_back(&entry.second);
    }
    luisa::sort(documents.begin(), documents.end(), [](const auto *a, const auto *b) {
        if (a->doc_id != b->doc_id) { return a->doc_id < b->doc_id; }
        return a->chunk_index < b->chunk_index;
    });

    yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
    if (doc == nullptr) {
        return make_error(RetrievalErrorCode::operational, "out of memory",
                          RetrievalSubCode::index);
    }
    auto *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add(root, yyjson_mut_str(doc, "version"), yyjson_mut_uint(doc, kSnapshotVersion));
    yyjson_mut_obj_add(root, yyjson_mut_str(doc, "dimension"), yyjson_mut_uint(doc, _dimension));
    yyjson_mut_obj_add(root, yyjson_mut_str(doc, "hybrid"), yyjson_mut_bool(doc, _hybrid));

    auto *roots = yyjson_mut_arr(doc);
    yyjson_mut_obj_add(root, yyjson_mut_str(doc, "roots"), roots);
    luisa::vector<const luisa::string *> root_keys;
    for (const auto &entry : _roots) { root_keys.push_back(&entry.first); }
    luisa::sort(root_keys.begin(), root_keys.end(), [](const auto *a, const auto *b) {
        return *a < *b;
    });
    for (const auto *key : root_keys) {
        const auto &record = _roots.find(*key)->second;
        auto *object = yyjson_mut_obj(doc);
        yyjson_mut_arr_append(roots, object);
        add_string(doc, object, "key", *key);
        add_string(doc, object, "path", _display_path(record.path));
        auto *extensions = yyjson_mut_arr(doc);
        yyjson_mut_obj_add(object, yyjson_mut_str(doc, "extensions"), extensions);
        for (const auto &extension : record.extensions) {
            yyjson_mut_arr_add_strncpy(doc, extensions, extension.data(), extension.size());
        }
        yyjson_mut_obj_add(object, yyjson_mut_str(doc, "chunk_size"),
                           yyjson_mut_uint(doc, record.chunk_options.chunk_size));
        yyjson_mut_obj_add(object, yyjson_mut_str(doc, "chunk_overlap"),
                           yyjson_mut_uint(doc, record.chunk_options.chunk_overlap));
        yyjson_mut_obj_add(object, yyjson_mut_str(doc, "chunk_by"),
                           yyjson_mut_uint(doc, static_cast<uint32_t>(record.chunk_options.by)));
        yyjson_mut_obj_add(object, yyjson_mut_str(doc, "chunk_max_seq_len"),
                           yyjson_mut_uint(doc, record.chunk_options.max_seq_len));
    }

    // The blob sits next to the manifest; the manifest references it by file
    // name only, so the pair is relocatable as a unit.
    const luisa::string blob_name = _path_key(path.filename()) + ".bin";
    add_string(doc, root, "blob", blob_name);
    // ColBERT matrices (opt-in via set_store_colbert) ride in a second packed
    // blob; per-record row counts live in the manifest. Snapshots written by
    // earlier 1.2 builds simply have no "colbert_blob" key and restore as
    // colbert-less corpora (queries degrade to the fused score, see query()).
    bool any_colbert = false;
    for (const auto *record : documents) {
        if (!record->embedding.colbert.empty()) {
            any_colbert = true;
            break;
        }
    }
    const luisa::string colbert_blob_name = _path_key(path.filename()) + ".colbert.bin";
    if (any_colbert) { add_string(doc, root, "colbert_blob", colbert_blob_name); }
    auto *documents_json = yyjson_mut_arr(doc);
    yyjson_mut_obj_add(root, yyjson_mut_str(doc, "documents"), documents_json);
    for (const auto *record : documents) {
        auto *object = yyjson_mut_obj(doc);
        yyjson_mut_arr_append(documents_json, object);
        add_string(doc, object, "doc_id", record->doc_id);
        yyjson_mut_obj_add(object, yyjson_mut_str(doc, "chunk_index"),
                           yyjson_mut_uint(doc, record->chunk_index));
        yyjson_mut_obj_add(object, yyjson_mut_str(doc, "byte_offset"),
                           yyjson_mut_uint(doc, record->byte_offset));
        yyjson_mut_obj_add(object, yyjson_mut_str(doc, "byte_length"),
                           yyjson_mut_uint(doc, record->byte_length));
        add_string(doc, object, "metadata", record->metadata_json);
        add_string(doc, object, "root_key", record->root_key);
        if (any_colbert) {
            const auto colbert_size = record->embedding.colbert.size();
            if (colbert_size % _dimension != 0u) {
                yyjson_mut_doc_free(doc);
                return make_error(RetrievalErrorCode::operational,
                                  luisa::format("document '{}' chunk {} has a malformed ColBERT matrix",
                                                record->doc_id, record->chunk_index),
                                  RetrievalSubCode::index);
            }
            yyjson_mut_obj_add(object, yyjson_mut_str(doc, "colbert_rows"),
                               yyjson_mut_uint(doc, colbert_size / _dimension));
        }
        if (record->path.empty()) {
            // Memory-only document: serialized with its text so the snapshot
            // restores a corpus that never existed on disk.
            add_string(doc, object, "text", record->text);
        } else {
            add_string(doc, object, "path", record->display_path);
        }
        if (_hybrid) {
            auto *sparse = yyjson_mut_arr(doc);
            yyjson_mut_obj_add(object, yyjson_mut_str(doc, "sparse"), sparse);
            for (const auto &pair : record->embedding.sparse) {
                auto *entry = yyjson_mut_arr(doc);
                yyjson_mut_arr_append(sparse, entry);
                yyjson_mut_arr_add_uint(doc, entry, pair.first);
                yyjson_mut_arr_add_real(doc, entry, pair.second);
            }
        }
    }
    size_t json_length = 0u;
    yyjson_write_err write_error{};
    char *json = yyjson_mut_write_opts(
        doc, YYJSON_WRITE_PRETTY | YYJSON_WRITE_NEWLINE_AT_END, nullptr,
        &json_length, &write_error);
    if (json == nullptr) {
        LUISA_WARNING("snapshot write failed: code={} msg={}", (int)write_error.code,
                      write_error.msg == nullptr ? "unknown" : write_error.msg);
    }
    yyjson_mut_doc_free(doc);
    if (json == nullptr) {
        return make_error(RetrievalErrorCode::operational, "cannot serialize the snapshot manifest",
                          RetrievalSubCode::io);
    }
    const auto manifest_ok = write_binary_file(
        path, luisa::span<const std::byte>{
                  reinterpret_cast<const std::byte *>(json), json_length});
    std::free(json);
    if (!manifest_ok) {
        return make_error(RetrievalErrorCode::operational,
                          luisa::format("cannot write '{}'", _display_path(path)),
                          RetrievalSubCode::io);
    }

    // Packed float32 dense blob in manifest order (host byte order).
    const auto mul_overflow = [](size_t a, size_t b) noexcept {
        return b != 0u && a > std::numeric_limits<size_t>::max() / b;
    };
    if (mul_overflow(documents.size(), _dimension) ||
        mul_overflow(documents.size() * _dimension, sizeof(float))) {
        return make_error(RetrievalErrorCode::operational,
                          "the snapshot is too large to serialize",
                          RetrievalSubCode::index);
    }
    luisa::vector<std::byte> blob;
    blob.resize(documents.size() * _dimension * sizeof(float));
    for (size_t i = 0u; i < documents.size(); i++) {
        const auto &dense = documents[i]->embedding.dense;
        if (dense.size() != _dimension) {
            return make_error(RetrievalErrorCode::operational,
                              "index contains inconsistent embedding dimensions",
                              RetrievalSubCode::index);
        }
        std::memcpy(blob.data() + i * _dimension * sizeof(float),
                    dense.data(), dense.size() * sizeof(float));
    }
    const auto blob_path = path.parent_path() / luisa::filesystem::path{luisa::string{blob_name}};
    if (!write_binary_file(blob_path, luisa::span<const std::byte>{blob.data(), blob.size()})) {
        return make_error(RetrievalErrorCode::operational,
                          luisa::format("cannot write '{}'", _display_path(blob_path)),
                          RetrievalSubCode::io);
    }

    if (any_colbert) {
        // Packed float32 ColBERT blob in the same manifest order; row counts
        // are in the manifest, so a record with 0 rows occupies no bytes.
        size_t colbert_floats = 0u;
        for (const auto *record : documents) {
            // Every record's matrix is already resident, so the running sum
            // cannot overflow size_t; only the final byte count is checked.
            colbert_floats += record->embedding.colbert.size();
        }
        if (mul_overflow(colbert_floats, sizeof(float))) {
            return make_error(RetrievalErrorCode::operational,
                              "the ColBERT snapshot is too large to serialize",
                              RetrievalSubCode::index);
        }
        luisa::vector<std::byte> colbert_blob;
        colbert_blob.resize(colbert_floats * sizeof(float));
        auto *colbert_out = reinterpret_cast<float *>(colbert_blob.data());
        for (const auto *record : documents) {
            std::memcpy(colbert_out, record->embedding.colbert.data(),
                        record->embedding.colbert.size() * sizeof(float));
            colbert_out += record->embedding.colbert.size();
        }
        const auto colbert_path =
            path.parent_path() / luisa::filesystem::path{luisa::string{colbert_blob_name}};
        if (!write_binary_file(colbert_path,
                               luisa::span<const std::byte>{colbert_blob.data(), colbert_blob.size()})) {
            return make_error(RetrievalErrorCode::operational,
                              luisa::format("cannot write '{}'", _display_path(colbert_path)),
                              RetrievalSubCode::io);
        }
    }
    return {};
}

RetrievalStatus RetrievalIndex::load_snapshot(const luisa::filesystem::path &path) {
    if (path.empty()) {
        return make_error(RetrievalErrorCode::invalid_argument, "snapshot path must not be empty");
    }
    bool ok = false;
    const auto manifest = read_binary_file(path, ok);
    if (!ok) {
        return make_error(RetrievalErrorCode::operational,
                          luisa::format("cannot read '{}'", _display_path(path)),
                          RetrievalSubCode::io);
    }
    yyjson_read_err read_error{};
    auto *doc = yyjson_read_opts(
        const_cast<char *>(manifest.data()),
        manifest.size(), 0, nullptr, &read_error);
    if (doc == nullptr) {
        return make_error(RetrievalErrorCode::operational,
                          luisa::format("'{}' is not a valid snapshot: {}",
                                        _display_path(path), read_error.msg),
                          RetrievalSubCode::io);
    }
    const auto fail_manifest = [doc](luisa::string message) {
        yyjson_doc_free(doc);
        return make_error(RetrievalErrorCode::operational, std::move(message),
                          RetrievalSubCode::index);
    };
    const auto *root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        return fail_manifest("snapshot manifest must be a JSON object");
    }
    const auto *version = yyjson_obj_get(root, "version");
    if (!yyjson_is_uint(version) || yyjson_get_uint(version) != kSnapshotVersion) {
        return fail_manifest("unsupported snapshot version");
    }
    const auto *dimension_value = yyjson_obj_get(root, "dimension");
    if (!yyjson_is_uint(dimension_value)) {
        return fail_manifest("snapshot manifest misses the embedding dimension");
    }
    const auto dimension = static_cast<size_t>(yyjson_get_uint(dimension_value));
    if (dimension == 0u) {
        return fail_manifest("snapshot manifest has a zero embedding dimension");
    }
    if (_dimension != 0u && _dimension != dimension) {
        return fail_manifest(luisa::format(
            "snapshot dimension {} does not match the current index dimension {}", dimension, _dimension));
    }
    const auto *hybrid_value = yyjson_obj_get(root, "hybrid");
    if (yyjson_is_bool(hybrid_value) && yyjson_get_bool(hybrid_value) != _hybrid) {
        return fail_manifest("snapshot hybrid mode does not match this index");
    }
    const auto *blob_value = yyjson_obj_get(root, "blob");
    if (!yyjson_is_str(blob_value)) {
        return fail_manifest("snapshot manifest misses the blob reference");
    }
    const auto *documents_value = yyjson_obj_get(root, "documents");
    if (!yyjson_is_arr(documents_value)) {
        return fail_manifest("snapshot manifest misses the document list");
    }
    const luisa::string blob_name = yyjson_get_str(blob_value);
    const auto blob_path = path.parent_path() / luisa::filesystem::path{blob_name};
    const auto chunk_records = yyjson_arr_size(documents_value);
    bool blob_ok = false;
    auto blob = read_binary_file(blob_path, blob_ok);
    if (!blob_ok) {
        yyjson_doc_free(doc);
        return make_error(RetrievalErrorCode::operational,
                          luisa::format("cannot read '{}'", _display_path(blob_path)),
                          RetrievalSubCode::io);
    }
    // Overflow-safe size equality: a corrupt manifest can advertise absurd
    // record/dimension counts, and the wrapped product must never pass the
    // check (it would turn the record slicing below into an OOB read).
    const auto mul_overflow = [](size_t a, size_t b) noexcept {
        return b != 0u && a > std::numeric_limits<size_t>::max() / b;
    };
    if (mul_overflow(chunk_records, dimension) ||
        mul_overflow(chunk_records * dimension, sizeof(float)) ||
        blob.size() != chunk_records * dimension * sizeof(float)) {
        yyjson_doc_free(doc);
        return make_error(RetrievalErrorCode::operational,
                          luisa::format("'{}' size does not match the manifest "
                                        "({} records x {} float(s) = {} byte(s))",
                                        _display_path(blob_path), chunk_records, dimension,
                                        chunk_records * dimension * sizeof(float)),
                          RetrievalSubCode::io);
    }

    // Optional ColBERT sidecar (present when the snapshot was written from a
    // corpus that stored matrices). Matrices are restored only when this
    // index opted into ColBERT storage — the create flag's RAM bound applies
    // to snapshots too; otherwise the sidecar is left on disk and the
    // restored corpus has no matrices, so want_colbert queries degrade to
    // the fused score per record (see query()).
    luisa::string colbert_blob;
    bool has_colbert_blob = false;
    size_t colbert_floats = 0u;
    if (_want_colbert) {
        const auto *colbert_value = yyjson_obj_get(root, "colbert_blob");
        if (yyjson_is_str(colbert_value)) {
            const auto colbert_path = path.parent_path() /
                                      luisa::filesystem::path{luisa::string{yyjson_get_str(colbert_value)}};
            bool colbert_ok = false;
            colbert_blob = read_binary_file(colbert_path, colbert_ok);
            if (!colbert_ok) {
                yyjson_doc_free(doc);
                return make_error(RetrievalErrorCode::operational,
                                  luisa::format("cannot read '{}'", _display_path(colbert_path)),
                                  RetrievalSubCode::io);
            }
            // First pass: total matrix rows across records (overflow-guarded).
            {
                size_t i0 = 0u;
                const yyjson_val *entry0 = nullptr;
                size_t imax0 = 0u;
                yyjson_arr_foreach(documents_value, i0, imax0, entry0) {
                    if (!yyjson_is_obj(entry0)) { continue; }
                    const auto *rows = yyjson_obj_get(entry0, "colbert_rows");
                    if (!yyjson_is_uint(rows)) { continue; }
                    const auto r = static_cast<size_t>(yyjson_get_uint(rows));
                    if (colbert_floats > std::numeric_limits<size_t>::max() - r) {
                        yyjson_doc_free(doc);
                        return make_error(RetrievalErrorCode::operational,
                                          "snapshot ColBERT row counts overflow",
                                          RetrievalSubCode::index);
                    }
                    colbert_floats += r;
                }
            }
            if (mul_overflow(colbert_floats, dimension) ||
                mul_overflow(colbert_floats * dimension, sizeof(float)) ||
                colbert_blob.size() != colbert_floats * dimension * sizeof(float)) {
                yyjson_doc_free(doc);
                return make_error(RetrievalErrorCode::operational,
                                  luisa::format("'{}' size does not match the manifest ColBERT rows "
                                                "({} rows x {} float(s))",
                                                _display_path(colbert_path), colbert_floats, dimension),
                                  RetrievalSubCode::io);
            }
            has_colbert_blob = true;
        }
    }

    // Build into temporaries; the index is swapped only on full success.
    luisa::unordered_map<luisa::string, RootRecord> roots;
    const auto *roots_value = yyjson_obj_get(root, "roots");
    if (yyjson_is_arr(roots_value)) {
        size_t i = 0u;
        const yyjson_val *entry = nullptr;
        size_t imax = 0u;
        yyjson_arr_foreach(roots_value, i, imax, entry) {
            if (!yyjson_is_obj(entry)) { continue; }
            const auto *key = yyjson_obj_get(entry, "key");
            const auto *path_value = yyjson_obj_get(entry, "path");
            if (!yyjson_is_str(key) || !yyjson_is_str(path_value)) { continue; }
            RootRecord record;
            record.path = luisa::filesystem::path{luisa::string{yyjson_get_str(path_value)}};
            record.comparison_path = record.path;
            const auto *extensions = yyjson_obj_get(entry, "extensions");
            if (yyjson_is_arr(extensions)) {
                size_t j = 0u;
                const yyjson_val *extension = nullptr;
                size_t jmax = 0u;
                yyjson_arr_foreach(extensions, j, jmax, extension) {
                    if (yyjson_is_str(extension)) {
                        record.extensions.emplace_back(yyjson_get_str(extension));
                    }
                }
            }
            if (const auto *v = yyjson_obj_get(entry, "chunk_size"); yyjson_is_uint(v)) {
                record.chunk_options.chunk_size = static_cast<uint32_t>(yyjson_get_uint(v));
            }
            if (const auto *v = yyjson_obj_get(entry, "chunk_overlap"); yyjson_is_uint(v)) {
                record.chunk_options.chunk_overlap = static_cast<uint32_t>(yyjson_get_uint(v));
            }
            if (const auto *v = yyjson_obj_get(entry, "chunk_by"); yyjson_is_uint(v)) {
                record.chunk_options.by = static_cast<ChunkOptions::By>(yyjson_get_uint(v));
            }
            if (const auto *v = yyjson_obj_get(entry, "chunk_max_seq_len"); yyjson_is_uint(v)) {
                record.chunk_options.max_seq_len = static_cast<uint32_t>(yyjson_get_uint(v));
            }
            roots.emplace(luisa::string{yyjson_get_str(key)}, std::move(record));
        }
    }

    luisa::unordered_map<luisa::string, DocumentRecord> documents;
    luisa::unordered_map<luisa::string, luisa::vector<luisa::string>> doc_chunks;
    // First pass: chunk count per doc_id (decides whole-file vs chunked keys).
    luisa::unordered_map<luisa::string, size_t> chunks_per_doc;
    {
        size_t i = 0u;
        const yyjson_val *entry = nullptr;
        size_t imax0 = 0u;
        yyjson_arr_foreach(documents_value, i, imax0, entry) {
            const auto *doc_id = yyjson_obj_get(entry, "doc_id");
            if (yyjson_is_str(doc_id)) {
                chunks_per_doc[luisa::string{yyjson_get_str(doc_id)}]++;
            }
        }
    }
    size_t record_index = 0u;
    size_t colbert_index = 0u;// running float offset into colbert_blob
    size_t i = 0u;
    size_t imax = 0u;
    const yyjson_val *entry = nullptr;
    yyjson_arr_foreach(documents_value, i, imax, entry) {
        const auto *doc_id_value = yyjson_obj_get(entry, "doc_id");
        if (!yyjson_is_str(doc_id_value)) {
            return fail_manifest("snapshot document misses its doc_id");
        }
        DocumentRecord record;
        record.doc_id = yyjson_get_str(doc_id_value);
        if (const auto *v = yyjson_obj_get(entry, "chunk_index"); yyjson_is_uint(v)) {
            record.chunk_index = static_cast<uint32_t>(yyjson_get_uint(v));
        }
        if (const auto *v = yyjson_obj_get(entry, "byte_offset"); yyjson_is_uint(v)) {
            record.byte_offset = yyjson_get_uint(v);
        }
        if (const auto *v = yyjson_obj_get(entry, "byte_length"); yyjson_is_uint(v)) {
            record.byte_length = yyjson_get_uint(v);
        }
        if (const auto *v = yyjson_obj_get(entry, "metadata"); yyjson_is_str(v)) {
            record.metadata_json = yyjson_get_str(v);
        }
        if (const auto *v = yyjson_obj_get(entry, "root_key"); yyjson_is_str(v)) {
            record.root_key = yyjson_get_str(v);
        }
        if (const auto *v = yyjson_obj_get(entry, "text"); yyjson_is_str(v)) {
            record.text = yyjson_get_str(v);// memory-only document
        }
        if (const auto *v = yyjson_obj_get(entry, "path"); yyjson_is_str(v)) {
            record.display_path = yyjson_get_str(v);
            record.path = luisa::filesystem::path{record.display_path};
            record.comparison_path = record.path;
        }
        const auto *dense = reinterpret_cast<const float *>(
            blob.data() + record_index * dimension * sizeof(float));
        record.embedding.dense.assign(dense, dense + dimension);
        for (const auto value : record.embedding.dense) {
            if (!std::isfinite(value)) {
                return fail_manifest("snapshot blob contains a non-finite dense embedding");
            }
        }
        const auto *sparse_value = yyjson_obj_get(entry, "sparse");
        if (_hybrid) {
            if (!yyjson_is_arr(sparse_value)) {
                return fail_manifest("hybrid snapshot document misses its sparse pairs");
            }
            luisa::vector<std::pair<uint32_t, float>> sparse;
            size_t j = 0u;
            const yyjson_val *pair = nullptr;
            size_t jmax = 0u;
            yyjson_arr_foreach(sparse_value, j, jmax, pair) {
                if (!yyjson_is_arr(pair) || yyjson_arr_size(pair) != 2u) {
                    return fail_manifest("snapshot sparse entry is malformed");
                }
                sparse.emplace_back(
                    static_cast<uint32_t>(yyjson_get_uint(yyjson_arr_get(pair, 0u))),
                    static_cast<float>(yyjson_get_num(yyjson_arr_get(pair, 1u))));
            }
            auto status = canonicalize_sparse(
                luisa::span<const std::pair<uint32_t, float>>{sparse.data(), sparse.size()},
                record.embedding.sparse);
            if (!status.ok()) {
                return fail_manifest(luisa::format("snapshot sparse entry is invalid: {}", status.message));
            }
        }
        if (has_colbert_blob) {
            if (const auto *v = yyjson_obj_get(entry, "colbert_rows"); yyjson_is_uint(v)) {
                const auto rows = static_cast<size_t>(yyjson_get_uint(v));
                if (rows > 0u) {
                    // The global float count was validated against the blob
                    // size above, so per-record slices stay in bounds.
                    const auto floats = rows * dimension;
                    const float *matrix = reinterpret_cast<const float *>(
                        colbert_blob.data() + colbert_index * dimension * sizeof(float));
                    record.embedding.colbert.assign(matrix, matrix + floats);
                    colbert_index += floats;
                    for (const auto value : record.embedding.colbert) {
                        if (!std::isfinite(value)) {
                            return fail_manifest("snapshot ColBERT blob contains a non-finite value");
                        }
                    }
                }
            }
        }
        const bool chunked = chunks_per_doc[record.doc_id] > 1u;
        const luisa::string base_key = record.path.empty() ?
                                           luisa::format("doc\x1f{}", record.doc_id) :
                                           _path_key(record.comparison_path);
        luisa::string key = chunked ? _chunk_key(base_key, record.chunk_index) : base_key;
        doc_chunks[record.doc_id].push_back(key);
        documents.emplace(std::move(key), std::move(record));
        record_index++;
    }
    yyjson_doc_free(doc);

    _roots = std::move(roots);
    _documents = std::move(documents);
    _doc_chunks = std::move(doc_chunks);
    _dimension = dimension;
    return {};
}

}// namespace luisa::embed
