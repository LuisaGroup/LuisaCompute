// Host-side retrieval index implementation (retrieval plan phase 1): canonical
// path ownership, deterministic discovery/ranking, and transactional incremental
// updates with per-file failure isolation.

#include "retrieval_index.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <system_error>

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

[[nodiscard]] RetrievalStatus make_error(RetrievalErrorCode code, luisa::string message) {
    return RetrievalStatus{code, std::move(message)};
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

}// namespace

RetrievalIndex::RetrievalIndex(RetrievalEncodeCallback encode) noexcept
    : _encode_callback{std::move(encode)} {}

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
                          luisa::format("cannot open '{}'", _display_path(path)));
    }
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    if (end < 0 || static_cast<uint64_t>(end) > std::numeric_limits<size_t>::max() ||
        static_cast<uint64_t>(end) > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        return make_error(RetrievalErrorCode::operational,
                          luisa::format("cannot determine size of '{}'", _display_path(path)));
    }
    luisa::string contents;
    contents.resize(static_cast<size_t>(end));
    stream.seekg(0, std::ios::beg);
    if (!contents.empty()) {
        stream.read(contents.data(), static_cast<std::streamsize>(contents.size()));
        if (!stream) {
            return make_error(RetrievalErrorCode::operational,
                              luisa::format("cannot read '{}'", _display_path(path)));
        }
    }
    if (!valid_utf8(contents)) {
        return make_error(RetrievalErrorCode::operational,
                          luisa::format("'{}' is not valid UTF-8", _display_path(path)));
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

RetrievalStatus RetrievalIndex::_encode(
    luisa::string_view text,
    luisa::vector<float> &embedding) const {
    if (!_encode_callback) {
        return make_error(RetrievalErrorCode::operational,
                          "no embedding encoder is configured");
    }
    auto candidate = _encode_callback(text);
    if (candidate.empty()) {
        return make_error(RetrievalErrorCode::operational,
                          "encoder returned an empty dense embedding");
    }
    if (_dimension != 0u && candidate.size() != _dimension) {
        return make_error(RetrievalErrorCode::operational,
                          luisa::format("embedding dimension {} does not match index dimension {}",
                                        candidate.size(), _dimension));
    }
    for (const auto value : candidate) {
        if (!std::isfinite(value)) {
            return make_error(RetrievalErrorCode::operational,
                              "encoder returned a non-finite dense embedding");
        }
    }
    embedding = std::move(candidate);
    return {};
}

RetrievalStatus RetrievalIndex::_apply_file(
    const luisa::string &root_key,
    const luisa::filesystem::path &file,
    FileOutcome &outcome,
    bool &cache_hit) {
    cache_hit = false;
    luisa::filesystem::path display;
    luisa::filesystem::path comparison;
    luisa::string key;
    auto status = _normalize_path(file, true, display, comparison, key);
    if (!status.ok()) {
        return status;
    }
    luisa::string contents;
    status = _read_text_file(display, contents);
    if (!status.ok()) {
        return status;
    }
    auto existing = _documents.find(key);
    if (existing != _documents.end() && existing->second.text == contents) {
        outcome = FileOutcome::unchanged;
        return {};
    }
    luisa::vector<float> embedding;
    if (_embed_cache != nullptr && _embed_cache->load(display, contents, embedding) &&
        (_dimension == 0u || embedding.size() == _dimension) &&
        std::ranges::all_of(embedding, [](const float v) { return std::isfinite(v); })) {
        // Served from the disk cache: the encoder stays untouched.
        cache_hit = true;
    } else {
        embedding.clear();
        status = _encode(contents, embedding);
        if (!status.ok()) {
            return status;
        }
        if (_embed_cache != nullptr) {
            _embed_cache->store(display, contents, embedding);
        }
    }
    DocumentRecord document;
    document.path = display;
    document.comparison_path = comparison;
    document.root_key = root_key;
    document.display_path = _display_path(display);
    document.text = std::move(contents);
    document.embedding = std::move(embedding);
    if (_dimension == 0u) {
        _dimension = document.embedding.size();
    }
    if (existing == _documents.end()) {
        _documents.emplace(key, std::move(document));
        outcome = FileOutcome::added;
    } else {
        existing->second = std::move(document);
        outcome = FileOutcome::updated;
    }
    return {};
}

RetrievalLoadResult RetrievalIndex::load(const luisa::filesystem::path &root) {
    const auto extensions = default_extensions();
    return load(root, luisa::span<const luisa::string>{extensions.data(), extensions.size()});
}

RetrievalLoadResult RetrievalIndex::load(
    const luisa::filesystem::path &root,
    luisa::span<const luisa::string> extensions) {
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
    for (const auto &file : discovery.files) {
        FileOutcome outcome = FileOutcome::added;
        bool cache_hit = false;
        status = _apply_file(key, file, outcome, cache_hit);
        if (!status.ok()) {
            result.failed++;
            LUISA_WARNING("retrieval load skipped '{}': {}", _display_path(file), status.message);
            continue;
        }
        if (outcome == FileOutcome::added) {
            result.added++;
        }
        if (cache_hit) {
            result.cached++;
        }
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
                                       luisa::format("cannot inspect '{}': {}", result.path, ec.message()));
        }
        result.total_documents = _documents.size();
        return result;
    }
    if (!regular_file || file_not_found(ec)) {
        if (existing != _documents.end()) {
            result.removed = _documents.erase(comparison_key);
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
    auto status = _apply_file(root_key, display_path, outcome, cache_hit);
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
        auto status = _apply_file(root_key, file, outcome, cache_hit);
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

RetrievalQueryResult RetrievalIndex::query(luisa::string_view text, uint32_t top_k) {
    RetrievalQueryResult result;
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
    if (_documents.empty()) {
        LUISA_WARNING("retrieval query rejected: the document corpus is empty");
        result.status = make_error(RetrievalErrorCode::operational,
                                   "cannot query an empty document corpus");
        return result;
    }
    luisa::vector<float> query_embedding;
    auto status = _encode(text, query_embedding);
    if (!status.ok()) {
        result.status = std::move(status);
        return result;
    }
    result.hits.reserve(_documents.size());
    for (const auto &entry : _documents) {
        const auto &document = entry.second;
        if (document.embedding.size() != query_embedding.size()) {
            result.status = make_error(RetrievalErrorCode::operational,
                                       "index contains inconsistent embedding dimensions");
            result.hits.clear();
            return result;
        }
        double score = 0.0;
        for (size_t i = 0u; i < query_embedding.size(); i++) {
            score += static_cast<double>(query_embedding[i]) *
                     static_cast<double>(document.embedding[i]);
        }
        result.hits.push_back(RetrievalHit{document.display_path, score});
    }
    luisa::sort(result.hits.begin(), result.hits.end(), [](const auto &a, const auto &b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.path < b.path;
    });
    if (result.hits.size() > top_k) {
        result.hits.resize(top_k);
    }
    return result;
}

}// namespace luisa::embed
