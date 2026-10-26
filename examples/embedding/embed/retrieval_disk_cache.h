// On-disk embedding cache for the interactive retrieval example: directory
// import persists encoded results so unchanged files skip the encoder on the
// next load. Per indexed directory it maintains
//   * .lc_embed_cache.json   -- directory meta (yyjson): per-file content hash,
//                              size, mtime, and embedding dimension;
//   * .lc_embed_cache/       -- one binary cache file per encoded result, named
//                              by the MD5 of the file content.
// Cache hits are loaded incrementally (one DStorage read per file) through the
// backend DStorageExt when it is available; otherwise a plain host read is
// used. A null device selects the host path (tests, host-only frontends).

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <system_error>

#include <yyjson.h>

#include <luisa/backends/ext/dstorage_ext.hpp>
#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/vstl/md5.h>

#include "retrieval_index.h"

namespace luisa::embed {

class DiskEmbedCache final : public RetrievalEmbedCache {

public:
    // `device` may be null: the cache then always falls back to host reads.
    explicit DiskEmbedCache(compute::Device *device) noexcept
        : _device{device} {}
    ~DiskEmbedCache() noexcept override = default;

    DiskEmbedCache(const DiskEmbedCache &) = delete;
    DiskEmbedCache(DiskEmbedCache &&) = delete;
    DiskEmbedCache &operator=(const DiskEmbedCache &) = delete;
    DiskEmbedCache &operator=(DiskEmbedCache &&) = delete;

    void begin_directory(
        const luisa::filesystem::path &root,
        luisa::span<const luisa::string> extensions) noexcept override {
        _root = root;
        _extensions.assign(extensions.begin(), extensions.end());
        _meta.clear();
        _meta_dirty = false;
        _read_meta();
        std::error_code ec;
        luisa::filesystem::create_directories(_cache_dir(), ec);
        if (ec) {
            LUISA_WARNING("embed disk cache: cannot create '{}': {}",
                          _display(_cache_dir()), ec.message());
        }
    }

    [[nodiscard]] bool load(
        const luisa::filesystem::path &file,
        luisa::string_view contents,
        luisa::vector<float> &embedding) noexcept override {
        const auto key = _path_key(file);
        const auto entry = _meta.find(key);
        if (entry == _meta.end()) {
            return false;
        }
        const auto digest = _hash(contents);
        if (entry->second.md5 != digest || entry->second.dimension == 0u) {
            return false;
        }
        const auto expected_bytes =
            static_cast<size_t>(entry->second.dimension) * sizeof(float);
        std::error_code ec;
        const auto actual_bytes = luisa::filesystem::file_size(
            _cache_file(digest), ec);
        if (ec || actual_bytes != expected_bytes) {
            return false;
        }
        luisa::vector<std::byte> bytes;
        if (!_read_binary(_cache_file(digest), bytes) ||
            bytes.size() != expected_bytes) {
            return false;
        }
        const auto *values = reinterpret_cast<const float *>(bytes.data());
        embedding.assign(values, values + entry->second.dimension);
        LUISA_VERBOSE("embed disk cache hit: {} ({} float(s))",
                      key, entry->second.dimension);
        return true;
    }

    void store(
        const luisa::filesystem::path &file,
        luisa::string_view contents,
        luisa::span<const float> embedding) noexcept override {
        if (embedding.empty()) {
            return;
        }
        const auto key = _path_key(file);
        const auto digest = _hash(contents);
        const auto path = _cache_file(digest);
        std::ofstream stream{path, std::ios::binary | std::ios::trunc};
        if (!stream.is_open()) {
            LUISA_WARNING("embed disk cache: cannot write '{}'", _display(path));
            return;
        }
        stream.write(reinterpret_cast<const char *>(embedding.data()),
                     static_cast<std::streamsize>(embedding.size_bytes()));
        stream.close();
        if (!stream) {
            LUISA_WARNING("embed disk cache: cannot write '{}'", _display(path));
            return;
        }
        FileMeta meta;
        meta.md5 = digest;
        meta.size = contents.size();
        meta.dimension = embedding.size();
        std::error_code ec;
        meta.mtime = luisa::filesystem::last_write_time(file, ec)
                         .time_since_epoch()
                         .count();
        if (ec) {
            meta.mtime = 0;
        }
        _meta[key] = std::move(meta);
        _dimension = embedding.size();
        _meta_dirty = true;
    }

    void end_directory() noexcept override {
        if (!_meta_dirty) {
            return;
        }
        yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
        if (doc == nullptr) {
            return;
        }
        auto *root = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, root);
        yyjson_mut_obj_add(root, yyjson_mut_str(doc, "version"), yyjson_mut_uint(doc, 1u));
        yyjson_mut_obj_add(root, yyjson_mut_str(doc, "dimension"), yyjson_mut_uint(doc, _dimension));
        auto *extensions = yyjson_mut_arr(doc);
        yyjson_mut_obj_add(root, yyjson_mut_str(doc, "extensions"), extensions);
        for (const auto &extension : _extensions) {
            yyjson_mut_arr_add_strcpy(doc, extensions, extension.c_str());
        }
        auto *files = yyjson_mut_arr(doc);
        yyjson_mut_obj_add(root, yyjson_mut_str(doc, "files"), files);
        luisa::vector<const luisa::string *> keys;
        keys.reserve(_meta.size());
        for (const auto &entry : _meta) {
            keys.push_back(&entry.first);
        }
        luisa::sort(keys.begin(), keys.end(), [](const auto *a, const auto *b) {
            return *a < *b;
        });
        for (const auto *key : keys) {
            const auto &meta = _meta.find(*key)->second;
            auto *object = yyjson_mut_obj(doc);
            yyjson_mut_arr_append(files, object);
            yyjson_mut_obj_add(object, yyjson_mut_str(doc, "path"), yyjson_mut_strcpy(doc, key->c_str()));
            yyjson_mut_obj_add(object, yyjson_mut_str(doc, "size"), yyjson_mut_uint(doc, meta.size));
            yyjson_mut_obj_add(object, yyjson_mut_str(doc, "mtime"), yyjson_mut_sint(doc, meta.mtime));
            yyjson_mut_obj_add(object, yyjson_mut_str(doc, "md5"), yyjson_mut_strcpy(doc, meta.md5.c_str()));
            yyjson_mut_obj_add(object, yyjson_mut_str(doc, "dimension"), yyjson_mut_uint(doc, meta.dimension));
        }
        size_t length = 0u;
        char *json = yyjson_mut_write(
            doc, YYJSON_WRITE_PRETTY | YYJSON_WRITE_NEWLINE_AT_END, &length);
        yyjson_mut_doc_free(doc);
        if (json == nullptr) {
            LUISA_WARNING("embed disk cache: cannot serialize meta for '{}'",
                          _display(_root));
            return;
        }
        const auto path = _meta_path();
        std::ofstream stream{path, std::ios::binary | std::ios::trunc};
        if (!stream.is_open()) {
            LUISA_WARNING("embed disk cache: cannot write '{}'", _display(path));
        } else {
            stream.write(json, static_cast<std::streamsize>(length));
        }
        std::free(json);
        _meta_dirty = false;
    }

private:
    struct FileMeta {
        luisa::string md5;
        size_t size{};
        int64_t mtime{};
        size_t dimension{};
    };

    [[nodiscard]] luisa::filesystem::path _cache_dir() const noexcept {
        return _root / ".lc_embed_cache";
    }
    [[nodiscard]] luisa::filesystem::path _meta_path() const noexcept {
        return _root / ".lc_embed_cache.json";
    }
    [[nodiscard]] luisa::filesystem::path _cache_file(const luisa::string &digest) const noexcept {
        auto name = digest;
        name += ".bin";
        return _cache_dir() / luisa::filesystem::path{name};
    }

    [[nodiscard]] static luisa::string _path_key(const luisa::filesystem::path &path) {
        const auto utf8 = path.generic_u8string();
        return {reinterpret_cast<const char *>(utf8.data()), utf8.size()};
    }

    [[nodiscard]] static luisa::string _display(const luisa::filesystem::path &path) {
        return _path_key(path);
    }

    [[nodiscard]] static luisa::string _hash(luisa::string_view contents) noexcept {
        const auto bytes = vstd::GetMD5FromArray(luisa::span<const uint8_t>{
            reinterpret_cast<const uint8_t *>(contents.data()), contents.size()});
        luisa::string digest;
        digest.reserve(bytes.size() * 2u);
        static constexpr char kHex[] = "0123456789abcdef";
        for (const auto byte : bytes) {
            digest.push_back(kHex[byte >> 4u]);
            digest.push_back(kHex[byte & 0x0fu]);
        }
        return digest;
    }

    // Lazily probes the backend DStorageExt; null when unavailable or when no
    // device was supplied, in which case reads use the host fallback.
    [[nodiscard]] compute::DStorageExt *_dstorage_ext() noexcept {
        if (!_dstorage_probed) {
            _dstorage_probed = true;
            if (_device != nullptr) {
                _dstorage = _device->extension<compute::DStorageExt>();
                if (_dstorage != nullptr) {
                    _dstream = _dstorage->create_stream();
                    if (!_dstream) {
                        _dstorage = nullptr;
                    }
                }
            }
            if (_dstorage != nullptr) {
                LUISA_INFO("embed disk cache: DStorageExt available on backend '{}'",
                           _device->backend_name());
            }
        }
        return _dstorage;
    }

    [[nodiscard]] bool _read_binary(const luisa::filesystem::path &path,
                                    luisa::vector<std::byte> &data) noexcept {
        if (auto *ext = _dstorage_ext()) {
            const auto utf8 = path.generic_u8string();
            const luisa::string_view utf8_view{
                reinterpret_cast<const char *>(utf8.data()), utf8.size()};
            auto file = ext->open_file(utf8_view);
            if (file) {
                data.resize(file.size_bytes());
                if (!data.empty()) {
                    _dstream << file.copy_to(data.data(), data.size())
                             << compute::synchronize();
                }
                return true;
            }
        }
        std::ifstream stream{path, std::ios::binary | std::ios::ate};
        if (!stream.is_open()) {
            return false;
        }
        const auto end = stream.tellg();
        if (end < 0) {
            return false;
        }
        data.resize(static_cast<size_t>(end));
        stream.seekg(0, std::ios::beg);
        if (!data.empty()) {
            stream.read(reinterpret_cast<char *>(data.data()),
                        static_cast<std::streamsize>(data.size()));
            if (!stream) {
                return false;
            }
        }
        return true;
    }

    void _read_meta() noexcept {
        const auto path = _meta_path();
        std::error_code ec;
        if (!luisa::filesystem::is_regular_file(path, ec) || ec) {
            return;
        }
        std::ifstream stream{path, std::ios::binary | std::ios::ate};
        if (!stream.is_open()) {
            return;
        }
        const auto end = stream.tellg();
        if (end <= 0) {
            return;
        }
        luisa::string text;
        text.resize(static_cast<size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(text.data(), static_cast<std::streamsize>(text.size()));
        if (!stream) {
            return;
        }
        yyjson_doc *doc = yyjson_read(text.data(), text.size(), 0);
        if (doc == nullptr) {
            LUISA_WARNING("embed disk cache: ignoring malformed meta '{}'",
                          _display(path));
            return;
        }
        auto *root = yyjson_doc_get_root(doc);
        auto *files = yyjson_obj_get(root, "files");
        if (yyjson_is_arr(files)) {
            yyjson_val *object;
            size_t index, max;
            yyjson_arr_foreach(files, index, max, object) {
                auto *path_value = yyjson_obj_get(object, "path");
                auto *md5_value = yyjson_obj_get(object, "md5");
                if (!yyjson_is_str(path_value) || !yyjson_is_str(md5_value)) {
                    continue;
                }
                FileMeta meta;
                meta.md5 = yyjson_get_str(md5_value);
                meta.size = yyjson_get_uint(yyjson_obj_get(object, "size"));
                meta.mtime = yyjson_get_sint(yyjson_obj_get(object, "mtime"));
                meta.dimension = yyjson_get_uint(yyjson_obj_get(object, "dimension"));
                if (meta.dimension != 0u) {
                    _meta.emplace(yyjson_get_str(path_value), std::move(meta));
                }
            }
        }
        yyjson_doc_free(doc);
    }

    compute::Device *_device;
    compute::DStorageExt *_dstorage{nullptr};
    bool _dstorage_probed{false};
    compute::Stream _dstream{};
    luisa::filesystem::path _root;
    luisa::vector<luisa::string> _extensions;
    luisa::unordered_map<luisa::string, FileMeta> _meta;
    size_t _dimension{};
    bool _meta_dirty{false};
};

}// namespace luisa::embed
