// OCR engine -- .lcocr v1 package reader/writer (Phase 1).
// Strict fail-closed parsing; payloads are read on demand. No exceptions.
#include "ocr/package.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "ocr/model_recipe.h"

#include <luisa/core/stl/unordered_map.h>
#include <yyjson.h>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace luisa::ocr {
namespace {

constexpr uint64_t kMagic = 0x00003152434f434cull;// "LCOCR1\0\0" little-endian

// ---------------------------------------------------------------------------
// Little-endian byte cursor (bounds-checked)
// ---------------------------------------------------------------------------

class ByteReader {
public:
    ByteReader(luisa::span<const std::byte> bytes, uint64_t base_offset) noexcept
        : _bytes{bytes}, _base{base_offset} {}

    [[nodiscard]] bool read_u32(uint32_t &out) noexcept {
        if (_pos + 4u > _bytes.size()) { return false; }
        uint32_t v = 0;
        for (uint32_t i = 0; i < 4u; i++) {
            v |= static_cast<uint32_t>(_bytes[_pos + i]) << (8u * i);
        }
        _pos += 4u;
        out = v;
        return true;
    }
    [[nodiscard]] bool read_u64(uint64_t &out) noexcept {
        if (_pos + 8u > _bytes.size()) { return false; }
        uint64_t v = 0;
        for (uint32_t i = 0; i < 8u; i++) {
            v |= static_cast<uint64_t>(_bytes[_pos + i]) << (8u * i);
        }
        _pos += 8u;
        out = v;
        return true;
    }
    [[nodiscard]] bool read_bytes(luisa::span<std::byte> out) noexcept {
        if (_pos + out.size() > _bytes.size()) { return false; }
        std::memcpy(out.data(), _bytes.data() + _pos, out.size());
        _pos += out.size();
        return true;
    }
    [[nodiscard]] uint64_t absolute_pos() const noexcept { return _base + _pos; }
    [[nodiscard]] bool at_end() const noexcept { return _pos == _bytes.size(); }

private:
    luisa::span<const std::byte> _bytes;
    uint64_t _base;
    size_t _pos{0};
};

void append_u32(luisa::vector<std::byte> &out, uint32_t v) {
    for (uint32_t i = 0; i < 4u; i++) {
        out.emplace_back(static_cast<std::byte>((v >> (8u * i)) & 0xffu));
    }
}
void append_u64(luisa::vector<std::byte> &out, uint64_t v) {
    for (uint32_t i = 0; i < 8u; i++) {
        out.emplace_back(static_cast<std::byte>((v >> (8u * i)) & 0xffu));
    }
}

// ---------------------------------------------------------------------------
// UTF-8 validation (rejects overlong forms, surrogates, > U+10FFFF)
// ---------------------------------------------------------------------------

[[nodiscard]] bool is_valid_utf8(luisa::string_view text) noexcept {
    size_t i = 0;
    while (i < text.size()) {
        const auto b0 = static_cast<uint8_t>(text[i]);
        uint32_t cp = 0;
        uint32_t extra = 0;
        if (b0 < 0x80u) {
            i += 1;
            continue;
        } else if ((b0 & 0xe0u) == 0xc0u) {
            cp = b0 & 0x1fu;
            extra = 1;
            if (cp < 2) { return false; }// overlong
        } else if ((b0 & 0xf0u) == 0xe0u) {
            cp = b0 & 0x0fu;
            extra = 2;
        } else if ((b0 & 0xf8u) == 0xf0u) {
            cp = b0 & 0x07u;
            extra = 3;
            if (cp > 4) { return false; }
        } else {
            return false;
        }
        if (i + extra >= text.size()) { return false; }
        for (uint32_t k = 1; k <= extra; k++) {
            const auto bk = static_cast<uint8_t>(text[i + k]);
            if ((bk & 0xc0u) != 0x80u) { return false; }
            cp = (cp << 6u) | (bk & 0x3fu);
        }
        if (extra == 2 && cp < 0x800u) { return false; }
        if (extra == 3 && cp < 0x10000u) { return false; }
        if (cp > 0x10ffffu || (cp >= 0xd800u && cp <= 0xdfffu)) { return false; }
        i += extra + 1;
    }
    return true;
}

[[nodiscard]] Status invalid_utf8_error(luisa::string_view what) {
    return Status::error(StatusCode::invalid_argument,
                         luisa::string{what} + " is not valid UTF-8", luisa::string{what});
}

// ---------------------------------------------------------------------------
// Manifest JSON (yyjson). Dynamic strings are copied into luisa containers
// before the document is freed; the writer borrows only string literals for
// keys and copies every dynamic value.
// ---------------------------------------------------------------------------

struct JsonDoc {
    yyjson_doc *doc{nullptr};
    ~JsonDoc() noexcept {
        if (doc != nullptr) { yyjson_doc_free(doc); }
    }
};

[[nodiscard]] Result<luisa::string> json_get_string(yyjson_val *obj, const char *key, bool required) {
    auto *v = yyjson_obj_get(obj, key);
    if (v == nullptr || yyjson_is_null(v)) {
        if (required) {
            return Result<luisa::string>::failure(StatusCode::invalid_argument,
                                                  luisa::string{"manifest is missing required field '"} + key + "'", key);
        }
        return Result<luisa::string>::success(luisa::string{});
    }
    if (!yyjson_is_str(v)) {
        return Result<luisa::string>::failure(StatusCode::invalid_argument,
                                              luisa::string{"manifest field '"} + key + "' must be a string", key);
    }
    return Result<luisa::string>::success(luisa::string{yyjson_get_str(v), yyjson_get_len(v)});
}

[[nodiscard]] bool manifest_key_known(luisa::string_view key) noexcept {
    static constexpr luisa::string_view known[] = {
        "format_version", "model", "recipe_version", "checkpoint_revision",
        "config_fingerprint", "license", "producer", "producer_version",
        "preprocessing_version", "tokenizer_version", "output_parser_version",
        "component", "dependencies", "source_hashes"};
    for (auto k : known) {
        if (k == key) { return true; }
    }
    return false;
}

// ---------------------------------------------------------------------------
// File helpers (stdio; no exceptions)
// ---------------------------------------------------------------------------

[[nodiscard]] Result<uint64_t> file_size_of(std::FILE *file) {
#ifdef _WIN32
    const int fd = _fileno(file);
    const auto size = _filelengthi64(fd);
    if (size < 0) {
        return Result<uint64_t>::failure(StatusCode::io_error, "cannot query file size", "file");
    }
    return Result<uint64_t>::success(static_cast<uint64_t>(size));
#else
    const int fd = fileno(file);
    const auto size = lseek(fd, 0, SEEK_END);
    if (size < 0 || lseek(fd, 0, SEEK_SET) < 0) {
        return Result<uint64_t>::failure(StatusCode::io_error, "cannot query file size", "file");
    }
    return Result<uint64_t>::success(static_cast<uint64_t>(size));
#endif
}

[[nodiscard]] Status seek_to(std::FILE *file, uint64_t offset) {
#ifdef _WIN32
    const int rc = _fseeki64(file, static_cast<int64_t>(offset), SEEK_SET);
#else
    const int rc = fseeko(file, static_cast<off_t>(offset), SEEK_SET);
#endif
    if (rc != 0) {
        return Status::error(StatusCode::io_error, "cannot seek in package file", "file");
    }
    return Status::success();
}

}// namespace

// Cross-module-safe deleters: destruction runs in this TU so the allocator
// that allocated the object also frees it (the static library may be linked
// into a separate module with its own allocator state).
void PackageReaderDeleter::operator()(PackageReader *reader) const noexcept { delete reader; }
void PackageWriterDeleter::operator()(PackageWriter *writer) const noexcept { delete writer; }

// ---------------------------------------------------------------------------
// Encoding / layout names
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::string_view tensor_encoding_name(TensorEncoding encoding) noexcept {
    switch (encoding) {
        case TensorEncoding::fp32: return "fp32";
        case TensorEncoding::fp16: return "fp16";
        case TensorEncoding::bf16: return "bf16";
        case TensorEncoding::int4_packed: return "int4_packed";
        case TensorEncoding::int8: return "int8";
        default: return "unknown";
    }
}

[[nodiscard]] bool is_quantized_encoding(TensorEncoding encoding) noexcept {
    return encoding == TensorEncoding::int4_packed || encoding == TensorEncoding::int8;
}

[[nodiscard]] uint64_t dense_encoding_element_size(TensorEncoding encoding) noexcept {
    switch (encoding) {
        case TensorEncoding::fp32: return 4;
        case TensorEncoding::fp16: return 2;
        case TensorEncoding::bf16: return 2;
        default: return 0;
    }
}

[[nodiscard]] luisa::string_view tensor_layout_name(TensorLayout layout) noexcept {
    switch (layout) {
        case TensorLayout::row_major: return "row_major";
        case TensorLayout::kn: return "kn";
        case TensorLayout::nk: return "nk";
        case TensorLayout::conv_oihw: return "conv_oihw";
        case TensorLayout::conv_ohwi: return "conv_ohwi";
        default: return "unknown";
    }
}

[[nodiscard]] luisa::optional<TensorLayout> tensor_layout_from_name(luisa::string_view name) noexcept {
    if (name == "row_major") { return TensorLayout::row_major; }
    if (name == "kn") { return TensorLayout::kn; }
    if (name == "nk") { return TensorLayout::nk; }
    if (name == "conv_oihw") { return TensorLayout::conv_oihw; }
    if (name == "conv_ohwi") { return TensorLayout::conv_ohwi; }
    return luisa::nullopt;
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

[[nodiscard]] Status PackageManifest::validate() const {
    if (model.empty()) {
        return Status::error(StatusCode::invalid_argument, "manifest.model must not be empty", "model");
    }
    if (model_id_from_name(model.substr(0, model.find(':'))) == ModelId::unknown) {
        return Status::error(StatusCode::invalid_argument,
                             "manifest.model is not a known semantic model target", "model");
    }
    const auto fields = {
        std::pair{luisa::string_view{"recipe_version"}, luisa::string_view{recipe_version}},
        std::pair{luisa::string_view{"checkpoint_revision"}, luisa::string_view{checkpoint_revision}},
        std::pair{luisa::string_view{"config_fingerprint"}, luisa::string_view{config_fingerprint}},
        std::pair{luisa::string_view{"producer"}, luisa::string_view{producer}},
        std::pair{luisa::string_view{"producer_version"}, luisa::string_view{producer_version}},
        std::pair{luisa::string_view{"preprocessing_version"}, luisa::string_view{preprocessing_version}},
        std::pair{luisa::string_view{"tokenizer_version"}, luisa::string_view{tokenizer_version}},
        std::pair{luisa::string_view{"output_parser_version"}, luisa::string_view{output_parser_version}},
    };
    for (const auto &[name, value] : fields) {
        if (value.empty()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::string{"manifest."} + luisa::string{name} + " must not be empty", luisa::string{name});
        }
    }
    for (const auto &dep : dependencies) {
        if (dep.component.empty() || dep.package.empty()) {
            return Status::error(StatusCode::invalid_argument,
                                 "manifest dependency requires component and package", "dependencies");
        }
    }
    for (const auto &hash : source_hashes) {
        if (hash.path.empty() || hash.algorithm.empty() || hash.digest.empty()) {
            return Status::error(StatusCode::invalid_argument,
                                 "manifest source hash requires path, algorithm and digest", "source_hashes");
        }
    }
    return Status::success();
}

[[nodiscard]] Result<luisa::string> serialize_package_manifest(const PackageManifest &manifest) {
    auto valid = manifest.validate();
    if (!valid.ok()) { return Result<luisa::string>::failure(std::move(valid)); }
    yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
    if (doc == nullptr) {
        return Result<luisa::string>::failure(StatusCode::out_of_memory, "cannot allocate manifest JSON document", "manifest");
    }
    auto *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_uint(doc, root, "format_version", kPackageFormatVersion);
    // Keys are literals (borrowed, static storage); every dynamic value is
    // copied into the document.
    const auto add_str = [&](const char *key, const luisa::string &value) {
        yyjson_mut_obj_add_strncpy(doc, root, key, value.data(), value.size());
    };
    add_str("model", manifest.model);
    add_str("recipe_version", manifest.recipe_version);
    add_str("checkpoint_revision", manifest.checkpoint_revision);
    add_str("config_fingerprint", manifest.config_fingerprint);
    add_str("license", manifest.license);
    add_str("producer", manifest.producer);
    add_str("producer_version", manifest.producer_version);
    add_str("preprocessing_version", manifest.preprocessing_version);
    add_str("tokenizer_version", manifest.tokenizer_version);
    add_str("output_parser_version", manifest.output_parser_version);
    add_str("component", manifest.component);
    auto *deps = yyjson_mut_arr(doc);
    for (const auto &dep : manifest.dependencies) {
        auto *item = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_strncpy(doc, item, "component", dep.component.data(), dep.component.size());
        yyjson_mut_obj_add_strncpy(doc, item, "package", dep.package.data(), dep.package.size());
        yyjson_mut_obj_add_strncpy(doc, item, "sha256", dep.sha256.data(), dep.sha256.size());
        yyjson_mut_arr_append(deps, item);
    }
    yyjson_mut_obj_add_val(doc, root, "dependencies", deps);
    auto *hashes = yyjson_mut_arr(doc);
    for (const auto &hash : manifest.source_hashes) {
        auto *item = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_strncpy(doc, item, "path", hash.path.data(), hash.path.size());
        yyjson_mut_obj_add_strncpy(doc, item, "algorithm", hash.algorithm.data(), hash.algorithm.size());
        yyjson_mut_obj_add_strncpy(doc, item, "digest", hash.digest.data(), hash.digest.size());
        yyjson_mut_arr_append(hashes, item);
    }
    yyjson_mut_obj_add_val(doc, root, "source_hashes", hashes);

    size_t length = 0;
    char *json = yyjson_mut_write(doc, YYJSON_WRITE_NOFLAG, &length);
    yyjson_mut_doc_free(doc);
    if (json == nullptr) {
        return Result<luisa::string>::failure(StatusCode::out_of_memory, "cannot serialize manifest JSON", "manifest");
    }
    luisa::string out{json, length};
    std::free(json);// yyjson default allocator
    return Result<luisa::string>::success(std::move(out));
}

[[nodiscard]] Result<PackageManifest> parse_package_manifest(luisa::string_view json) {
    if (json.size() > kPackageMaxManifestBytes) {
        return Result<PackageManifest>::failure(StatusCode::invalid_argument, "manifest exceeds the size limit", "manifest");
    }
    JsonDoc holder;
    yyjson_read_err err;
    holder.doc = yyjson_read_opts(const_cast<char *>(json.data()), json.size(), YYJSON_READ_NOFLAG, nullptr, &err);
    if (holder.doc == nullptr) {
        return Result<PackageManifest>::failure(StatusCode::invalid_argument,
                                                luisa::string{"manifest JSON parse error: "} + (err.msg != nullptr ? err.msg : "unknown"), "manifest");
    }
    auto *root = yyjson_doc_get_root(holder.doc);
    if (!yyjson_is_obj(root)) {
        return Result<PackageManifest>::failure(StatusCode::invalid_argument, "manifest root must be an object", "manifest");
    }
    // Reject unknown keys and duplicate keys (yyjson keeps duplicates).
    luisa::unordered_map<luisa::string, uint32_t> seen;
    yyjson_val *key;
    yyjson_val *val;
    size_t idx, max;
    yyjson_obj_foreach(root, idx, max, key, val) {
        const luisa::string_view name{yyjson_get_str(key), yyjson_get_len(key)};
        if (!manifest_key_known(name)) {
            return Result<PackageManifest>::failure(StatusCode::invalid_argument,
                                                    luisa::string{"unknown manifest field '"} + luisa::string{name} + "'", luisa::string{name});
        }
        if (++seen[luisa::string{name}] > 1u) {
            return Result<PackageManifest>::failure(StatusCode::invalid_argument,
                                                    luisa::string{"duplicate manifest field '"} + luisa::string{name} + "'", luisa::string{name});
        }
    }
    auto *version = yyjson_obj_get(root, "format_version");
    if (version == nullptr || !yyjson_is_uint(version) || yyjson_get_uint(version) != kPackageFormatVersion) {
        return Result<PackageManifest>::failure(StatusCode::unsupported, "manifest format_version must be 1", "format_version");
    }
    PackageManifest manifest;
    const auto assign = [&](const char *key, luisa::string &dst, bool required) -> Status {
        auto r = json_get_string(root, key, required);
        if (!r.ok()) { return r.status; }
        dst = std::move(*r.value);
        return Status::success();
    };
    auto s = assign("model", manifest.model, true);
    if (!s.ok()) { return Result<PackageManifest>::failure(std::move(s)); }
    s = assign("recipe_version", manifest.recipe_version, true);
    if (!s.ok()) { return Result<PackageManifest>::failure(std::move(s)); }
    s = assign("checkpoint_revision", manifest.checkpoint_revision, true);
    if (!s.ok()) { return Result<PackageManifest>::failure(std::move(s)); }
    s = assign("config_fingerprint", manifest.config_fingerprint, true);
    if (!s.ok()) { return Result<PackageManifest>::failure(std::move(s)); }
    s = assign("producer", manifest.producer, true);
    if (!s.ok()) { return Result<PackageManifest>::failure(std::move(s)); }
    s = assign("producer_version", manifest.producer_version, true);
    if (!s.ok()) { return Result<PackageManifest>::failure(std::move(s)); }
    s = assign("preprocessing_version", manifest.preprocessing_version, true);
    if (!s.ok()) { return Result<PackageManifest>::failure(std::move(s)); }
    s = assign("tokenizer_version", manifest.tokenizer_version, true);
    if (!s.ok()) { return Result<PackageManifest>::failure(std::move(s)); }
    s = assign("output_parser_version", manifest.output_parser_version, true);
    if (!s.ok()) { return Result<PackageManifest>::failure(std::move(s)); }
    s = assign("license", manifest.license, false);
    if (!s.ok()) { return Result<PackageManifest>::failure(std::move(s)); }
    s = assign("component", manifest.component, false);
    if (!s.ok()) { return Result<PackageManifest>::failure(std::move(s)); }

    if (auto *deps = yyjson_obj_get(root, "dependencies"); deps != nullptr && !yyjson_is_null(deps)) {
        if (!yyjson_is_arr(deps)) {
            return Result<PackageManifest>::failure(StatusCode::invalid_argument, "manifest.dependencies must be an array", "dependencies");
        }
        yyjson_val *item;
        size_t idx, max;
        yyjson_arr_foreach(deps, idx, max, item) {
            if (!yyjson_is_obj(item)) {
                return Result<PackageManifest>::failure(StatusCode::invalid_argument, "dependency entry must be an object", "dependencies");
            }
            ComponentDependency dep;
            auto c = json_get_string(item, "component", true);
            if (!c.ok()) { return Result<PackageManifest>::failure(c.status); }
            auto p = json_get_string(item, "package", true);
            if (!p.ok()) { return Result<PackageManifest>::failure(p.status); }
            auto h = json_get_string(item, "sha256", false);
            if (!h.ok()) { return Result<PackageManifest>::failure(h.status); }
            dep.component = std::move(*c.value);
            dep.package = std::move(*p.value);
            dep.sha256 = std::move(*h.value);
            manifest.dependencies.emplace_back(std::move(dep));
        }
    }
    if (auto *hashes = yyjson_obj_get(root, "source_hashes"); hashes != nullptr && !yyjson_is_null(hashes)) {
        if (!yyjson_is_arr(hashes)) {
            return Result<PackageManifest>::failure(StatusCode::invalid_argument, "manifest.source_hashes must be an array", "source_hashes");
        }
        yyjson_val *item;
        size_t idx, max;
        yyjson_arr_foreach(hashes, idx, max, item) {
            if (!yyjson_is_obj(item)) {
                return Result<PackageManifest>::failure(StatusCode::invalid_argument, "source hash entry must be an object", "source_hashes");
            }
            SourceHash hash;
            auto p = json_get_string(item, "path", true);
            if (!p.ok()) { return Result<PackageManifest>::failure(p.status); }
            auto a = json_get_string(item, "algorithm", true);
            if (!a.ok()) { return Result<PackageManifest>::failure(a.status); }
            auto d = json_get_string(item, "digest", true);
            if (!d.ok()) { return Result<PackageManifest>::failure(d.status); }
            hash.path = std::move(*p.value);
            hash.algorithm = std::move(*a.value);
            hash.digest = std::move(*d.value);
            manifest.source_hashes.emplace_back(std::move(hash));
        }
    }
    auto valid = manifest.validate();
    if (!valid.ok()) { return Result<PackageManifest>::failure(std::move(valid)); }
    return Result<PackageManifest>::success(std::move(manifest));
}

// ---------------------------------------------------------------------------
// PackageReader
// ---------------------------------------------------------------------------

PackageReader::~PackageReader() noexcept {
    if (_file != nullptr) { std::fclose(_file); }
}

[[nodiscard]] Status PackageReader::read_exact(uint64_t offset, luisa::span<std::byte> out) {
    if (out.empty()) { return Status::success(); }
    uint64_t end = 0;
    if (!checked_add(offset, out.size(), end) || end > _file_size) {
        return Status::error(StatusCode::io_error, "read range exceeds the package file size", "range");
    }
    auto seek = seek_to(_file, offset);
    if (!seek.ok()) { return seek; }
    size_t remaining = out.size();
    size_t pos = 0;
    while (remaining > 0) {
        const size_t chunk = remaining > (1u << 20u) ? (1u << 20u) : remaining;
        const size_t got = std::fread(out.data() + pos, 1, chunk, _file);
        if (got == 0) {
            return Status::error(StatusCode::io_error, "unexpected end of package file", "read");
        }
        pos += got;
        remaining -= got;
    }
    return Status::success();
}

[[nodiscard]] Result<luisa::unique_ptr<PackageReader, PackageReaderDeleter>> PackageReader::open(const luisa::filesystem::path &path) {
    using R = Result<luisa::unique_ptr<PackageReader, PackageReaderDeleter>>;
    const auto path_text = luisa::to_string(path);
    std::FILE *file = std::fopen(path_text.c_str(), "rb");
    if (file == nullptr) {
        return R::failure(StatusCode::not_found,
                          luisa::string{"cannot open package file '"} + path_text + "'", "package");
    }
    auto reader = luisa::unique_ptr<PackageReader, PackageReaderDeleter>{new PackageReader{}};
    reader->_file = file;
    auto size = file_size_of(file);
    if (!size.ok()) { return R::failure(size.status); }
    reader->_file_size = *size.value;
    if (reader->_file_size < kPackageHeaderSize) {
        return R::failure(StatusCode::invalid_argument, "package file is smaller than the header", "header");
    }

    // --- header ---
    std::byte header_bytes[kPackageHeaderSize];
    auto hr = reader->read_exact(0, header_bytes);
    if (!hr.ok()) { return R::failure(hr); }
    ByteReader header{header_bytes, 0};
    uint64_t magic = 0, manifest_offset = 0, manifest_size = 0, table_offset = 0,
             table_size = 0, payload_base = 0, file_size = 0;
    uint32_t version = 0, header_size = 0;
    if (!header.read_u64(magic) || magic != kMagic) {
        return R::failure(StatusCode::invalid_argument, "bad package magic (not a .lcocr v1 file)", "header.magic");
    }
    if (!header.read_u32(version) || version != kPackageFormatVersion) {
        return R::failure(StatusCode::unsupported, "unsupported package format version", "header.format_version");
    }
    if (!header.read_u32(header_size) || header_size != kPackageHeaderSize) {
        return R::failure(StatusCode::invalid_argument, "unexpected header size", "header.header_size");
    }
    if (!header.read_u64(manifest_offset) || !header.read_u64(manifest_size) ||
        !header.read_u64(table_offset) || !header.read_u64(table_size) ||
        !header.read_u64(payload_base) || !header.read_u64(file_size)) {
        return R::failure(StatusCode::invalid_argument, "truncated package header", "header");
    }
    if (file_size != reader->_file_size) {
        return R::failure(StatusCode::invalid_argument, "header file_size does not match the actual file size", "header.file_size");
    }
    if (manifest_size == 0 || manifest_size > kPackageMaxManifestBytes) {
        return R::failure(StatusCode::invalid_argument, "manifest size is zero or exceeds the limit", "header.manifest_size");
    }
    if (table_size < 8 || table_size > kPackageMaxTableBytes) {
        return R::failure(StatusCode::invalid_argument, "table size is invalid or exceeds the limit", "header.table_size");
    }
    uint64_t manifest_end = 0, table_end = 0;
    if (!checked_add(manifest_offset, manifest_size, manifest_end) || manifest_end > file_size ||
        manifest_offset < kPackageHeaderSize) {
        return R::failure(StatusCode::invalid_argument, "manifest range is out of bounds", "header.manifest");
    }
    if (!checked_add(table_offset, table_size, table_end) || table_end > file_size ||
        table_offset < manifest_end) {
        return R::failure(StatusCode::invalid_argument, "table range is out of bounds or overlaps the manifest", "header.table");
    }
    // The table is the LAST region: no trailing bytes are permitted after it.
    if (table_end != file_size) {
        return R::failure(StatusCode::invalid_argument, "trailing data after the tensor table", "header.table");
    }
    // Layout: header, then 256-aligned payloads, then manifest, then table.
    // An EMPTY package has no payloads: payload_base == manifest_offset (which
    // may be the header end, 64, when no padding was needed).
    if (payload_base < kPackageHeaderSize || payload_base > manifest_offset ||
        (payload_base % kPackagePayloadAlignment) != 0) {
        return R::failure(StatusCode::invalid_argument, "payload base is misaligned or out of bounds", "header.payload_base");
    }

    // --- manifest ---
    luisa::vector<std::byte> manifest_bytes;
    manifest_bytes.resize(static_cast<size_t>(manifest_size));
    hr = reader->read_exact(manifest_offset, manifest_bytes);
    if (!hr.ok()) { return R::failure(hr); }
    const luisa::string_view manifest_text{reinterpret_cast<const char *>(manifest_bytes.data()), manifest_bytes.size()};
    if (!is_valid_utf8(manifest_text)) {
        return R::failure(invalid_utf8_error("manifest"));
    }
    auto manifest = parse_package_manifest(manifest_text);
    if (!manifest.ok()) { return R::failure(manifest.status); }
    reader->_manifest = std::move(*manifest.value);

    // --- table ---
    luisa::vector<std::byte> table_bytes;
    table_bytes.resize(static_cast<size_t>(table_size));
    hr = reader->read_exact(table_offset, table_bytes);
    if (!hr.ok()) { return R::failure(hr); }
    ByteReader table{table_bytes, table_offset};
    uint64_t tensor_count = 0;
    if (!table.read_u64(tensor_count)) {
        return R::failure(StatusCode::invalid_argument, "truncated tensor table", "table.count");
    }
    if (tensor_count > kPackageMaxTensors) {
        return R::failure(StatusCode::invalid_argument, "tensor count exceeds the limit", "table.count");
    }
    // Collect payload ranges for overlap / coverage checks.
    struct Range {
        uint64_t offset;
        uint64_t size;
        luisa::string owner;
    };
    luisa::vector<Range> ranges;
    luisa::unordered_map<luisa::string, uint32_t> names;
    reader->_tensors.reserve(static_cast<size_t>(tensor_count));
    for (uint64_t t = 0; t < tensor_count; t++) {
        PackageTensor tensor;
        uint32_t name_size = 0, rank = 0, encoding = 0, layout = 0, quant_kind = 0, group_size = 0, reserved32 = 0;
        if (!table.read_u32(name_size) || !table.read_u32(rank) || !table.read_u32(encoding) ||
            !table.read_u32(layout) || !table.read_u32(quant_kind) || !table.read_u32(group_size) ||
            !table.read_u32(reserved32) || !table.read_u64(tensor.element_count)) {
            return R::failure(StatusCode::invalid_argument, "truncated tensor record", "table.record");
        }
        if (reserved32 != 0) {
            return R::failure(StatusCode::invalid_argument, "reserved tensor field must be zero", "table.reserved");
        }
        if (name_size == 0 || name_size > kPackageMaxNameBytes) {
            return R::failure(StatusCode::invalid_argument, "tensor name size is zero or exceeds the limit", "table.name_size");
        }
        if (rank > kPackageMaxRank) {
            return R::failure(StatusCode::unsupported, "tensor rank exceeds the documented maximum (8)", "table.rank");
        }
        if (encoding > static_cast<uint32_t>(TensorEncoding::int8)) {
            return R::failure(StatusCode::unsupported, "unsupported tensor encoding", "table.encoding");
        }
        tensor.encoding = static_cast<TensorEncoding>(encoding);
        if (layout > static_cast<uint32_t>(TensorLayout::conv_ohwi)) {
            return R::failure(StatusCode::unsupported, "unsupported tensor layout tag", "table.layout");
        }
        tensor.layout = static_cast<TensorLayout>(layout);
        if (quant_kind > static_cast<uint32_t>(QuantKind::int8_symmetric)) {
            return R::failure(StatusCode::unsupported, "unsupported quantization kind", "table.quant_kind");
        }
        tensor.quant_kind = static_cast<QuantKind>(quant_kind);
        tensor.group_size = group_size;
        tensor.shape.resize(rank);
        tensor.strides.resize(rank);
        for (uint32_t i = 0; i < rank; i++) {
            if (!table.read_u64(tensor.shape[i])) {
                return R::failure(StatusCode::invalid_argument, "truncated tensor shape", "table.shape");
            }
        }
        for (uint32_t i = 0; i < rank; i++) {
            if (!table.read_u64(tensor.strides[i])) {
                return R::failure(StatusCode::invalid_argument, "truncated tensor strides", "table.strides");
            }
        }
        if (!table.read_u64(tensor.payload_offset) || !table.read_u64(tensor.payload_size) ||
            !table.read_u64(tensor.scales_offset) || !table.read_u64(tensor.scales_size) ||
            !table.read_u64(tensor.zero_points_offset) || !table.read_u64(tensor.zero_points_size)) {
            return R::failure(StatusCode::invalid_argument, "truncated tensor payload metadata", "table.payload");
        }
        luisa::vector<std::byte> name_bytes;
        name_bytes.resize(name_size);
        if (!table.read_bytes(name_bytes)) {
            return R::failure(StatusCode::invalid_argument, "truncated tensor name", "table.name");
        }
        tensor.name.assign(reinterpret_cast<const char *>(name_bytes.data()), name_bytes.size());
        if (!is_valid_utf8(tensor.name)) {
            return R::failure(invalid_utf8_error("tensor name"));
        }
        if (tensor.name.find('\0') != luisa::string::npos) {
            return R::failure(StatusCode::invalid_argument, "tensor name contains a NUL byte", "table.name");
        }
        if (++names[tensor.name] > 1u) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::string{"duplicate tensor name '"} + tensor.name + "'", tensor.name);
        }

        // Shape / stride consistency.
        uint64_t count = 0;
        if (!checked_element_count(tensor.shape, count) || count != tensor.element_count) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::string{"tensor '"} + tensor.name + "' shape product does not match element_count", tensor.name);
        }
        uint64_t expected_stride = 1;
        bool strides_ok = true;
        for (uint32_t i = rank; i > 0; i--) {
            if (tensor.strides[i - 1] != expected_stride) { strides_ok = false; }
            uint64_t next = 0;
            if (!checked_mul(expected_stride, tensor.shape[i - 1], next)) { strides_ok = false; break; }
            expected_stride = next;
        }
        if (!strides_ok) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::string{"tensor '"} + tensor.name + "' strides are not row-major", tensor.name);
        }

        // Payload size by encoding.
        uint64_t expected_payload = 0;
        if (is_quantized_encoding(tensor.encoding)) {
            if (tensor.encoding == TensorEncoding::int4_packed) {
                if (tensor.quant_kind != QuantKind::int4_symmetric && tensor.quant_kind != QuantKind::int4_asymmetric) {
                    return R::failure(StatusCode::invalid_argument,
                                      luisa::string{"tensor '"} + tensor.name + "' INT4 encoding requires an INT4 quant kind", tensor.name);
                }
                if (!int4_packed_size(tensor.element_count, expected_payload)) {
                    return R::failure(StatusCode::invalid_argument, "INT4 packed size overflow", tensor.name);
                }
            } else {
                if (tensor.quant_kind != QuantKind::int8_symmetric) {
                    return R::failure(StatusCode::invalid_argument,
                                      luisa::string{"tensor '"} + tensor.name + "' INT8 encoding requires int8_symmetric", tensor.name);
                }
                expected_payload = tensor.element_count;
            }
            if (!is_supported_group_size(tensor.group_size)) {
                return R::failure(StatusCode::unsupported,
                                  luisa::string{"tensor '"} + tensor.name + "' group size must be 64 or 128", tensor.name);
            }
            uint64_t groups = 0;
            if (!group_count_for(tensor.element_count, tensor.group_size, groups)) {
                return R::failure(StatusCode::invalid_argument, "group count overflow", tensor.name);
            }
            uint64_t scales_bytes = 0;
            if (!checked_mul(groups, 4u, scales_bytes) || tensor.scales_size != scales_bytes) {
                return R::failure(StatusCode::invalid_argument,
                                  luisa::string{"tensor '"} + tensor.name + "' scales size does not match the group count", tensor.name);
            }
            const uint64_t zp_bytes = tensor.quant_kind == QuantKind::int4_asymmetric ? groups : 0u;
            if (tensor.zero_points_size != zp_bytes) {
                return R::failure(StatusCode::invalid_argument,
                                  luisa::string{"tensor '"} + tensor.name + "' zero-point size does not match the quant kind", tensor.name);
            }
        } else {
            if (tensor.quant_kind != QuantKind::none || tensor.group_size != 0 ||
                tensor.scales_size != 0 || tensor.zero_points_size != 0 ||
                tensor.scales_offset != 0 || tensor.zero_points_offset != 0) {
                return R::failure(StatusCode::invalid_argument,
                                  luisa::string{"tensor '"} + tensor.name + "' dense encoding must not carry quantization metadata", tensor.name);
            }
            const auto element_size = dense_encoding_element_size(tensor.encoding);
            if (!checked_mul(tensor.element_count, element_size, expected_payload)) {
                return R::failure(StatusCode::invalid_argument, "dense payload size overflow", tensor.name);
            }
        }
        if (tensor.payload_size != expected_payload) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::string{"tensor '"} + tensor.name + "' payload size does not match shape/encoding", tensor.name);
        }

        // Range checks (alignment, bounds). Empty payloads must have offset 0.
        const auto check_range = [&](uint64_t offset, uint64_t size, const char *what) -> Status {
            if (size == 0) {
                // Canonical zero-size rule: offset 0 (no blob) OR any 256-byte
                // aligned in-bounds position within the payload region. Both
                // writers emit the aligned current position; readers accept
                // both forms. Zero-size ranges never participate in coverage.
                if (offset == 0) { return Status::success(); }
                if (offset < payload_base || offset > manifest_offset ||
                    (offset % kPackagePayloadAlignment) != 0) {
                    return Status::error(StatusCode::invalid_argument,
                                         luisa::string{"tensor '"} + tensor.name + "' empty " + what + " offset is not 0 or an aligned in-bounds position", tensor.name);
                }
                return Status::success();
            }
            uint64_t end = 0;
            if (!checked_add(offset, size, end) || end > file_size || offset < payload_base) {
                return Status::error(StatusCode::invalid_argument,
                                     luisa::string{"tensor '"} + tensor.name + "' " + what + " range is out of bounds", tensor.name);
            }
            if ((offset % kPackagePayloadAlignment) != 0) {
                return Status::error(StatusCode::invalid_argument,
                                     luisa::string{"tensor '"} + tensor.name + "' " + what + " offset is not 256-byte aligned", tensor.name);
            }
            ranges.emplace_back(Range{offset, size, tensor.name});
            return Status::success();
        };
        auto rs = check_range(tensor.payload_offset, tensor.payload_size, "payload");
        if (!rs.ok()) { return R::failure(std::move(rs)); }
        rs = check_range(tensor.scales_offset, tensor.scales_size, "scales");
        if (!rs.ok()) { return R::failure(std::move(rs)); }
        rs = check_range(tensor.zero_points_offset, tensor.zero_points_size, "zero points");
        if (!rs.ok()) { return R::failure(std::move(rs)); }
        reader->_tensors.emplace_back(std::move(tensor));
    }
    if (!table.at_end()) {
        return R::failure(StatusCode::invalid_argument, "trailing bytes after the tensor table", "table");
    }

    // Overlap + padding + coverage: sort ranges by offset, require zero gaps
    // content and full coverage of [kPackageHeaderSize, manifest_offset).
    // An EMPTY package has no payload ranges: its payload region is empty and
    // payload_base must equal manifest_offset (no padding, no payloads).
    if (ranges.empty()) {
        if (payload_base != manifest_offset) {
            return R::failure(StatusCode::invalid_argument,
                              "empty package must have payload_base == manifest_offset", "header.payload_base");
        }
    }
    std::sort(ranges.begin(), ranges.end(), [](const Range &a, const Range &b) { return a.offset < b.offset; });
    uint64_t cursor = kPackageHeaderSize;
    for (const auto &range : ranges) {
        if (range.offset < cursor) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::string{"tensor '"} + range.owner + "' payload overlaps another blob", range.owner);
        }
        if (range.offset > cursor) {
            // Padding gap: must be all zero.
            const uint64_t gap = range.offset - cursor;
            luisa::vector<std::byte> gap_bytes;
            gap_bytes.resize(static_cast<size_t>(std::min<uint64_t>(gap, 1u << 20u)), std::byte{0});
            uint64_t pos = cursor;
            while (pos < range.offset) {
                const auto chunk = static_cast<size_t>(std::min<uint64_t>(range.offset - pos, gap_bytes.size()));
                auto gr = reader->read_exact(pos, luisa::span<std::byte>{gap_bytes.data(), chunk});
                if (!gr.ok()) { return R::failure(gr); }
                for (size_t i = 0; i < chunk; i++) {
                    if (gap_bytes[i] != std::byte{0}) {
                        return R::failure(StatusCode::invalid_argument, "nonzero padding bytes between payloads", "padding");
                    }
                }
                pos += chunk;
            }
        }
        cursor = range.offset + range.size;
    }
    // Any remaining region [cursor, manifest_offset) is trailing padding (this
    // includes the whole payload region of an EMPTY package, which is zero
    // padding up to the aligned manifest offset). It must be all zero.
    if (cursor > manifest_offset) {
        return R::failure(StatusCode::invalid_argument, "payload region overruns the manifest", "payload");
    }
    if (cursor < manifest_offset) {
        const uint64_t gap = manifest_offset - cursor;
        luisa::vector<std::byte> gap_bytes;
        gap_bytes.resize(static_cast<size_t>(std::min<uint64_t>(gap, 1u << 20u)), std::byte{0});
        uint64_t pos = cursor;
        while (pos < manifest_offset) {
            const auto chunk = static_cast<size_t>(std::min<uint64_t>(manifest_offset - pos, gap_bytes.size()));
            auto gr = reader->read_exact(pos, luisa::span<std::byte>{gap_bytes.data(), chunk});
            if (!gr.ok()) { return R::failure(gr); }
            for (size_t i = 0; i < chunk; i++) {
                if (gap_bytes[i] != std::byte{0}) {
                    return R::failure(StatusCode::invalid_argument, "nonzero trailing padding bytes", "padding");
                }
            }
            pos += chunk;
        }
    }

    // Validate quantization metadata values (scales finite, zero points in range).
    for (const auto &tensor : reader->_tensors) {
        if (!is_quantized_encoding(tensor.encoding)) { continue; }
        luisa::vector<float> scales;
        scales.resize(static_cast<size_t>(tensor.scales_size / 4u));
        auto sr = reader->read_tensor_scales(tensor, scales);
        if (!sr.ok()) { return R::failure(sr); }
        luisa::vector<uint8_t> zero_points;
        zero_points.resize(static_cast<size_t>(tensor.zero_points_size));
        auto zr = reader->read_tensor_zero_points(tensor, zero_points);
        if (!zr.ok()) { return R::failure(zr); }
        auto meta = validate_quant_metadata(tensor.quant_kind, tensor.group_size, tensor.element_count, scales, zero_points);
        if (!meta.ok()) {
            return R::failure(Status::error(meta.code,
                                            luisa::string{"tensor '"} + tensor.name + "': " + meta.message, tensor.name));
        }
    }
    return R::success(std::move(reader));
}

[[nodiscard]] const PackageTensor *PackageReader::find(luisa::string_view name) const noexcept {
    for (const auto &tensor : _tensors) {
        if (tensor.name == name) { return &tensor; }
    }
    return nullptr;
}

[[nodiscard]] Status PackageReader::read_tensor_payload(const PackageTensor &tensor, luisa::span<std::byte> out) {
    if (static_cast<uint64_t>(out.size()) != tensor.payload_size) {
        return Status::error(StatusCode::invalid_argument, "output span does not match the payload size", tensor.name);
    }
    return read_exact(tensor.payload_offset, out);
}

[[nodiscard]] Status PackageReader::read_tensor_scales(const PackageTensor &tensor, luisa::span<float> out) {
    if (static_cast<uint64_t>(out.size()) * 4u != tensor.scales_size) {
        return Status::error(StatusCode::invalid_argument, "output span does not match the scales size", tensor.name);
    }
    luisa::vector<std::byte> raw;
    raw.resize(static_cast<size_t>(tensor.scales_size));
    auto r = read_exact(tensor.scales_offset, raw);
    if (!r.ok()) { return r; }
    for (size_t i = 0; i < out.size(); i++) {
        uint32_t bits = 0;
        for (uint32_t b = 0; b < 4u; b++) {
            bits |= static_cast<uint32_t>(raw[i * 4u + b]) << (8u * b);
        }
        std::memcpy(&out[i], &bits, sizeof(float));
    }
    return Status::success();
}

[[nodiscard]] Status PackageReader::read_tensor_zero_points(const PackageTensor &tensor, luisa::span<uint8_t> out) {
    if (static_cast<uint64_t>(out.size()) != tensor.zero_points_size) {
        return Status::error(StatusCode::invalid_argument, "output span does not match the zero-point size", tensor.name);
    }
    luisa::vector<std::byte> raw;
    raw.resize(static_cast<size_t>(tensor.zero_points_size));
    auto r = read_exact(tensor.zero_points_offset, raw);
    if (!r.ok()) { return r; }
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = static_cast<uint8_t>(raw[i]);
    }
    return Status::success();
}

[[nodiscard]] Status PackageReader::read_tensor_f32(const PackageTensor &tensor, luisa::span<float> out) {
    if (is_quantized_encoding(tensor.encoding)) {
        return Status::error(StatusCode::invalid_argument,
                             "quantized tensor requires dequantization, not a dense read", tensor.name);
    }
    if (static_cast<uint64_t>(out.size()) != tensor.element_count) {
        return Status::error(StatusCode::invalid_argument, "output span does not match the element count", tensor.name);
    }
    luisa::vector<std::byte> raw;
    raw.resize(static_cast<size_t>(tensor.payload_size));
    auto r = read_tensor_payload(tensor, raw);
    if (!r.ok()) { return r; }
    if (tensor.encoding == TensorEncoding::fp32) {
        for (size_t i = 0; i < out.size(); i++) {
            uint32_t bits = 0;
            for (uint32_t b = 0; b < 4u; b++) {
                bits |= static_cast<uint32_t>(raw[i * 4u + b]) << (8u * b);
            }
            std::memcpy(&out[i], &bits, sizeof(float));
        }
    } else {
        for (size_t i = 0; i < out.size(); i++) {
            const uint16_t bits = static_cast<uint16_t>(raw[i * 2u]) |
                                  (static_cast<uint16_t>(raw[i * 2u + 1u]) << 8u);
            out[i] = tensor.encoding == TensorEncoding::fp16 ? fp16_to_fp32(bits) : bf16_to_fp32(bits);
        }
    }
    return Status::success();
}

[[nodiscard]] Status PackageReader::read_tensor_dequantized(const PackageTensor &tensor, luisa::span<float> out) {
    if (!is_quantized_encoding(tensor.encoding)) {
        return Status::error(StatusCode::invalid_argument, "tensor is not quantized", tensor.name);
    }
    if (static_cast<uint64_t>(out.size()) != tensor.element_count) {
        return Status::error(StatusCode::invalid_argument, "output span does not match the element count", tensor.name);
    }
    luisa::vector<float> scales;
    scales.resize(static_cast<size_t>(tensor.scales_size / 4u));
    auto sr = read_tensor_scales(tensor, scales);
    if (!sr.ok()) { return sr; }
    luisa::vector<uint8_t> zero_points;
    zero_points.resize(static_cast<size_t>(tensor.zero_points_size));
    auto zr = read_tensor_zero_points(tensor, zero_points);
    if (!zr.ok()) { return zr; }
    luisa::vector<std::byte> raw;
    raw.resize(static_cast<size_t>(tensor.payload_size));
    auto pr = read_tensor_payload(tensor, raw);
    if (!pr.ok()) { return pr; }
    if (tensor.encoding == TensorEncoding::int4_packed) {
        luisa::vector<uint8_t> packed;
        packed.resize(raw.size());
        for (size_t i = 0; i < raw.size(); i++) { packed[i] = static_cast<uint8_t>(raw[i]); }
        return dequantize_int4(packed, tensor.element_count, scales, zero_points,
                               tensor.group_size, tensor.quant_kind == QuantKind::int4_symmetric, out);
    }
    luisa::vector<int8_t> codes;
    codes.resize(raw.size());
    for (size_t i = 0; i < raw.size(); i++) { codes[i] = static_cast<int8_t>(raw[i]); }
    return dequantize_int8(codes, tensor.element_count, scales, tensor.group_size, out);
}

// ---------------------------------------------------------------------------
// PackageWriter
// ---------------------------------------------------------------------------

PackageWriter::~PackageWriter() noexcept {
    if (_file != nullptr) { std::fclose(_file); }
}

[[nodiscard]] Result<luisa::unique_ptr<PackageWriter, PackageWriterDeleter>> PackageWriter::create(const luisa::filesystem::path &path) {
    using R = Result<luisa::unique_ptr<PackageWriter, PackageWriterDeleter>>;
    const auto path_text = luisa::to_string(path);
    std::FILE *file = std::fopen(path_text.c_str(), "wb");
    if (file == nullptr) {
        return R::failure(StatusCode::io_error,
                          luisa::string{"cannot create package file '"} + path_text + "'", "package");
    }
    auto writer = luisa::unique_ptr<PackageWriter, PackageWriterDeleter>{new PackageWriter{}};
    writer->_file = file;
    writer->_path = path;
    // Reserve the fixed header; rewritten by finish().
    std::byte header[kPackageHeaderSize] = {};
    if (std::fwrite(header, 1, sizeof(header), file) != sizeof(header)) {
        return R::failure(StatusCode::io_error, "cannot write package header placeholder", "header");
    }
    writer->_offset = kPackageHeaderSize;
    return R::success(std::move(writer));
}

[[nodiscard]] Status PackageWriter::write_all(luisa::span<const std::byte> bytes) {
    if (bytes.empty()) { return Status::success(); }
    if (std::fwrite(bytes.data(), 1, bytes.size(), _file) != bytes.size()) {
        return Status::error(StatusCode::io_error, "short write to package file", "write");
    }
    uint64_t next = 0;
    if (!checked_add(_offset, bytes.size(), next)) {
        return Status::error(StatusCode::invalid_argument, "package size overflow", "write");
    }
    _offset = next;
    return Status::success();
}

[[nodiscard]] Status PackageWriter::pad_to_alignment() {
    uint64_t aligned = 0;
    if (!checked_align_up(_offset, kPackagePayloadAlignment, aligned)) {
        return Status::error(StatusCode::invalid_argument, "alignment overflow", "align");
    }
    const auto padding = static_cast<size_t>(aligned - _offset);
    if (padding > 0) {
        std::byte zeros[kPackagePayloadAlignment] = {};
        auto w = write_all(luisa::span<const std::byte>{zeros, padding});
        if (!w.ok()) { return w; }
    }
    return Status::success();
}

[[nodiscard]] Status PackageWriter::add_tensor(const TensorSpec &spec) {
    if (_finished) {
        return Status::error(StatusCode::internal_error, "writer already finished", "writer");
    }
    if (spec.name.empty() || spec.name.size() > kPackageMaxNameBytes ||
        !is_valid_utf8(spec.name) || spec.name.find('\0') != luisa::string::npos) {
        return Status::error(StatusCode::invalid_argument, "invalid tensor name", "name");
    }
    for (const auto &t : _tensors) {
        if (t.name == spec.name) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::string{"duplicate tensor name '"} + spec.name + "'", spec.name);
        }
    }
    if (spec.shape.size() > kPackageMaxRank) {
        return Status::error(StatusCode::unsupported, "tensor rank exceeds the documented maximum (8)", "shape");
    }
    PackageTensor tensor;
    tensor.name = spec.name;
    tensor.encoding = spec.encoding;
    tensor.layout = spec.layout;
    tensor.quant_kind = spec.quant_kind;
    tensor.group_size = spec.group_size;
    tensor.shape = spec.shape;
    if (!checked_element_count(spec.shape, tensor.element_count)) {
        return Status::error(StatusCode::invalid_argument, "shape product overflow", "shape");
    }
    tensor.strides.resize(spec.shape.size());
    uint64_t stride = 1;
    for (size_t i = spec.shape.size(); i > 0; i--) {
        tensor.strides[i - 1] = stride;
        if (!checked_mul(stride, spec.shape[i - 1], stride)) {
            return Status::error(StatusCode::invalid_argument, "stride overflow", "shape");
        }
    }

    // Validate payload size against the encoding.
    if (is_quantized_encoding(spec.encoding)) {
        uint64_t expected = 0;
        if (spec.encoding == TensorEncoding::int4_packed) {
            if (spec.quant_kind != QuantKind::int4_symmetric && spec.quant_kind != QuantKind::int4_asymmetric) {
                return Status::error(StatusCode::invalid_argument, "INT4 encoding requires an INT4 quant kind", spec.name);
            }
            if (!int4_packed_size(tensor.element_count, expected)) {
                return Status::error(StatusCode::invalid_argument, "INT4 packed size overflow", spec.name);
            }
        } else {
            if (spec.quant_kind != QuantKind::int8_symmetric) {
                return Status::error(StatusCode::invalid_argument, "INT8 encoding requires int8_symmetric", spec.name);
            }
            expected = tensor.element_count;
        }
        if (spec.payload.size() != expected) {
            return Status::error(StatusCode::invalid_argument, "payload size does not match shape/encoding", spec.name);
        }
        luisa::vector<uint8_t> zp = spec.zero_points;
        auto meta = validate_quant_metadata(spec.quant_kind, spec.group_size, tensor.element_count, spec.scales, zp);
        if (!meta.ok()) { return meta; }
    } else {
        if (spec.quant_kind != QuantKind::none || spec.group_size != 0 ||
            !spec.scales.empty() || !spec.zero_points.empty()) {
            return Status::error(StatusCode::invalid_argument, "dense encoding must not carry quantization metadata", spec.name);
        }
        const auto element_size = dense_encoding_element_size(spec.encoding);
        uint64_t expected = 0;
        if (!checked_mul(tensor.element_count, element_size, expected) || spec.payload.size() != expected) {
            return Status::error(StatusCode::invalid_argument, "payload size does not match shape/encoding", spec.name);
        }
    }

    // Stream payload, scales, zero points (each 256-byte aligned).
    auto pad = pad_to_alignment();
    if (!pad.ok()) { return pad; }
    if (_payload_base == 0) { _payload_base = _offset; }
    tensor.payload_offset = _offset;
    tensor.payload_size = spec.payload.size();
    auto w = write_all(spec.payload);
    if (!w.ok()) { return w; }
    if (!spec.scales.empty()) {
        pad = pad_to_alignment();
        if (!pad.ok()) { return pad; }
        tensor.scales_offset = _offset;
        tensor.scales_size = spec.scales.size() * 4u;
        luisa::vector<std::byte> raw;
        raw.reserve(static_cast<size_t>(tensor.scales_size));
        for (auto s : spec.scales) {
            uint32_t bits = 0;
            std::memcpy(&bits, &s, sizeof(bits));
            append_u32(raw, bits);
        }
        w = write_all(raw);
        if (!w.ok()) { return w; }
    }
    if (!spec.zero_points.empty()) {
        pad = pad_to_alignment();
        if (!pad.ok()) { return pad; }
        tensor.zero_points_offset = _offset;
        tensor.zero_points_size = spec.zero_points.size();
        luisa::vector<std::byte> raw;
        raw.reserve(spec.zero_points.size());
        for (auto zp : spec.zero_points) { raw.emplace_back(static_cast<std::byte>(zp)); }
        w = write_all(raw);
        if (!w.ok()) { return w; }
    }
    _tensors.emplace_back(std::move(tensor));
    return Status::success();
}

[[nodiscard]] Status PackageWriter::finish(const PackageManifest &manifest) {
    if (_finished) {
        return Status::error(StatusCode::internal_error, "writer already finished", "writer");
    }
    auto manifest_json = serialize_package_manifest(manifest);
    if (!manifest_json.ok()) { return manifest_json.status; }
    const auto &json = *manifest_json.value;
    if (json.size() > kPackageMaxManifestBytes) {
        return Status::error(StatusCode::invalid_argument, "manifest exceeds the size limit", "manifest");
    }
    // Canonical empty layout: with no payloads, pad to the 256-byte payload
    // alignment so payload_base == manifest_offset is aligned (matching the
    // non-empty case where the first payload is aligned). The padding is zero.
    if (_tensors.empty()) {
        auto pad = pad_to_alignment();
        if (!pad.ok()) { return pad; }
    }
    const uint64_t manifest_offset = _offset;
    auto w = write_all(luisa::span<const std::byte>{reinterpret_cast<const std::byte *>(json.data()), json.size()});
    if (!w.ok()) { return w; }
    const uint64_t table_offset = _offset;

    // Build the table in memory (metadata only).
    luisa::vector<std::byte> table;
    append_u64(table, _tensors.size());
    for (const auto &tensor : _tensors) {
        append_u32(table, static_cast<uint32_t>(tensor.name.size()));
        append_u32(table, static_cast<uint32_t>(tensor.shape.size()));
        append_u32(table, static_cast<uint32_t>(tensor.encoding));
        append_u32(table, static_cast<uint32_t>(tensor.layout));
        append_u32(table, static_cast<uint32_t>(tensor.quant_kind));
        append_u32(table, tensor.group_size);
        append_u32(table, 0u);
        append_u64(table, tensor.element_count);
        for (auto d : tensor.shape) { append_u64(table, d); }
        for (auto s : tensor.strides) { append_u64(table, s); }
        append_u64(table, tensor.payload_offset);
        append_u64(table, tensor.payload_size);
        append_u64(table, tensor.scales_offset);
        append_u64(table, tensor.scales_size);
        append_u64(table, tensor.zero_points_offset);
        append_u64(table, tensor.zero_points_size);
        for (auto c : tensor.name) { table.emplace_back(static_cast<std::byte>(c)); }
    }
    if (table.size() > kPackageMaxTableBytes) {
        return Status::error(StatusCode::invalid_argument, "table exceeds the size limit", "table");
    }
    w = write_all(table);
    if (!w.ok()) { return w; }
    const uint64_t file_size = _offset;

    // Rewrite the header.
    luisa::vector<std::byte> header;
    append_u64(header, kMagic);
    append_u32(header, kPackageFormatVersion);
    append_u32(header, kPackageHeaderSize);
    append_u64(header, manifest_offset);
    append_u64(header, json.size());
    append_u64(header, table_offset);
    append_u64(header, table.size());
    // Empty packages have no payloads: payload_base == manifest_offset (the
    // empty payload region starts where the manifest begins). Non-empty
    // packages use the first payload's aligned offset.
    append_u64(header, _payload_base == 0 ? manifest_offset : _payload_base);
    append_u64(header, file_size);
    auto seek = seek_to(_file, 0);
    if (!seek.ok()) { return seek; }
    if (std::fwrite(header.data(), 1, header.size(), _file) != header.size()) {
        return Status::error(StatusCode::io_error, "cannot rewrite package header", "header");
    }
    if (std::fflush(_file) != 0) {
        return Status::error(StatusCode::io_error, "cannot flush package file", "file");
    }
    std::fclose(_file);
    _file = nullptr;
    _finished = true;

    // Prove the artifact by reopening it with the strict reader.
    auto reopened = PackageReader::open(_path);
    if (!reopened.ok()) { return reopened.status; }
    return Status::success();
}

}// namespace luisa::ocr