// OCR engine -- .ocrfeat feature bundle interchange (Phase 1).
// Same checked range/overlap discipline as the .lcocr package.
#include "ocr/feature_bundle.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include <luisa/core/stl/unordered_map.h>
#include <yyjson.h>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace luisa::ocr {
namespace {

constexpr uint32_t kMagic = 0x4652434fu;// "OCRF" little-endian

struct JsonDoc {
    yyjson_doc *doc{nullptr};
    ~JsonDoc() noexcept {
        if (doc != nullptr) { yyjson_doc_free(doc); }
    }
};

[[nodiscard]] luisa::string_view json_str(yyjson_val *v) noexcept {
    if (v == nullptr || !yyjson_is_str(v)) { return {}; }
    return {yyjson_get_str(v), yyjson_get_len(v)};
}

void append_u32(luisa::vector<std::byte> &out, uint32_t v) {
    for (uint32_t i = 0; i < 4u; i++) { out.emplace_back(static_cast<std::byte>((v >> (8u * i)) & 0xffu)); }
}
void append_u64(luisa::vector<std::byte> &out, uint64_t v) {
    for (uint32_t i = 0; i < 8u; i++) { out.emplace_back(static_cast<std::byte>((v >> (8u * i)) & 0xffu)); }
}

[[nodiscard]] Result<uint64_t> file_size_of(std::FILE *file) {
#ifdef _WIN32
    const auto size = _filelengthi64(_fileno(file));
    if (size < 0) { return Result<uint64_t>::failure(StatusCode::io_error, "cannot query file size", "file"); }
    return Result<uint64_t>::success(static_cast<uint64_t>(size));
#else
    const int fd = fileno(file);
    const auto size = lseek(fd, 0, SEEK_END);
    if (size < 0 || lseek(fd, 0, SEEK_SET) < 0) { return Result<uint64_t>::failure(StatusCode::io_error, "cannot query file size", "file"); }
    return Result<uint64_t>::success(static_cast<uint64_t>(size));
#endif
}

[[nodiscard]] Status seek_to(std::FILE *file, uint64_t offset) {
#ifdef _WIN32
    const int rc = _fseeki64(file, static_cast<int64_t>(offset), SEEK_SET);
#else
    const int rc = fseeko(file, static_cast<off_t>(offset), SEEK_SET);
#endif
    if (rc != 0) { return Status::error(StatusCode::io_error, "cannot seek in feature bundle", "file"); }
    return Status::success();
}

}// namespace

void FeatureBundleReaderDeleter::operator()(FeatureBundleReader *reader) const noexcept { delete reader; }
void FeatureBundleWriterDeleter::operator()(FeatureBundleWriter *writer) const noexcept { delete writer; }

[[nodiscard]] luisa::string_view feature_dtype_name(FeatureDtype dtype) noexcept {
    switch (dtype) {
        case FeatureDtype::f32: return "f32";
        case FeatureDtype::i32: return "i32";
        case FeatureDtype::i64: return "i64";
        case FeatureDtype::u8: return "u8";
        default: return "unknown";
    }
}

[[nodiscard]] uint64_t feature_dtype_size(FeatureDtype dtype) noexcept {
    switch (dtype) {
        case FeatureDtype::f32:
        case FeatureDtype::i32: return 4;
        case FeatureDtype::i64: return 8;
        case FeatureDtype::u8: return 1;
        default: return 0;
    }
}

[[nodiscard]] Result<luisa::string> serialize_feature_meta(const FeatureBundleMeta &meta) {
    using R = Result<luisa::string>;
    if (meta.producer.empty() || meta.processor.empty()) {
        return R::failure(StatusCode::invalid_argument, "feature meta requires producer and processor", "meta");
    }
    yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
    if (doc == nullptr) { return R::failure(StatusCode::out_of_memory, "cannot allocate feature meta", "meta"); }
    auto *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_uint(doc, root, "format_version", kFeatureBundleVersion);
    const auto add = [&](const char *key, const luisa::string &value) {
        yyjson_mut_obj_add_strncpy(doc, root, key, value.data(), value.size());
    };
    add("producer", meta.producer);
    add("processor", meta.processor);
    add("processor_revision", meta.processor_revision);
    add("source_image_hash", meta.source_image_hash);
    add("decode_settings", meta.decode_settings);
    size_t length = 0;
    char *json = yyjson_mut_write(doc, YYJSON_WRITE_NOFLAG, &length);
    yyjson_mut_doc_free(doc);
    if (json == nullptr) { return R::failure(StatusCode::out_of_memory, "cannot serialize feature meta", "meta"); }
    luisa::string out{json, length};
    std::free(json);
    return R::success(std::move(out));
}

[[nodiscard]] Result<FeatureBundleMeta> parse_feature_meta(luisa::string_view json) {
    using R = Result<FeatureBundleMeta>;
    JsonDoc holder;
    yyjson_read_err err;
    holder.doc = yyjson_read_opts(const_cast<char *>(json.data()), json.size(), YYJSON_READ_NOFLAG, nullptr, &err);
    if (holder.doc == nullptr) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::string{"feature meta JSON parse error: "} + (err.msg != nullptr ? err.msg : "unknown"), "meta");
    }
    auto *root = yyjson_doc_get_root(holder.doc);
    if (!yyjson_is_obj(root)) { return R::failure(StatusCode::invalid_argument, "feature meta root must be an object", "meta"); }
    auto *version = yyjson_obj_get(root, "format_version");
    if (version == nullptr || !yyjson_is_uint(version) || yyjson_get_uint(version) != kFeatureBundleVersion) {
        return R::failure(StatusCode::unsupported, "feature meta format_version must be 1", "format_version");
    }
    FeatureBundleMeta meta;
    meta.producer = luisa::string{json_str(yyjson_obj_get(root, "producer"))};
    meta.processor = luisa::string{json_str(yyjson_obj_get(root, "processor"))};
    meta.processor_revision = luisa::string{json_str(yyjson_obj_get(root, "processor_revision"))};
    meta.source_image_hash = luisa::string{json_str(yyjson_obj_get(root, "source_image_hash"))};
    meta.decode_settings = luisa::string{json_str(yyjson_obj_get(root, "decode_settings"))};
    if (meta.producer.empty() || meta.processor.empty()) {
        return R::failure(StatusCode::invalid_argument, "feature meta requires producer and processor", "meta");
    }
    return R::success(std::move(meta));
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

FeatureBundleReader::~FeatureBundleReader() noexcept {
    if (_file != nullptr) { std::fclose(_file); }
}

[[nodiscard]] Status FeatureBundleReader::read_exact(uint64_t offset, luisa::span<std::byte> out) {
    if (out.empty()) { return Status::success(); }
    uint64_t end = 0;
    if (!checked_add(offset, out.size(), end) || end > _file_size) {
        return Status::error(StatusCode::io_error, "read range exceeds the feature bundle size", "range");
    }
    auto seek = seek_to(_file, offset);
    if (!seek.ok()) { return seek; }
    size_t remaining = out.size();
    size_t pos = 0;
    while (remaining > 0) {
        const size_t chunk = remaining > (1u << 20u) ? (1u << 20u) : remaining;
        const size_t got = std::fread(out.data() + pos, 1, chunk, _file);
        if (got == 0) { return Status::error(StatusCode::io_error, "unexpected end of feature bundle", "read"); }
        pos += got;
        remaining -= got;
    }
    return Status::success();
}

[[nodiscard]] Result<luisa::unique_ptr<FeatureBundleReader, FeatureBundleReaderDeleter>> FeatureBundleReader::open(const luisa::filesystem::path &path) {
    using R = Result<luisa::unique_ptr<FeatureBundleReader, FeatureBundleReaderDeleter>>;
    const auto path_text = luisa::to_string(path);
    std::FILE *file = std::fopen(path_text.c_str(), "rb");
    if (file == nullptr) {
        return R::failure(StatusCode::not_found, luisa::string{"cannot open feature bundle '"} + path_text + "'", "feature_bundle");
    }
    auto reader = luisa::unique_ptr<FeatureBundleReader, FeatureBundleReaderDeleter>{new FeatureBundleReader{}};
    reader->_file = file;
    auto size = file_size_of(file);
    if (!size.ok()) { return R::failure(size.status); }
    reader->_file_size = *size.value;
    if (reader->_file_size < kFeatureBundleHeaderSize) {
        return R::failure(StatusCode::invalid_argument, "feature bundle smaller than the header", "header");
    }
    std::byte header[kFeatureBundleHeaderSize];
    auto hr = reader->read_exact(0, header);
    if (!hr.ok()) { return R::failure(hr); }
    const auto rd64 = [&](size_t off) -> uint64_t {
        uint64_t v = 0;
        for (uint32_t i = 0; i < 8u; i++) { v |= static_cast<uint64_t>(header[off + i]) << (8u * i); }
        return v;
    };
    const auto rd32 = [&](size_t off) -> uint32_t {
        uint32_t v = 0;
        for (uint32_t i = 0; i < 4u; i++) { v |= static_cast<uint32_t>(header[off + i]) << (8u * i); }
        return v;
    };
    if (rd32(0) != kMagic) { return R::failure(StatusCode::invalid_argument, "bad feature bundle magic", "header.magic"); }
    if (rd32(4) != kFeatureBundleVersion) { return R::failure(StatusCode::unsupported, "unsupported feature bundle version", "header.version"); }
    if (rd32(8) != kFeatureBundleHeaderSize) { return R::failure(StatusCode::invalid_argument, "unexpected feature header size", "header.size"); }
    if (rd32(12) != 0u) { return R::failure(StatusCode::invalid_argument, "feature header reserved field must be zero", "header.reserved"); }
    const uint64_t meta_offset = rd64(16), meta_size = rd64(24);
    const uint64_t table_offset = rd64(32), table_size = rd64(40);
    const uint64_t payload_base = rd64(48), file_size = rd64(56);
    if (file_size != reader->_file_size) {
        return R::failure(StatusCode::invalid_argument, "feature header file_size mismatch", "header.file_size");
    }
    uint64_t meta_end = 0, table_end = 0;
    if (meta_size == 0 || !checked_add(meta_offset, meta_size, meta_end) || meta_end > file_size || meta_offset < kFeatureBundleHeaderSize) {
        return R::failure(StatusCode::invalid_argument, "feature meta range out of bounds", "header.meta");
    }
    if (table_size < 8 || !checked_add(table_offset, table_size, table_end) || table_end > file_size || table_offset < meta_end) {
        return R::failure(StatusCode::invalid_argument, "feature table range out of bounds", "header.table");
    }
    if (payload_base < kFeatureBundleHeaderSize || payload_base > meta_offset || (payload_base % kFeatureBundleAlignment) != 0) {
        return R::failure(StatusCode::invalid_argument, "feature payload base misaligned/out of bounds", "header.payload_base");
    }
    // meta
    luisa::vector<std::byte> meta_bytes;
    meta_bytes.resize(static_cast<size_t>(meta_size));
    hr = reader->read_exact(meta_offset, meta_bytes);
    if (!hr.ok()) { return R::failure(hr); }
    auto meta = parse_feature_meta(luisa::string_view{reinterpret_cast<const char *>(meta_bytes.data()), meta_bytes.size()});
    if (!meta.ok()) { return R::failure(meta.status); }
    reader->_meta = std::move(*meta.value);
    // table
    luisa::vector<std::byte> table_bytes;
    table_bytes.resize(static_cast<size_t>(table_size));
    hr = reader->read_exact(table_offset, table_bytes);
    if (!hr.ok()) { return R::failure(hr); }
    size_t tpos = 0;
    const auto t64 = [&](uint64_t &v) -> bool {
        if (tpos + 8 > table_bytes.size()) { return false; }
        v = 0;
        for (uint32_t i = 0; i < 8u; i++) { v |= static_cast<uint64_t>(table_bytes[tpos + i]) << (8u * i); }
        tpos += 8;
        return true;
    };
    const auto t32 = [&](uint32_t &v) -> bool {
        if (tpos + 4 > table_bytes.size()) { return false; }
        v = 0;
        for (uint32_t i = 0; i < 4u; i++) { v |= static_cast<uint32_t>(table_bytes[tpos + i]) << (8u * i); }
        tpos += 4;
        return true;
    };
    uint64_t count = 0;
    if (!t64(count) || count > (1ull << 20u)) {
        return R::failure(StatusCode::invalid_argument, "feature tensor count invalid", "table.count");
    }
    struct Range { uint64_t offset; uint64_t size; luisa::string owner; };
    luisa::vector<Range> ranges;
    luisa::unordered_map<luisa::string, uint32_t> names;
    for (uint64_t t = 0; t < count; t++) {
        FeatureTensor tensor;
        uint32_t name_size = 0, dtype = 0, rank = 0, reserved = 0;
        if (!t32(name_size) || !t32(dtype) || !t32(rank) || !t32(reserved) || !t64(tensor.element_count)) {
            return R::failure(StatusCode::invalid_argument, "truncated feature tensor record", "table.record");
        }
        if (reserved != 0 || name_size == 0 || name_size > 4096u || rank > kFeatureBundleMaxRank) {
            return R::failure(StatusCode::invalid_argument, "invalid feature tensor record fields", "table.record");
        }
        if (dtype > static_cast<uint32_t>(FeatureDtype::u8)) {
            return R::failure(StatusCode::unsupported, "unsupported feature dtype", "table.dtype");
        }
        tensor.dtype = static_cast<FeatureDtype>(dtype);
        tensor.shape.resize(rank);
        for (uint32_t i = 0; i < rank; i++) {
            if (!t64(tensor.shape[i])) { return R::failure(StatusCode::invalid_argument, "truncated feature shape", "table.shape"); }
        }
        if (!t64(tensor.payload_offset) || !t64(tensor.payload_size)) {
            return R::failure(StatusCode::invalid_argument, "truncated feature payload metadata", "table.payload");
        }
        if (tpos + name_size > table_bytes.size()) {
            return R::failure(StatusCode::invalid_argument, "truncated feature tensor name", "table.name");
        }
        tensor.name.assign(reinterpret_cast<const char *>(table_bytes.data() + tpos), name_size);
        tpos += name_size;
        if (++names[tensor.name] > 1u) {
            return R::failure(StatusCode::invalid_argument, luisa::string{"duplicate feature tensor '"} + tensor.name + "'", tensor.name);
        }
        uint64_t element_count = 0;
        if (!checked_element_count(tensor.shape, element_count) || element_count != tensor.element_count) {
            return R::failure(StatusCode::invalid_argument, "feature shape product mismatch", tensor.name);
        }
        uint64_t expected = 0;
        if (!checked_mul(element_count, feature_dtype_size(tensor.dtype), expected) || tensor.payload_size != expected) {
            return R::failure(StatusCode::invalid_argument, "feature payload size mismatch", tensor.name);
        }
        if (expected > 0) {
            uint64_t end = 0;
            if (!checked_add(tensor.payload_offset, expected, end) || end > file_size ||
                tensor.payload_offset < payload_base || (tensor.payload_offset % kFeatureBundleAlignment) != 0) {
                return R::failure(StatusCode::invalid_argument, "feature payload range out of bounds/misaligned", tensor.name);
            }
            ranges.emplace_back(Range{tensor.payload_offset, expected, tensor.name});
        } else if (tensor.payload_offset != 0 &&
                   (tensor.payload_offset < payload_base || tensor.payload_offset > meta_offset ||
                    (tensor.payload_offset % kFeatureBundleAlignment) != 0)) {
            // Same canonical zero-size rule as .lcocr: offset 0 OR an aligned
            // in-bounds position in the payload region.
            return R::failure(StatusCode::invalid_argument,
                              "empty feature payload offset is not 0 or an aligned in-bounds position", tensor.name);
        }
        reader->_tensors.emplace_back(std::move(tensor));
    }
    if (tpos != table_bytes.size()) {
        return R::failure(StatusCode::invalid_argument, "trailing bytes after the feature table", "table");
    }
    std::sort(ranges.begin(), ranges.end(), [](const Range &a, const Range &b) { return a.offset < b.offset; });
    uint64_t cursor = kFeatureBundleHeaderSize;
    for (const auto &range : ranges) {
        if (range.offset < cursor) {
            return R::failure(StatusCode::invalid_argument, luisa::string{"feature tensor '"} + range.owner + "' overlaps another", range.owner);
        }
        if (range.offset > cursor) {
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
                        return R::failure(StatusCode::invalid_argument, "nonzero feature padding", "padding");
                    }
                }
                pos += chunk;
            }
        }
        cursor = range.offset + range.size;
    }
    if (cursor != meta_offset) {
        return R::failure(StatusCode::invalid_argument, "trailing data after the last feature payload", "payload");
    }
    return R::success(std::move(reader));
}

[[nodiscard]] const FeatureTensor *FeatureBundleReader::find(luisa::string_view name) const noexcept {
    for (const auto &tensor : _tensors) {
        if (tensor.name == name) { return &tensor; }
    }
    return nullptr;
}

[[nodiscard]] Status FeatureBundleReader::read_tensor_bytes(const FeatureTensor &tensor, luisa::span<std::byte> out) {
    if (static_cast<uint64_t>(out.size()) != tensor.payload_size) {
        return Status::error(StatusCode::invalid_argument, "output span does not match the feature payload size", tensor.name);
    }
    return read_exact(tensor.payload_offset, out);
}

[[nodiscard]] Status FeatureBundleReader::read_tensor_f32(const FeatureTensor &tensor, luisa::span<float> out) {
    if (tensor.dtype != FeatureDtype::f32) {
        return Status::error(StatusCode::invalid_argument, "feature tensor is not f32", tensor.name);
    }
    if (static_cast<uint64_t>(out.size()) != tensor.element_count) {
        return Status::error(StatusCode::invalid_argument, "output span does not match the element count", tensor.name);
    }
    luisa::vector<std::byte> raw;
    raw.resize(static_cast<size_t>(tensor.payload_size));
    auto r = read_tensor_bytes(tensor, raw);
    if (!r.ok()) { return r; }
    for (size_t i = 0; i < out.size(); i++) {
        uint32_t bits = 0;
        for (uint32_t b = 0; b < 4u; b++) { bits |= static_cast<uint32_t>(raw[i * 4u + b]) << (8u * b); }
        std::memcpy(&out[i], &bits, sizeof(float));
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

FeatureBundleWriter::~FeatureBundleWriter() noexcept {
    if (_file != nullptr) { std::fclose(_file); }
}

[[nodiscard]] Result<luisa::unique_ptr<FeatureBundleWriter, FeatureBundleWriterDeleter>> FeatureBundleWriter::create(const luisa::filesystem::path &path) {
    using R = Result<luisa::unique_ptr<FeatureBundleWriter, FeatureBundleWriterDeleter>>;
    const auto path_text = luisa::to_string(path);
    std::FILE *file = std::fopen(path_text.c_str(), "wb");
    if (file == nullptr) {
        return R::failure(StatusCode::io_error, luisa::string{"cannot create feature bundle '"} + path_text + "'", "feature_bundle");
    }
    auto writer = luisa::unique_ptr<FeatureBundleWriter, FeatureBundleWriterDeleter>{new FeatureBundleWriter{}};
    writer->_file = file;
    writer->_path = path;
    std::byte header[kFeatureBundleHeaderSize] = {};
    if (std::fwrite(header, 1, sizeof(header), file) != sizeof(header)) {
        return R::failure(StatusCode::io_error, "cannot write feature header placeholder", "header");
    }
    writer->_offset = kFeatureBundleHeaderSize;
    return R::success(std::move(writer));
}

[[nodiscard]] Status FeatureBundleWriter::write_all(luisa::span<const std::byte> bytes) {
    if (bytes.empty()) { return Status::success(); }
    if (std::fwrite(bytes.data(), 1, bytes.size(), _file) != bytes.size()) {
        return Status::error(StatusCode::io_error, "short write to feature bundle", "write");
    }
    uint64_t next = 0;
    if (!checked_add(_offset, bytes.size(), next)) {
        return Status::error(StatusCode::invalid_argument, "feature bundle size overflow", "write");
    }
    _offset = next;
    return Status::success();
}

[[nodiscard]] Status FeatureBundleWriter::pad_to_alignment() {
    uint64_t aligned = 0;
    if (!checked_align_up(_offset, kFeatureBundleAlignment, aligned)) {
        return Status::error(StatusCode::invalid_argument, "alignment overflow", "align");
    }
    const auto padding = static_cast<size_t>(aligned - _offset);
    if (padding > 0) {
        std::byte zeros[kFeatureBundleAlignment] = {};
        auto w = write_all(luisa::span<const std::byte>{zeros, padding});
        if (!w.ok()) { return w; }
    }
    return Status::success();
}

[[nodiscard]] Status FeatureBundleWriter::add_tensor(const TensorSpec &spec) {
    if (_finished) { return Status::error(StatusCode::internal_error, "writer already finished", "writer"); }
    if (spec.name.empty() || spec.name.size() > 4096u || spec.shape.size() > kFeatureBundleMaxRank) {
        return Status::error(StatusCode::invalid_argument, "invalid feature tensor name/shape", spec.name);
    }
    for (const auto &t : _tensors) {
        if (t.name == spec.name) {
            return Status::error(StatusCode::invalid_argument, luisa::string{"duplicate feature tensor '"} + spec.name + "'", spec.name);
        }
    }
    FeatureTensor tensor;
    tensor.name = spec.name;
    tensor.dtype = spec.dtype;
    tensor.shape = spec.shape;
    if (!checked_element_count(spec.shape, tensor.element_count)) {
        return Status::error(StatusCode::invalid_argument, "feature shape product overflow", spec.name);
    }
    uint64_t expected = 0;
    if (!checked_mul(tensor.element_count, feature_dtype_size(spec.dtype), expected) || spec.payload.size() != expected) {
        return Status::error(StatusCode::invalid_argument, "feature payload size mismatch", spec.name);
    }
    auto pad = pad_to_alignment();
    if (!pad.ok()) { return pad; }
    if (_payload_base == 0) { _payload_base = _offset; }
    tensor.payload_offset = _offset;
    tensor.payload_size = spec.payload.size();
    auto w = write_all(spec.payload);
    if (!w.ok()) { return w; }
    _tensors.emplace_back(std::move(tensor));
    return Status::success();
}

[[nodiscard]] Status FeatureBundleWriter::finish(const FeatureBundleMeta &meta) {
    if (_finished) { return Status::error(StatusCode::internal_error, "writer already finished", "writer"); }
    auto meta_json = serialize_feature_meta(meta);
    if (!meta_json.ok()) { return meta_json.status; }
    const auto &json = *meta_json.value;
    const uint64_t meta_offset = _offset;
    auto w = write_all(luisa::span<const std::byte>{reinterpret_cast<const std::byte *>(json.data()), json.size()});
    if (!w.ok()) { return w; }
    const uint64_t table_offset = _offset;
    luisa::vector<std::byte> table;
    append_u64(table, _tensors.size());
    for (const auto &tensor : _tensors) {
        append_u32(table, static_cast<uint32_t>(tensor.name.size()));
        append_u32(table, static_cast<uint32_t>(tensor.dtype));
        append_u32(table, static_cast<uint32_t>(tensor.shape.size()));
        append_u32(table, 0u);
        append_u64(table, tensor.element_count);
        for (auto d : tensor.shape) { append_u64(table, d); }
        append_u64(table, tensor.payload_offset);
        append_u64(table, tensor.payload_size);
        for (auto c : tensor.name) { table.emplace_back(static_cast<std::byte>(c)); }
    }
    w = write_all(table);
    if (!w.ok()) { return w; }
    const uint64_t file_size = _offset;
    luisa::vector<std::byte> header;
    append_u32(header, kMagic);
    append_u32(header, kFeatureBundleVersion);
    append_u32(header, kFeatureBundleHeaderSize);
    append_u32(header, 0u);
    append_u64(header, meta_offset);
    append_u64(header, json.size());
    append_u64(header, table_offset);
    append_u64(header, table.size());
    append_u64(header, _payload_base == 0 ? file_size : _payload_base);
    append_u64(header, file_size);
    append_u64(header, 0u);
    auto seek = seek_to(_file, 0);
    if (!seek.ok()) { return seek; }
    if (std::fwrite(header.data(), 1, header.size(), _file) != header.size()) {
        return Status::error(StatusCode::io_error, "cannot rewrite feature header", "header");
    }
    if (std::fflush(_file) != 0) { return Status::error(StatusCode::io_error, "cannot flush feature bundle", "file"); }
    std::fclose(_file);
    _file = nullptr;
    _finished = true;
    auto reopened = FeatureBundleReader::open(_path);
    if (!reopened.ok()) { return reopened.status; }
    return Status::success();
}

}// namespace luisa::ocr