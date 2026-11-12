// OCR engine -- checked host-side safetensors reader (Phase 1).
// Strict fail-closed parsing; data bytes are read on demand. No exceptions.
#include "ocr/safetensors_reader.h"

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

struct JsonDoc {
    yyjson_doc *doc{nullptr};
    ~JsonDoc() noexcept {
        if (doc != nullptr) { yyjson_doc_free(doc); }
    }
};

[[nodiscard]] Result<uint64_t> file_size_of(std::FILE *file) {
#ifdef _WIN32
    const auto size = _filelengthi64(_fileno(file));
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
        return Status::error(StatusCode::io_error, "cannot seek in safetensors file", "file");
    }
    return Status::success();
}

[[nodiscard]] luisa::optional<SafetensorsDtype> dtype_from_name(luisa::string_view name) noexcept {
    if (name == "F32") { return SafetensorsDtype::f32; }
    if (name == "F16") { return SafetensorsDtype::f16; }
    if (name == "BF16") { return SafetensorsDtype::bf16; }
    if (name == "I8") { return SafetensorsDtype::i8; }
    if (name == "U8") { return SafetensorsDtype::u8; }
    if (name == "I16") { return SafetensorsDtype::i16; }
    if (name == "I32") { return SafetensorsDtype::i32; }
    if (name == "I64") { return SafetensorsDtype::i64; }
    if (name == "U16") { return SafetensorsDtype::u16; }
    if (name == "U32") { return SafetensorsDtype::u32; }
    if (name == "U64") { return SafetensorsDtype::u64; }
    if (name == "BOOL") { return SafetensorsDtype::boolean; }
    return luisa::nullopt;
}

}// namespace

void SafetensorsReaderDeleter::operator()(SafetensorsReader *reader) const noexcept { delete reader; }

[[nodiscard]] luisa::string_view safetensors_dtype_name(SafetensorsDtype dtype) noexcept {
    switch (dtype) {
        case SafetensorsDtype::f32: return "F32";
        case SafetensorsDtype::f16: return "F16";
        case SafetensorsDtype::bf16: return "BF16";
        case SafetensorsDtype::i8: return "I8";
        case SafetensorsDtype::u8: return "U8";
        case SafetensorsDtype::i16: return "I16";
        case SafetensorsDtype::i32: return "I32";
        case SafetensorsDtype::i64: return "I64";
        case SafetensorsDtype::u16: return "U16";
        case SafetensorsDtype::u32: return "U32";
        case SafetensorsDtype::u64: return "U64";
        case SafetensorsDtype::boolean: return "BOOL";
        default: return "unknown";
    }
}

[[nodiscard]] uint64_t safetensors_dtype_size(SafetensorsDtype dtype) noexcept {
    switch (dtype) {
        case SafetensorsDtype::f32:
        case SafetensorsDtype::i32:
        case SafetensorsDtype::u32: return 4;
        case SafetensorsDtype::f16:
        case SafetensorsDtype::bf16:
        case SafetensorsDtype::i16:
        case SafetensorsDtype::u16: return 2;
        case SafetensorsDtype::i64:
        case SafetensorsDtype::u64: return 8;
        case SafetensorsDtype::i8:
        case SafetensorsDtype::u8:
        case SafetensorsDtype::boolean: return 1;
        default: return 0;
    }
}

SafetensorsReader::~SafetensorsReader() noexcept {
    if (_file != nullptr) { std::fclose(_file); }
}

[[nodiscard]] Status SafetensorsReader::read_exact(uint64_t offset, luisa::span<std::byte> out) {
    if (out.empty()) { return Status::success(); }
    uint64_t end = 0;
    if (!checked_add(offset, out.size(), end) || end > _file_size) {
        return Status::error(StatusCode::io_error, "read range exceeds the safetensors file size", "range");
    }
    auto seek = seek_to(_file, offset);
    if (!seek.ok()) { return seek; }
    size_t remaining = out.size();
    size_t pos = 0;
    while (remaining > 0) {
        const size_t chunk = remaining > (1u << 20u) ? (1u << 20u) : remaining;
        const size_t got = std::fread(out.data() + pos, 1, chunk, _file);
        if (got == 0) {
            return Status::error(StatusCode::io_error, "unexpected end of safetensors file", "read");
        }
        pos += got;
        remaining -= got;
    }
    return Status::success();
}

[[nodiscard]] Result<luisa::unique_ptr<SafetensorsReader, SafetensorsReaderDeleter>> SafetensorsReader::open(const luisa::filesystem::path &path) {
    using R = Result<luisa::unique_ptr<SafetensorsReader, SafetensorsReaderDeleter>>;
    const auto path_text = luisa::to_string(path);
    std::FILE *file = std::fopen(path_text.c_str(), "rb");
    if (file == nullptr) {
        return R::failure(StatusCode::not_found,
                          luisa::string{"cannot open safetensors file '"} + path_text + "'", "safetensors");
    }
    auto reader = luisa::unique_ptr<SafetensorsReader, SafetensorsReaderDeleter>{new SafetensorsReader{}};
    reader->_file = file;
    auto size = file_size_of(file);
    if (!size.ok()) { return R::failure(size.status); }
    reader->_file_size = *size.value;
    if (reader->_file_size < 8) {
        return R::failure(StatusCode::invalid_argument, "safetensors file is smaller than the header length", "header");
    }

    // Header length (u64 little-endian).
    std::byte length_bytes[8];
    auto hr = reader->read_exact(0, length_bytes);
    if (!hr.ok()) { return R::failure(hr); }
    uint64_t header_length = 0;
    for (uint32_t i = 0; i < 8u; i++) {
        header_length |= static_cast<uint64_t>(length_bytes[i]) << (8u * i);
    }
    if (header_length == 0 || header_length > kSafetensorsMaxHeaderBytes) {
        return R::failure(StatusCode::invalid_argument, "safetensors header length is zero or exceeds the limit", "header.length");
    }
    uint64_t data_start = 0;
    if (!checked_add(8u, header_length, data_start) || data_start > reader->_file_size) {
        return R::failure(StatusCode::invalid_argument, "safetensors header exceeds the file size", "header");
    }
    reader->_data_start = data_start;

    luisa::vector<std::byte> header_bytes;
    header_bytes.resize(static_cast<size_t>(header_length));
    hr = reader->read_exact(8, header_bytes);
    if (!hr.ok()) { return R::failure(hr); }

    JsonDoc holder;
    yyjson_read_err err;
    holder.doc = yyjson_read_opts(reinterpret_cast<char *>(header_bytes.data()), header_bytes.size(),
                                  YYJSON_READ_NOFLAG, nullptr, &err);
    if (holder.doc == nullptr) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::string{"safetensors header JSON parse error: "} + (err.msg != nullptr ? err.msg : "unknown"), "header");
    }
    auto *root = yyjson_doc_get_root(holder.doc);
    if (!yyjson_is_obj(root)) {
        return R::failure(StatusCode::invalid_argument, "safetensors header root must be an object", "header");
    }

    struct Range {
        uint64_t begin;
        uint64_t end;
        luisa::string owner;
    };
    luisa::vector<Range> ranges;
    luisa::unordered_map<luisa::string, uint32_t> names;

    yyjson_val *key;
    yyjson_val *val;
    size_t idx, max;
    yyjson_obj_foreach(root, idx, max, key, val) {
        const luisa::string_view name{yyjson_get_str(key), yyjson_get_len(key)};
        if (name == "__metadata__") {
            if (!yyjson_is_obj(val)) {
                return R::failure(StatusCode::invalid_argument, "__metadata__ must be an object", "__metadata__");
            }
            yyjson_val *mk;
            yyjson_val *mv;
            size_t mi, mm;
            yyjson_obj_foreach(val, mi, mm, mk, mv) {
                if (!yyjson_is_str(mv)) {
                    return R::failure(StatusCode::invalid_argument, "__metadata__ values must be strings", "__metadata__");
                }
                reader->_metadata.emplace_back(
                    luisa::string{yyjson_get_str(mk), yyjson_get_len(mk)},
                    luisa::string{yyjson_get_str(mv), yyjson_get_len(mv)});
            }
            continue;
        }
        if (name.empty() || name.size() > kSafetensorsMaxNameBytes) {
            return R::failure(StatusCode::invalid_argument, "tensor name is empty or exceeds the limit", "name");
        }
        if (++names[luisa::string{name}] > 1u) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::string{"duplicate tensor name '"} + luisa::string{name} + "'", luisa::string{name});
        }
        if (!yyjson_is_obj(val)) {
            return R::failure(StatusCode::invalid_argument, "tensor entry must be an object", luisa::string{name});
        }
        auto *dtype_val = yyjson_obj_get(val, "dtype");
        auto *shape_val = yyjson_obj_get(val, "shape");
        auto *offsets_val = yyjson_obj_get(val, "data_offsets");
        if (dtype_val == nullptr || !yyjson_is_str(dtype_val)) {
            return R::failure(StatusCode::invalid_argument, "tensor entry is missing a string dtype", luisa::string{name});
        }
        const auto dtype = dtype_from_name(luisa::string_view{yyjson_get_str(dtype_val), yyjson_get_len(dtype_val)});
        if (!dtype.has_value()) {
            return R::failure(StatusCode::unsupported,
                              luisa::string{"unsupported safetensors dtype '"} + yyjson_get_str(dtype_val) + "' (repack required)", luisa::string{name});
        }
        if (shape_val == nullptr || !yyjson_is_arr(shape_val)) {
            return R::failure(StatusCode::invalid_argument, "tensor entry is missing a shape array", luisa::string{name});
        }
        if (yyjson_arr_size(shape_val) > kSafetensorsMaxRank) {
            return R::failure(StatusCode::unsupported, "tensor rank exceeds the documented maximum (8)", luisa::string{name});
        }
        if (offsets_val == nullptr || !yyjson_is_arr(offsets_val) || yyjson_arr_size(offsets_val) != 2) {
            return R::failure(StatusCode::invalid_argument, "data_offsets must be a [begin, end] array", luisa::string{name});
        }
        auto *begin_val = yyjson_arr_get(offsets_val, 0);
        auto *end_val = yyjson_arr_get(offsets_val, 1);
        if (!yyjson_is_uint(begin_val) || !yyjson_is_uint(end_val)) {
            return R::failure(StatusCode::invalid_argument, "data_offsets must be unsigned integers", luisa::string{name});
        }
        const uint64_t rel_begin = yyjson_get_uint(begin_val);
        const uint64_t rel_end = yyjson_get_uint(end_val);
        if (rel_end < rel_begin) {
            return R::failure(StatusCode::invalid_argument, "data_offsets are reversed", luisa::string{name});
        }
        uint64_t abs_begin = 0, abs_end = 0;
        if (!checked_add(data_start, rel_begin, abs_begin) ||
            !checked_add(data_start, rel_end, abs_end) || abs_end > reader->_file_size) {
            return R::failure(StatusCode::invalid_argument, "tensor data range exceeds the file size", luisa::string{name});
        }

        SafetensorsTensor tensor;
        tensor.name = luisa::string{name};
        tensor.dtype = *dtype;
        uint64_t count = 1;
        yyjson_val *dim;
        size_t di, dm;
        yyjson_arr_foreach(shape_val, di, dm, dim) {
            if (!yyjson_is_uint(dim)) {
                return R::failure(StatusCode::invalid_argument, "shape dimensions must be unsigned integers", tensor.name);
            }
            const uint64_t d = yyjson_get_uint(dim);
            if (!checked_mul(count, d, count)) {
                return R::failure(StatusCode::invalid_argument, "shape product overflow", tensor.name);
            }
            tensor.shape.emplace_back(d);
        }
        tensor.element_count = count;
        uint64_t expected_size = 0;
        if (!checked_mul(count, safetensors_dtype_size(*dtype), expected_size)) {
            return R::failure(StatusCode::invalid_argument, "tensor byte size overflow", tensor.name);
        }
        if (rel_end - rel_begin != expected_size) {
            return R::failure(StatusCode::invalid_argument,
                              "tensor data size does not match shape * dtype size", tensor.name);
        }
        tensor.data_offset = abs_begin;
        tensor.data_size = expected_size;
        if (expected_size > 0) {
            ranges.emplace_back(Range{abs_begin, abs_end, tensor.name});
        }
        reader->_tensors.emplace_back(std::move(tensor));
    }

    // Overlap + gap + trailing-data checks over the data section.
    std::sort(ranges.begin(), ranges.end(), [](const Range &a, const Range &b) { return a.begin < b.begin; });
    uint64_t cursor = data_start;
    for (const auto &range : ranges) {
        if (range.begin < cursor) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::string{"tensor '"} + range.owner + "' data range overlaps another tensor", range.owner);
        }
        if (range.begin > cursor) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::string{"unaccounted gap before tensor '"} + range.owner + "'", range.owner);
        }
        cursor = range.end;
    }
    if (cursor != reader->_file_size) {
        return R::failure(StatusCode::invalid_argument, "trailing data after the last tensor", "data");
    }
    return R::success(std::move(reader));
}

[[nodiscard]] const SafetensorsTensor *SafetensorsReader::find(luisa::string_view name) const noexcept {
    for (const auto &tensor : _tensors) {
        if (tensor.name == name) { return &tensor; }
    }
    return nullptr;
}

[[nodiscard]] Status SafetensorsReader::read_tensor_bytes(const SafetensorsTensor &tensor, luisa::span<std::byte> out) {
    if (static_cast<uint64_t>(out.size()) != tensor.data_size) {
        return Status::error(StatusCode::invalid_argument, "output span does not match the tensor byte size", tensor.name);
    }
    return read_exact(tensor.data_offset, out);
}

[[nodiscard]] Status SafetensorsReader::read_tensor_f32(const SafetensorsTensor &tensor, luisa::span<float> out) {
    if (tensor.dtype != SafetensorsDtype::f32 && tensor.dtype != SafetensorsDtype::f16 &&
        tensor.dtype != SafetensorsDtype::bf16) {
        return Status::error(StatusCode::invalid_argument,
                             "read_tensor_f32 requires an F32/F16/BF16 tensor (no reinterpretation)", tensor.name);
    }
    if (static_cast<uint64_t>(out.size()) != tensor.element_count) {
        return Status::error(StatusCode::invalid_argument, "output span does not match the element count", tensor.name);
    }
    luisa::vector<std::byte> raw;
    raw.resize(static_cast<size_t>(tensor.data_size));
    auto r = read_tensor_bytes(tensor, raw);
    if (!r.ok()) { return r; }
    if (tensor.dtype == SafetensorsDtype::f32) {
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
            out[i] = tensor.dtype == SafetensorsDtype::f16 ? fp16_to_fp32(bits) : bf16_to_fp32(bits);
        }
    }
    return Status::success();
}

}// namespace luisa::ocr