#pragma once
// OCR engine -- checked host-side safetensors reader (Phase 1).
// Host-only: no DSL/runtime device includes.
//
// This is an INSPECTION/conversion reader for packaging tools and C++ tests,
// not a production inference path (production loads validated .lcocr
// artifacts). It parses the header length + JSON metadata + tensor table,
// validates dtypes/ranks/shapes/offset ranges, duplicate names and file
// bounds, and exposes checked dtype views. Data bytes are read ON DEMAND;
// opening a multi-GB shard never copies the whole file into host memory.
//
// Supported inspection dtypes: F32, F16, BF16, I8, U8, I16, I32, I64, U16,
// U32, U64, BOOL. Unsupported dtypes (F8_*, F4, complex, ...) fail closed.
#include <cstdint>
#include <cstdio>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/quantization.h"
#include "ocr/status.h"

namespace luisa::ocr {

enum class SafetensorsDtype : uint32_t {
    f32 = 0,
    f16 = 1,
    bf16 = 2,
    i8 = 3,
    u8 = 4,
    i16 = 5,
    i32 = 6,
    i64 = 7,
    u16 = 8,
    u32 = 9,
    u64 = 10,
    boolean = 11,
};

[[nodiscard]] luisa::string_view safetensors_dtype_name(SafetensorsDtype dtype) noexcept;
[[nodiscard]] uint64_t safetensors_dtype_size(SafetensorsDtype dtype) noexcept;

inline constexpr uint32_t kSafetensorsMaxRank = 8u;
inline constexpr uint64_t kSafetensorsMaxHeaderBytes = 100ull * 1024ull * 1024ull;
inline constexpr uint64_t kSafetensorsMaxNameBytes = 4096u;

struct SafetensorsTensor {
    luisa::string name;
    SafetensorsDtype dtype{SafetensorsDtype::f32};
    luisa::vector<uint64_t> shape;
    uint64_t element_count{0};
    // Absolute byte range within the file (data_start + relative offsets).
    uint64_t data_offset{0};
    uint64_t data_size{0};
};

class SafetensorsReader;
struct SafetensorsReaderDeleter {
    void operator()(SafetensorsReader *reader) const noexcept;
};

class SafetensorsReader {
public:
    [[nodiscard]] static Result<luisa::unique_ptr<SafetensorsReader, SafetensorsReaderDeleter>> open(const luisa::filesystem::path &path);
    ~SafetensorsReader() noexcept;
    SafetensorsReader(const SafetensorsReader &) = delete;
    SafetensorsReader &operator=(const SafetensorsReader &) = delete;

    [[nodiscard]] luisa::span<const SafetensorsTensor> tensors() const noexcept { return _tensors; }
    [[nodiscard]] const SafetensorsTensor *find(luisa::string_view name) const noexcept;
    [[nodiscard]] uint64_t file_size() const noexcept { return _file_size; }
    // The "__metadata__" string map (format / producer hints), may be empty.
    [[nodiscard]] const luisa::vector<std::pair<luisa::string, luisa::string>> &metadata() const noexcept { return _metadata; }

    // Read the raw stored bytes of one tensor (no dtype conversion).
    [[nodiscard]] Status read_tensor_bytes(const SafetensorsTensor &tensor, luisa::span<std::byte> out);
    // Decode a float tensor (F32/F16/BF16) to FP32 with explicit conversion.
    // Non-float dtypes are rejected (no silent reinterpretation).
    [[nodiscard]] Status read_tensor_f32(const SafetensorsTensor &tensor, luisa::span<float> out);

private:
    SafetensorsReader() noexcept = default;
    [[nodiscard]] Status read_exact(uint64_t offset, luisa::span<std::byte> out);

    std::FILE *_file{nullptr};
    uint64_t _file_size{0};
    uint64_t _data_start{0};
    luisa::vector<SafetensorsTensor> _tensors;
    luisa::vector<std::pair<luisa::string, luisa::string>> _metadata;
};

}// namespace luisa::ocr