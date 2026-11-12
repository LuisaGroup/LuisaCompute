#pragma once
// OCR engine -- versioned .lcocr v1 model package (Phase 1).
// Host-only: no DSL/runtime device includes. See FORMAT.md for the full
// on-disk specification; the layout in brief:
//
//   [0..64)     fixed little-endian header:
//               magic "LCOCR1\0\0" (8 bytes), u32 format_version = 1,
//               u32 header_size = 64, u64 manifest_offset, u64 manifest_size,
//               u64 table_offset, u64 table_size,
//               u64 payload_base, u64 file_size, u64 reserved (0).
//   manifest    UTF-8 JSON (identity/dependencies/versions; see PackageManifest)
//   table       u64 tensor_count, then per-tensor fixed records + name bytes:
//               u32 name_size, u32 rank, u32 encoding, u32 layout,
//               u32 quant_kind, u32 group_size, u32 reserved (0),
//               u64 element_count, u64 shape[rank], u64 strides[rank]
//               (element strides, row-major), u64 payload_offset (absolute),
//               u64 payload_size, u64 scales_offset, u64 scales_size,
//               u64 zero_points_offset, u64 zero_points_size,
//               name bytes (UTF-8, no NUL).
//   payloads    each blob starts at a 256-byte aligned absolute offset;
//               padding bytes must be zero; no overlap; no trailing data.
//
// Every offset/size is 64-bit and validated with checked arithmetic against
// the actual file size. Tensor lookup is by NAME. Rank is arbitrary within
// kPackageMaxRank (8). Encodings: FP32/FP16/BF16 dense, grouped INT4
// (symmetric/asymmetric) and grouped INT8 (see quantization.h). FP16 is never
// reinterpreted as BF16 and INT8 is never reinterpreted as INT4.
//
// The reader parses header/manifest/table eagerly but reads payload bytes ON
// DEMAND (read_tensor_payload), so opening a multi-GB package never copies
// the whole file into host memory. PackageWriter streams tensors to disk in
// the same order, used by fixtures/tools.
#include <cstdint>
#include <cstdio>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/quantization.h"
#include "ocr/status.h"

namespace luisa::ocr {

inline constexpr uint32_t kPackageFormatVersion = 1u;
inline constexpr uint32_t kPackageHeaderSize = 64u;
inline constexpr uint32_t kPackageMaxRank = 8u;
inline constexpr uint64_t kPackagePayloadAlignment = 256u;
inline constexpr uint64_t kPackageMaxManifestBytes = 16ull * 1024ull * 1024ull;
inline constexpr uint64_t kPackageMaxTableBytes = 256ull * 1024ull * 1024ull;
inline constexpr uint64_t kPackageMaxNameBytes = 4096u;
inline constexpr uint64_t kPackageMaxTensors = 1ull << 20ull;

// Storage encodings (table `encoding` field).
enum class TensorEncoding : uint32_t {
    fp32 = 0,
    fp16 = 1,
    bf16 = 2,
    int4_packed = 3,// grouped; quant_kind selects symmetric/asymmetric
    int8 = 4,       // grouped symmetric
};

[[nodiscard]] luisa::string_view tensor_encoding_name(TensorEncoding encoding) noexcept;
[[nodiscard]] bool is_quantized_encoding(TensorEncoding encoding) noexcept;
// Bytes per stored element for dense encodings (0 for quantized kinds).
[[nodiscard]] uint64_t dense_encoding_element_size(TensorEncoding encoding) noexcept;

// Documented layout tags. Strides are always stored explicitly; the tag
// records the semantic convention (e.g. "kn" = linear weight [K, N]).
enum class TensorLayout : uint32_t {
    row_major = 0,
    kn = 1,       // packed linear weight, [K, N]
    nk = 2,       // transposed linear weight, [N, K]
    conv_oihw = 3,// [out, in, kh, kw]
    conv_ohwi = 4,// [out, kh, kw, in]
};

[[nodiscard]] luisa::string_view tensor_layout_name(TensorLayout layout) noexcept;
[[nodiscard]] luisa::optional<TensorLayout> tensor_layout_from_name(luisa::string_view name) noexcept;

// One source artifact hash (path + digest), recorded in the manifest.
struct SourceHash {
    luisa::string path;
    luisa::string algorithm;// e.g. "sha256"
    luisa::string digest;
};

// Component dependency: another .lcocr package referenced by relative path.
struct ComponentDependency {
    luisa::string component;// role, e.g. "layout", "vision", "decoder"
    luisa::string package;  // relative path to the component package
    luisa::string sha256;   // optional expected digest ("" = not pinned)
};

struct PackageManifest {
    luisa::string model;             // semantic target selector ("glm_ocr", "pp_ocrv6:medium", ...)
    luisa::string recipe_version;
    luisa::string checkpoint_revision;
    luisa::string config_fingerprint;
    luisa::string license;           // SPDX identifier or note
    luisa::string producer;          // producer tool identity
    luisa::string producer_version;
    luisa::string preprocessing_version;
    luisa::string tokenizer_version;
    luisa::string output_parser_version;
    luisa::string component;         // this package's component role ("" = whole model)
    luisa::vector<ComponentDependency> dependencies;
    luisa::vector<SourceHash> source_hashes;

    [[nodiscard]] Status validate() const;
};

// Serialize/parse the manifest JSON (yyjson). Parsing is strict: unknown
// top-level keys are rejected, required identity fields must be present.
[[nodiscard]] Result<luisa::string> serialize_package_manifest(const PackageManifest &manifest);
[[nodiscard]] Result<PackageManifest> parse_package_manifest(luisa::string_view json);

// One tensor table entry (host metadata; payload bytes are read on demand).
struct PackageTensor {
    luisa::string name;
    TensorEncoding encoding{TensorEncoding::fp32};
    TensorLayout layout{TensorLayout::row_major};
    QuantKind quant_kind{QuantKind::none};
    uint32_t group_size{0};
    uint64_t element_count{0};
    luisa::vector<uint64_t> shape;
    luisa::vector<uint64_t> strides;// element strides, row-major
    uint64_t payload_offset{0};
    uint64_t payload_size{0};
    uint64_t scales_offset{0};
    uint64_t scales_size{0};
    uint64_t zero_points_offset{0};
    uint64_t zero_points_size{0};
};

// Reader for one .lcocr file. Owns an open read handle; payloads are read on
// demand into caller-provided storage (no whole-file host duplication).
//
// Ownership: instances are created by open() and must be destroyed through
// PackageReaderDeleter (the return type's deleter). The deleter runs delete in
// this library's TU so allocation and deallocation use the same allocator even
// when the caller links the static library into a separate module.
class PackageReader;
struct PackageReaderDeleter {
    void operator()(PackageReader *reader) const noexcept;
};

class PackageReader {
public:
    [[nodiscard]] static Result<luisa::unique_ptr<PackageReader, PackageReaderDeleter>> open(const luisa::filesystem::path &path);
    ~PackageReader() noexcept;
    PackageReader(const PackageReader &) = delete;
    PackageReader &operator=(const PackageReader &) = delete;

    [[nodiscard]] const PackageManifest &manifest() const noexcept { return _manifest; }
    [[nodiscard]] luisa::span<const PackageTensor> tensors() const noexcept { return _tensors; }
    [[nodiscard]] uint64_t file_size() const noexcept { return _file_size; }
    [[nodiscard]] const PackageTensor *find(luisa::string_view name) const noexcept;

    // Read the raw payload bytes of one tensor. `out.size()` must equal the
    // recorded payload_size. Quantized payloads are returned as stored (codes
    // / scales / zero points), never silently converted.
    [[nodiscard]] Status read_tensor_payload(const PackageTensor &tensor, luisa::span<std::byte> out);
    [[nodiscard]] Status read_tensor_scales(const PackageTensor &tensor, luisa::span<float> out);
    [[nodiscard]] Status read_tensor_zero_points(const PackageTensor &tensor, luisa::span<uint8_t> out);

    // Convenience: decode a dense FP32/FP16/BF16 tensor to FP32 (explicit
    // conversion, never reinterpretation). Quantized tensors are rejected;
    // use the quantization codecs with the metadata instead.
    [[nodiscard]] Status read_tensor_f32(const PackageTensor &tensor, luisa::span<float> out);
    // Convenience: dequantize an INT4/INT8 tensor to FP32 using its metadata.
    [[nodiscard]] Status read_tensor_dequantized(const PackageTensor &tensor, luisa::span<float> out);

private:
    PackageReader() noexcept = default;
    [[nodiscard]] Status read_exact(uint64_t offset, luisa::span<std::byte> out);

    std::FILE *_file{nullptr};
    uint64_t _file_size{0};
    PackageManifest _manifest;
    luisa::vector<PackageTensor> _tensors;
};

// Streaming writer: add tensors one at a time; payload bytes are written to
// disk immediately (no whole-model host buffer). finish() writes the manifest
// and table and validates the result by reopening it. Destroy through
// PackageWriterDeleter (same cross-module allocator contract as the reader).
class PackageWriter;
struct PackageWriterDeleter {
    void operator()(PackageWriter *writer) const noexcept;
};

class PackageWriter {
public:
    struct TensorSpec {
        luisa::string name;
        TensorEncoding encoding{TensorEncoding::fp32};
        TensorLayout layout{TensorLayout::row_major};
        luisa::vector<uint64_t> shape;
        // Quantization metadata (quantized encodings only).
        QuantKind quant_kind{QuantKind::none};
        uint32_t group_size{0};
        luisa::vector<float> scales;
        luisa::vector<uint8_t> zero_points;
        // Raw stored payload bytes (dense values or packed codes).
        luisa::vector<std::byte> payload;
    };

    [[nodiscard]] static Result<luisa::unique_ptr<PackageWriter, PackageWriterDeleter>> create(const luisa::filesystem::path &path);
    ~PackageWriter() noexcept;
    PackageWriter(const PackageWriter &) = delete;
    PackageWriter &operator=(const PackageWriter &) = delete;

    // Appends one tensor; the payload is streamed to disk immediately.
    [[nodiscard]] Status add_tensor(const TensorSpec &spec);
    // Writes manifest + table, closes the file, and reopens it with
    // PackageReader to prove the artifact is well-formed.
    [[nodiscard]] Status finish(const PackageManifest &manifest);

private:
    PackageWriter() noexcept = default;
    [[nodiscard]] Status write_all(luisa::span<const std::byte> bytes);
    [[nodiscard]] Status pad_to_alignment();

    std::FILE *_file{nullptr};
    luisa::filesystem::path _path;
    uint64_t _offset{0};
    uint64_t _payload_base{0};
    luisa::vector<PackageTensor> _tensors;
    bool _finished{false};
};

}// namespace luisa::ocr