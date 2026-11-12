#pragma once
// OCR engine -- .ocrfeat feature bundle interchange (Phase 1). Host-only.
//
// A versioned, checked binary interchange for official-AutoProcessor feature
// differential tests (produced by tools/preprocess.py). It is a DEBUG path,
// not required for ordinary input. Format v1:
//
//   [0..64)  header: magic "OCRF" (u32), u32 version=1, u32 header_size=64,
//            u32 reserved(0), u64 meta_offset, u64 meta_size,
//            u64 tensor_table_offset, u64 tensor_table_size,
//            u64 payload_base, u64 file_size.
//   meta     UTF-8 JSON: producer, processor identity/revision, source image
//            hash, deterministic decode settings.
//   table    u64 count, then per tensor: u32 name_size, u32 dtype, u32 rank,
//            u32 reserved(0), u64 element_count, u64 shape[rank],
//            u64 payload_offset, u64 payload_size, name bytes.
//   payloads 256-byte aligned, zero padding, no overlap/trailing data.
//
// Reuses the same checked range/overlap discipline as the .lcocr package.
#include <cstdint>
#include <cstdio>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/quantization.h"
#include "ocr/status.h"

namespace luisa::ocr {

inline constexpr uint32_t kFeatureBundleVersion = 1u;
inline constexpr uint32_t kFeatureBundleHeaderSize = 64u;
inline constexpr uint64_t kFeatureBundleAlignment = 256u;
inline constexpr uint32_t kFeatureBundleMaxRank = 8u;

enum class FeatureDtype : uint32_t {
    f32 = 0,
    i32 = 1,
    i64 = 2,
    u8 = 3,
};

[[nodiscard]] luisa::string_view feature_dtype_name(FeatureDtype dtype) noexcept;
[[nodiscard]] uint64_t feature_dtype_size(FeatureDtype dtype) noexcept;

struct FeatureBundleMeta {
    luisa::string producer;
    luisa::string processor;         // official processor identity
    luisa::string processor_revision;
    luisa::string source_image_hash;
    luisa::string decode_settings;   // deterministic decode settings (JSON text)
};

[[nodiscard]] Result<luisa::string> serialize_feature_meta(const FeatureBundleMeta &meta);
[[nodiscard]] Result<FeatureBundleMeta> parse_feature_meta(luisa::string_view json);

struct FeatureTensor {
    luisa::string name;
    FeatureDtype dtype{FeatureDtype::f32};
    luisa::vector<uint64_t> shape;
    uint64_t element_count{0};
    uint64_t payload_offset{0};
    uint64_t payload_size{0};
};

class FeatureBundleReader;
struct FeatureBundleReaderDeleter {
    void operator()(FeatureBundleReader *reader) const noexcept;
};

class FeatureBundleReader {
public:
    [[nodiscard]] static Result<luisa::unique_ptr<FeatureBundleReader, FeatureBundleReaderDeleter>> open(const luisa::filesystem::path &path);
    ~FeatureBundleReader() noexcept;
    FeatureBundleReader(const FeatureBundleReader &) = delete;
    FeatureBundleReader &operator=(const FeatureBundleReader &) = delete;

    [[nodiscard]] const FeatureBundleMeta &meta() const noexcept { return _meta; }
    [[nodiscard]] luisa::span<const FeatureTensor> tensors() const noexcept { return _tensors; }
    [[nodiscard]] const FeatureTensor *find(luisa::string_view name) const noexcept;
    [[nodiscard]] Status read_tensor_bytes(const FeatureTensor &tensor, luisa::span<std::byte> out);
    [[nodiscard]] Status read_tensor_f32(const FeatureTensor &tensor, luisa::span<float> out);

private:
    FeatureBundleReader() noexcept = default;
    [[nodiscard]] Status read_exact(uint64_t offset, luisa::span<std::byte> out);
    std::FILE *_file{nullptr};
    uint64_t _file_size{0};
    FeatureBundleMeta _meta;
    luisa::vector<FeatureTensor> _tensors;
};

class FeatureBundleWriter;
struct FeatureBundleWriterDeleter {
    void operator()(FeatureBundleWriter *writer) const noexcept;
};

class FeatureBundleWriter {
public:
    struct TensorSpec {
        luisa::string name;
        FeatureDtype dtype{FeatureDtype::f32};
        luisa::vector<uint64_t> shape;
        luisa::vector<std::byte> payload;
    };
    [[nodiscard]] static Result<luisa::unique_ptr<FeatureBundleWriter, FeatureBundleWriterDeleter>> create(const luisa::filesystem::path &path);
    ~FeatureBundleWriter() noexcept;
    FeatureBundleWriter(const FeatureBundleWriter &) = delete;
    FeatureBundleWriter &operator=(const FeatureBundleWriter &) = delete;
    [[nodiscard]] Status add_tensor(const TensorSpec &spec);
    [[nodiscard]] Status finish(const FeatureBundleMeta &meta);

private:
    FeatureBundleWriter() noexcept = default;
    [[nodiscard]] Status write_all(luisa::span<const std::byte> bytes);
    [[nodiscard]] Status pad_to_alignment();
    std::FILE *_file{nullptr};
    luisa::filesystem::path _path;
    uint64_t _offset{0};
    uint64_t _payload_base{0};
    luisa::vector<FeatureTensor> _tensors;
    bool _finished{false};
};

}// namespace luisa::ocr