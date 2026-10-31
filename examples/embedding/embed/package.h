// bge-m3 packed-model loader interface (embedding_model_plan.md §3.3, milestone M2
// loader half). PARENT-OWNED INTERFACE: B3 implements package.cpp; B1 codes the
// Session/executor against THIS header only.
//
// The on-disk format is model.bgm VERSION 1, specified in the header comment of
// examples/embedding/tools/pack_model.cpp (read it — magic "BGM3", table of
// {fnv1a64(name), offset, size, rank, dims[4]}, 256-byte aligned fp32 blobs,
// canonical names "emb.word", "layer{i}.qkv", ..., "head.colbert.b").
//
// Key facts the implementation MUST honor (verified by the packer, plan §0/§3.3):
// - "emb.word" already has the token-type embedding row 0 folded in.
// - Position embeddings are a raw table; the ONNX graph uses cumsum-based
//   positions with XLM-R's +2 offset (first real token = row 2 of "emb.pos").
//   The executor/kernels own that indexing; the loader just serves the table.
// - VERSION 1 packages: precision must be 0 (fp32); 1/2 are rejected in v1.
// - VERSION 2 packages (optimization plan §4 Q1/Q2, O-G): precision 1 = bf16,
//   2 = fp16; every blob stores 2-byte little-endian elements (round-to-
//   nearest-even from fp32 at pack time), size_bytes = prod(dims)*2. Names,
//   order, dims, alignment and the table layout are identical to v1. The v1
//   fp32 format stays byte-identical and is loaded by the same code path.
// - VERSION 3 packages (INT8 quantization, phase 1): precision 3 = per-tensor
//   symmetric int8. A blob is exactly prod(dims) signed int8 bytes (no per-blob
//   header) and the tensor's positive fp32 scale occupies the reserved trailing
//   4 bytes (entry offset 44) of its 48-byte table entry. Names, order, dims,
//   alignment and the table layout are again identical to v1/v2; v1/v2 entries
//   keep the reserved bytes zero. The scale is validated finite and > 0 on load
//   (fail-closed) and the math lives in embed/quantization.h.
#pragma once

#include <cstdint>
#include <filesystem>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::embed {

// Package element format, mirroring the header `precision` field (opt plan
// §4 Q1/Q2). fp32 == BGM3 v1; bf16/fp16 == BGM3 v2; int8 == BGM3 v3
// (per-tensor symmetric int8, see embed/quantization.h).
enum class PackagePrecision : uint32_t {
    fp32 = 0u,
    bf16 = 1u,
    fp16 = 2u,
    int8 = 3u,
};
// "fp32" / "bf16" / "fp16" / "int8" (log/test reporting).
[[nodiscard]] const char *package_precision_name(PackagePrecision p) noexcept;

// One tensor descriptor from the bgm table. The typed views are host-side
// pointers valid for the lifetime of the Package (backed by the loaded blob):
// exactly one of them is non-null, according to the package precision.
// - fp32 package: `data` is the float blob (as before, bit-identical behavior).
// - bf16/fp16 package: `half_data` carries the uint16 blob, `data` is null.
// - int8 package: `int8_data` carries the signed int8 blob and `int8_scale` the
//   per-tensor scale read from the table entry's reserved trailing 4 bytes;
//   `data` and `half_data` are both null (typed apart so a consumer cannot
//   mistake one format for another). Dequantization: q * int8_scale, see
//   embed/quantization.h.
// Fields not used by the package precision stay at their defaults (null / 0).
struct Tensor {
    const float *data{};
    const uint16_t *half_data{};
    const int8_t *int8_data{};
    // Per-tensor INT8 scale (v3 only): always finite and > 0 when precision()
    // is int8, 0 otherwise. Present in every entry so a consumer never needs a
    // parallel array.
    float int8_scale{};
    uint64_t size_bytes{};
    uint32_t rank{};
    uint32_t dims[4]{};// outermost-first; dims[i] == 0 for i >= rank
    [[nodiscard]] uint64_t element_count() const noexcept;
    [[nodiscard]] uint64_t dim(uint32_t i) const noexcept { return i < rank ? dims[i] : 0; }
};

class Package {
public:
    // Loads and validates the header/table. Blobs stay on disk (mapped) or are
    // read on demand — implementation choice; upload to device buffers is the
    // Session's job via tensors().
    static luisa::unique_ptr<Package> load(const std::filesystem::path &bgm_file);

    // Look up by canonical name ("layer7.qkv"). Returns nullptr if absent.
    [[nodiscard]] virtual const Tensor *find(luisa::string_view canonical_name) const noexcept = 0;
    // All tensors in file (emission) order.
    [[nodiscard]] virtual luisa::span<const Tensor> tensors() const noexcept = 0;
    // Element format of the file (header precision field, validated against
    // the version on load).
    [[nodiscard]] virtual PackagePrecision precision() const noexcept = 0;
    // Convenience: model hyper-parameters baked into the package (from the
    // packer constants; validated against table dims on load).
    [[nodiscard]] virtual uint32_t num_layers() const noexcept = 0;
    [[nodiscard]] virtual uint32_t hidden_size() const noexcept = 0;
    [[nodiscard]] virtual uint32_t num_heads() const noexcept = 0;
    [[nodiscard]] virtual uint32_t intermediate_size() const noexcept = 0;
    [[nodiscard]] virtual uint32_t vocab_size() const noexcept = 0;
    [[nodiscard]] virtual uint32_t max_positions() const noexcept = 0;

    virtual ~Package() = default;
};

}// namespace luisa::embed
