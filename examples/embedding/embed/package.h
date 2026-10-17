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
// - precision must be 0 (fp32); reject 1 (bf16 is plan M7).
#pragma once

#include <cstdint>
#include <filesystem>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::embed {

// One tensor descriptor from the bgm table. `data` is a host-side view valid
// for the lifetime of the Package (backed by the mapped/loaded blob).
struct Tensor {
    const float *data{};
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
