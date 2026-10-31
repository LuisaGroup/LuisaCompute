#pragma once
// mm_embedding engine -- packed model package loader for model.mmegm.
//
// Binary package format (EGM2):
// - Magic "EGM2", little-endian, 256-byte aligned tensor blobs.
// - Supports fp32 (0), bf16 (1), and fp16 (2) storage precisions.
// - Canonical names:
//     emb.word                      [262144, 512]
//     layer{i}.input_norm           [512]
//     layer{i}.q_proj               [512, 1024] (or [512, 2048] for full attn)
//     layer{i}.k_proj               [512, 512]
//     layer{i}.v_proj               [512, 512]
//     layer{i}.out_proj             [1024, 512] (or [2048, 512] for full attn)
//     layer{i}.post_attn_norm       [512]
//     layer{i}.pre_ffn_norm         [512]
//     layer{i}.gate_proj            [512, 2048]
//     layer{i}.up_proj              [512, 2048]
//     layer{i}.down_proj            [2048, 512]
//     layer{i}.post_ffn_norm        [512]
//     final_norm                    [512]
//     embedding_proj                [512, 768]

#include <cstdint>
#include <filesystem>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::mm_embed {

enum class PackagePrecision : uint32_t {
    fp32 = 0u,
    bf16 = 1u,
    fp16 = 2u,
};

[[nodiscard]] const char *package_precision_name(PackagePrecision p) noexcept;

struct Tensor {
    const float *data{};
    const uint16_t *half_data{};
    uint64_t size_bytes{};
    uint32_t rank{};
    uint32_t dims[4]{};

    [[nodiscard]] uint64_t element_count() const noexcept;
    [[nodiscard]] uint64_t dim(uint32_t i) const noexcept { return i < rank ? dims[i] : 0; }
};

class Package {
public:
    static luisa::unique_ptr<Package> load(const std::filesystem::path &egm_file);

    [[nodiscard]] virtual const Tensor *find(luisa::string_view canonical_name) const noexcept = 0;
    [[nodiscard]] virtual luisa::span<const Tensor> tensors() const noexcept = 0;
    [[nodiscard]] virtual PackagePrecision precision() const noexcept = 0;

    // Model hyper-parameters
    [[nodiscard]] virtual uint32_t num_layers() const noexcept = 0;
    [[nodiscard]] virtual uint32_t hidden_size() const noexcept = 0;
    [[nodiscard]] virtual uint32_t num_heads() const noexcept = 0;
    [[nodiscard]] virtual uint32_t num_kv_heads() const noexcept = 0;
    [[nodiscard]] virtual uint32_t head_dim() const noexcept = 0;
    [[nodiscard]] virtual uint32_t intermediate_size() const noexcept = 0;
    [[nodiscard]] virtual uint32_t vocab_size() const noexcept = 0;
    [[nodiscard]] virtual uint32_t embedding_dim() const noexcept = 0;
    [[nodiscard]] virtual uint32_t max_positions() const noexcept = 0;
    [[nodiscard]] virtual bool has_vision() const noexcept = 0;
    [[nodiscard]] virtual bool has_audio() const noexcept = 0;

    virtual ~Package() = default;
};

}// namespace luisa::mm_embed
