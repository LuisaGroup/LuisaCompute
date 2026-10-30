#pragma once
// mm_embedding engine -- forward pass graph executor.
//
// Manages device buffers, shader compilation, activation pools, and records
// the complete EmbeddingGemma-2 forward pass into compute::CommandList:
// 1. Token embedding lookup + embed scaling (sqrt(512))
// 2. 24 Transformer layers (RMSNorm, RoPE, GQA, GeGLU, alternating post-norms)
// 3. Final RMSNorm
// 4. Sequence mask-aware Mean Pooling
// 5. Projection (512 -> 768)
// 6. Matryoshka Representation Learning (MRL) truncation + L2 Normalization
//
// Includes modular interface hooks for Phase 2 (Vision) and Phase 3 (Audio).

#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

#include "mm_embedding/embed/package.h"

namespace luisa::mm_embed {

struct MultimodalInputs {
    compute::BufferView<float> vision_embeds{};// [B * num_patches, 512]
    compute::BufferView<float> audio_embeds{}; // [B * num_frames, 512]
};

class Executor {
public:
    static luisa::unique_ptr<Executor> create(compute::Device &device, const Package &package);

    virtual ~Executor() = default;

    Executor(const Executor &) = delete;
    Executor(Executor &&) = delete;
    Executor &operator=(const Executor &) = delete;
    Executor &operator=(Executor &&) = delete;

    /**
     * Records the text embedding forward pass into cmd_list.
     *
     * @param cmd_list Destination command list.
     * @param input_ids Device buffer of token IDs [B * S].
     * @param attention_mask Device buffer of float mask [B * S] (1.0 = valid, 0.0 = pad).
     * @param out_embeddings Destination device buffer for normalized embeddings [B * target_dim].
     * @param B Batch size.
     * @param S Sequence length (padded to bucket or batch max).
     * @param target_dim Output MRL dimension (e.g. 768, 512, 256, 128).
     * @param mm_inputs Optional multimodal embeddings (Phase 2/3 hook).
     */
    virtual void record_encode(
        compute::CommandList &cmd_list,
        compute::BufferView<uint32_t> input_ids,
        compute::BufferView<float> attention_mask,
        compute::BufferView<float> out_embeddings,
        uint32_t B,
        uint32_t S,
        uint32_t target_dim = 768u,
        const MultimodalInputs &mm_inputs = {}) = 0;

protected:
    Executor() = default;
};

}// namespace luisa::mm_embed
