#pragma once
// mm_embedding engine -- forward pass graph executor.
//
// Manages device buffers, shader compilation, activation pools, and records
// the complete EmbeddingGemma-2 forward pass into compute::CommandList:
// 1. Vision Tower (Phase 2): Patch embedding, 16 ViT layers, 2D pooling, connector
// 2. Token embedding lookup + embed scaling (sqrt(512))
// 3. 24 Transformer layers (RMSNorm, RoPE, GQA, GeGLU, alternating post-norms)
// 4. Final RMSNorm
// 5. Sequence mask-aware Mean Pooling
// 6. Projection (512 -> 768)
// 7. Matryoshka Representation Learning (MRL) truncation + L2 Normalization
//
// Supports Selective Encoder Loading: vision encoder can be toggled on/off to save VRAM.

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
    compute::BufferView<float> vision_embeds{};// [num_vision_tokens, 512]
    uint32_t vision_start_idx{2u};             // Token index where soft tokens are inserted
    uint32_t num_vision_tokens{256u};          // Number of soft tokens

    compute::BufferView<float> audio_embeds{}; // [num_audio_tokens, 512]
    uint32_t audio_start_idx{2u};              // Token index where soft tokens are inserted
    uint32_t num_audio_tokens{77u};            // Number of audio soft tokens
};

class Executor {
public:
    static luisa::unique_ptr<Executor> create(
        compute::Device &device,
        const Package &package,
        bool enable_vision = true,
        bool enable_audio = true);

    virtual ~Executor() = default;

    Executor(const Executor &) = delete;
    Executor(Executor &&) = delete;
    Executor &operator=(const Executor &) = delete;
    Executor &operator=(Executor &&) = delete;

    [[nodiscard]] virtual bool has_vision() const noexcept = 0;
    [[nodiscard]] virtual bool has_audio() const noexcept = 0;

    /**
     * Records the vision tower forward pass into cmd_list.
     *
     * @param cmd_list Destination command list.
     * @param pixel_values Scaled patches buffer [B * max_patches * 768] (floats in [0, 1]).
     * @param position_ids Patch 2D position coordinates [B * max_patches * 2] (int32).
     * @param out_soft_tokens Destination device buffer for projected soft tokens [B * num_soft_tokens * 512].
     * @param B Batch size (typically 1).
     * @param num_patches_w Number of active patches horizontally (e.g. 48 for 768px).
     * @param num_patches_h Number of active patches vertically (e.g. 48 for 768px).
     * @param max_patches Max patches with padding (2520).
     * @param num_soft_tokens Number of output soft tokens (e.g. 256 for 48x48, 253 for 69x33).
     */
    virtual void record_vision_encode(
        compute::CommandList &cmd_list,
        compute::BufferView<float> pixel_values,
        compute::BufferView<int32_t> position_ids,
        compute::BufferView<float> out_soft_tokens,
        uint32_t B = 1u,
        uint32_t num_patches_w = 48u,
        uint32_t num_patches_h = 48u,
        uint32_t max_patches = 2520u,
        uint32_t num_soft_tokens = 256u) = 0;

    struct AudioDebugActivations {
        compute::BufferView<float> subsample_proj{};   // [num_soft_tokens, 1024]
        compute::BufferView<float> layer0_out{};       // [num_soft_tokens, 1024]
        compute::BufferView<float> layer11_out{};      // [num_soft_tokens, 1024]
        compute::BufferView<float> audio_tower_out{};  // [num_soft_tokens, 1536]
    };

    /**
     * Records the audio tower forward pass into cmd_list.
     *
     * @param cmd_list Destination command list.
     * @param input_features Mel spectrogram features buffer [T_in * 128] (e.g. 307 * 128).
     * @param out_soft_tokens Destination device buffer for projected soft tokens [T_out * 512] (e.g. 77 * 512).
     * @param num_frames Total input frames (e.g. 307).
     * @param num_mel_bins Number of mel filter bins (128).
     * @param num_soft_tokens Number of output soft tokens (e.g. 77).
     * @param debug_acts Optional buffer views to capture intermediate layer activations for testing.
     */
    virtual void record_audio_encode(
        compute::CommandList &cmd_list,
        compute::BufferView<float> input_features,
        compute::BufferView<float> out_soft_tokens,
        uint32_t num_frames = 307u,
        uint32_t num_mel_bins = 128u,
        uint32_t num_soft_tokens = 77u,
        AudioDebugActivations *debug_acts = nullptr) = 0;

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
     * @param mm_inputs Optional multimodal soft token embeddings to scatter into input sequence.
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
