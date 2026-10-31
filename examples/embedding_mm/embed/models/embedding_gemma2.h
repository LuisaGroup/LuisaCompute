#pragma once
// embedding_mm engine -- Top-level EmbeddingGemma-2 multimodal model orchestrator.
//
// Coordinates Selective Encoder Loading across:
// - LanguageModel (`model.language_model` + `1_Pooling` + `2_Dense` + MRL head)
// - VisionTower (`model.vision_tower` + `model.embed_vision`, optional)
// - AudioTower (`model.audio_tower` + `model.embed_audio`, optional)

#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>

#include "embedding_mm/embed/executor.h"
#include "embedding_mm/embed/models/audio_tower.h"
#include "embedding_mm/embed/models/language_model.h"
#include "embedding_mm/embed/models/vision_tower.h"
#include "embedding_mm/embed/package.h"

namespace luisa::embed_mm {

[[nodiscard]] compute::Buffer<float> upload_tensor_from_package(
    compute::Device &device,
    const Package &package,
    luisa::string_view name,
    compute::CommandList &upload_cmds,
    luisa::vector<luisa::vector<float>> &half_storage);

class EmbeddingGemma2 final : public Executor {
public:
    explicit EmbeddingGemma2(compute::Device &device) noexcept
        : _device(device), _language_model(device) {}

    void initialize(const Package &package, bool enable_vision, bool enable_audio);

    [[nodiscard]] bool has_vision() const noexcept override { return _vision_tower.has_value(); }
    [[nodiscard]] bool has_audio() const noexcept override { return _audio_tower.has_value(); }

    void record_vision_encode(
        compute::CommandList &cmd_list,
        compute::BufferView<float> pixel_values,
        compute::BufferView<int32_t> position_ids,
        compute::BufferView<float> out_soft_tokens,
        uint32_t B,
        uint32_t num_patches_w,
        uint32_t num_patches_h,
        uint32_t max_patches,
        uint32_t num_soft_tokens,
        VisionDebugActivations *debug_acts) override;

    void record_audio_encode(
        compute::CommandList &cmd_list,
        compute::BufferView<float> input_features,
        compute::BufferView<float> out_soft_tokens,
        uint32_t num_frames,
        uint32_t num_mel_bins,
        uint32_t num_soft_tokens,
        AudioDebugActivations *debug_acts) override;

    void record_encode(
        compute::CommandList &cmd_list,
        compute::BufferView<uint32_t> input_ids,
        compute::BufferView<float> attention_mask,
        compute::BufferView<float> out_embeddings,
        uint32_t B,
        uint32_t S,
        uint32_t target_dim,
        const MultimodalInputs &mm_inputs) override;

private:
    compute::Device &_device;
    LanguageModel _language_model;
    luisa::optional<VisionTower> _vision_tower;
    luisa::optional<AudioTower> _audio_tower;
};

}// namespace luisa::embed_mm
