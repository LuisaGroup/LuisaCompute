// embedding_mm engine -- Top-level EmbeddingGemma-2 multimodal model implementation.
#include "embedding_mm/embed/models/embedding_gemma2.h"

#include <cstring>
#include <luisa/core/logging.h>
#include <luisa/runtime/stream.h>

namespace luisa::embed_mm {

compute::Buffer<float> upload_tensor_from_package(
    compute::Device &device,
    const Package &package,
    luisa::string_view name,
    compute::CommandList &upload_cmds,
    luisa::vector<luisa::vector<float>> &half_storage) {
    const auto *t = package.find(name);
    if (!t) {
        LUISA_ERROR("EmbeddingGemma2: missing required weight tensor '{}'", name);
    }
    auto elem_count = static_cast<size_t>(t->element_count());
    auto buf = device.create_buffer<float>(elem_count);

    if (t->data) {
        upload_cmds << buf.copy_from(t->data);
    } else if (t->half_data) {
        auto &fp32_data = half_storage.emplace_back(elem_count);
        for (size_t i = 0; i < elem_count; i++) {
            uint16_t h = t->half_data[i];
            if (package.precision() == PackagePrecision::bf16) {
                uint32_t w = static_cast<uint32_t>(h) << 16;
                std::memcpy(&fp32_data[i], &w, sizeof(float));
            } else {
                uint32_t sign = (h & 0x8000u) << 16;
                uint32_t exp = (h & 0x7C00u) >> 10;
                uint32_t mant = (h & 0x03FFu) << 13;
                uint32_t f_val = 0;
                if (exp == 0x1Fu) {
                    f_val = sign | 0x7F800000u | mant;
                } else if (exp == 0) {
                    if (mant != 0) {
                        exp = 113;
                        while ((mant & (1u << 23)) == 0) {
                            mant <<= 1;
                            exp--;
                        }
                        mant &= 0x007FFFFFu;
                        f_val = sign | (exp << 23) | mant;
                    } else {
                        f_val = sign;
                    }
                } else {
                    f_val = sign | ((exp + 112) << 23) | mant;
                }
                std::memcpy(&fp32_data[i], &f_val, sizeof(float));
            }
        }
        upload_cmds << buf.copy_from(fp32_data.data());
    }
    return buf;
}

void EmbeddingGemma2::initialize(const Package &package, bool enable_vision, bool enable_audio) {
    auto upload_stream = _device.create_stream(compute::StreamTag::COMPUTE);
    compute::CommandList upload_cmds;
    luisa::vector<luisa::vector<float>> half_storage;

    // 1. Load Language Model weights (~297M params)
    _language_model.load_weights(package, upload_cmds, half_storage);

    // 2. Selective Encoder Loading: Vision Tower (~170M params)
    if (enable_vision && package.has_vision()) {
        LUISA_INFO("EmbeddingGemma2: Selective Encoder Loading -> initializing Vision Tower (~170M params)...");
        _vision_tower.emplace(_device);
        _vision_tower->load_weights(package, upload_cmds, half_storage);
    } else {
        LUISA_INFO("EmbeddingGemma2: Selective Encoder Loading -> Vision Tower skipped.");
    }

    // 3. Selective Encoder Loading: Audio Tower (~300M params)
    if (enable_audio && package.has_audio()) {
        LUISA_INFO("EmbeddingGemma2: Selective Encoder Loading -> initializing Audio Tower (~300M params)...");
        _audio_tower.emplace(_device);
        _audio_tower->load_weights(package, upload_cmds, half_storage);
    } else {
        LUISA_INFO("EmbeddingGemma2: Selective Encoder Loading -> Audio Tower skipped.");
    }

    upload_stream << upload_cmds.commit() << compute::synchronize();

    // 4. Compile DSL operators for active sub-models
    _language_model.compile_shaders();
    if (_vision_tower) {
        _vision_tower->compile_shaders();
    }
    if (_audio_tower) {
        _audio_tower->compile_shaders();
    }

    LUISA_INFO("EmbeddingGemma2: initialized and compiled all kernels (vision={}, audio={})",
               has_vision() ? "enabled" : "disabled",
               has_audio() ? "enabled" : "disabled");
}

void EmbeddingGemma2::record_vision_encode(
    compute::CommandList &cmd_list,
    compute::BufferView<float> pixel_values,
    compute::BufferView<int32_t> position_ids,
    compute::BufferView<float> out_soft_tokens,
    uint32_t B,
    uint32_t num_patches_w,
    uint32_t num_patches_h,
    uint32_t max_patches,
    uint32_t num_soft_tokens,
    VisionDebugActivations *debug_acts) {
    if (!_vision_tower) {
        LUISA_ERROR("record_vision_encode: Vision Tower is not loaded (Selective Encoder Loading)");
    }
    _vision_tower->record_encode(
        cmd_list, pixel_values, position_ids, out_soft_tokens,
        B, num_patches_w, num_patches_h, max_patches, num_soft_tokens, debug_acts);
}

void EmbeddingGemma2::record_audio_encode(
    compute::CommandList &cmd_list,
    compute::BufferView<float> input_features,
    compute::BufferView<float> out_soft_tokens,
    uint32_t num_frames,
    uint32_t num_mel_bins,
    uint32_t num_soft_tokens,
    AudioDebugActivations *debug_acts) {
    if (!_audio_tower) {
        LUISA_ERROR("record_audio_encode: Audio Tower is not loaded (Selective Encoder Loading)");
    }
    _audio_tower->record_encode(
        cmd_list, input_features, out_soft_tokens,
        num_frames, num_mel_bins, num_soft_tokens, debug_acts);
}

void EmbeddingGemma2::record_encode(
    compute::CommandList &cmd_list,
    compute::BufferView<uint32_t> input_ids,
    compute::BufferView<float> attention_mask,
    compute::BufferView<float> out_embeddings,
    uint32_t B,
    uint32_t S,
    uint32_t target_dim,
    const MultimodalInputs &mm_inputs) {
    _language_model.record_encode(
        cmd_list, input_ids, attention_mask, out_embeddings,
        B, S, target_dim, mm_inputs);
}

luisa::unique_ptr<Executor> Executor::create(
    compute::Device &device,
    const Package &package,
    bool enable_vision,
    bool enable_audio) {
    auto model = luisa::make_unique<EmbeddingGemma2>(device);
    model->initialize(package, enable_vision, enable_audio);
    return model;
}

}// namespace luisa::embed_mm
