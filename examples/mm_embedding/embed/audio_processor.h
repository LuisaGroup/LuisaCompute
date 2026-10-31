#pragma once
// mm_embedding engine -- C++ Audio Processor matching Gemma4AudioFeatureExtractor.
//
// Operations:
// 1. WAV decoding (PCM int16, float32, mono or multi-channel, arbitrary sample rate).
// 2. High-fidelity resampling to 16000 Hz mono.
// 3. Semicausal padding (160 zeros at start).
// 4. Periodic Hann window (320 samples, 20 ms) with 160 hop (10 ms).
// 5. Radix-2 Real FFT of length 512 -> 257 frequency bins.
// 6. HTK Mel filterbank (128 channels, 0 to 8000 Hz) -> [num_frames, 128].
// 7. Log Mel spectrogram: ln(mel + 0.001).
// 8. Output frame count and soft token calculation (num_soft_tokens = ((num_frames - 1)/2 + 1 - 1)/2 + 1).

#include <cstdint>
#include <filesystem>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>

namespace luisa::mm_embed {

class Package;

struct ProcessedAudio {
    luisa::vector<float> features;        // [num_frames * 128] log-mel features
    luisa::vector<float> raw_samples_16k; // resampled 16 kHz mono float samples
    uint32_t num_frames{0u};              // T (e.g. 307 for ~3.07s)
    uint32_t num_features{128u};          // 128 Mel channels
    uint32_t num_soft_tokens{0u};         // Soft tokens after 4x conv subsampling (e.g. 77)
    uint32_t sample_rate{16000u};
};

class AudioProcessor {
public:
    static ProcessedAudio process_file(
        const std::filesystem::path &wav_path,
        const Package *package = nullptr);

    static ProcessedAudio process_memory(
        const float *samples,
        size_t num_samples,
        uint32_t input_sample_rate = 16000u,
        const Package *package = nullptr);

    static ProcessedAudio process_wav_bytes(
        const uint8_t *bytes,
        size_t size_bytes,
        const Package *package = nullptr);

    // Compute expected soft token count from frame count
    [[nodiscard]] static uint32_t compute_num_soft_tokens(uint32_t num_frames) noexcept;

    // Resample waveform from in_sr to 16000 Hz mono
    static luisa::vector<float> resample_to_16k(
        luisa::span<const float> in_samples,
        uint32_t in_sr);
};

}// namespace luisa::mm_embed
