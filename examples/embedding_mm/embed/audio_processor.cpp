#include "audio_processor.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <numbers>

#include <luisa/core/logging.h>
#include "package.h"

namespace luisa::embed_mm {

namespace {

constexpr float kPi = std::numbers::pi_v<float>;
constexpr uint32_t kTargetSampleRate = 16000u;
constexpr uint32_t kFrameLength = 320u; // 20 ms
constexpr uint32_t kHopLength = 160u;   // 10 ms
constexpr uint32_t kFftLength = 512u;   // 2^9
constexpr uint32_t kNumBins = 257u;     // kFftLength / 2 + 1
constexpr uint32_t kNumFilters = 128u;  // 128 Mel channels

// Cooley-Tukey Radix-2 FFT (in-place on 512 complex numbers)
void fft_512(float *real, float *imag) noexcept {
    constexpr uint32_t N = kFftLength;
    // Bit reversal permutation
    for (uint32_t i = 1, j = 0; i < N; i++) {
        uint32_t bit = N >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    // Cooley-Tukey butterflies
    for (uint32_t len = 2; len <= N; len <<= 1) {
        float angle = -2.0f * kPi / static_cast<float>(len);
        float wlen_r = std::cos(angle);
        float wlen_i = std::sin(angle);
        uint32_t half = len >> 1;

        for (uint32_t i = 0; i < N; i += len) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            for (uint32_t j = 0; j < half; j++) {
                uint32_t u_idx = i + j;
                uint32_t v_idx = i + j + half;

                float u_r = real[u_idx];
                float u_i = imag[u_idx];
                float v_r = real[v_idx] * w_r - imag[v_idx] * w_i;
                float v_i = real[v_idx] * w_i + imag[v_idx] * w_r;

                real[u_idx] = u_r + v_r;
                imag[u_idx] = u_i + v_i;
                real[v_idx] = u_r - v_r;
                imag[v_idx] = u_i - v_i;

                float next_w_r = w_r * wlen_r - w_i * wlen_i;
                float next_w_i = w_r * wlen_i + w_i * wlen_r;
                w_r = next_w_r;
                w_i = next_w_i;
            }
        }
    }
}

// Generate HTK Mel filterbank [257, 128]
luisa::vector<float> generate_htk_mel_filters() {
    luisa::vector<float> weights(kNumBins * kNumFilters, 0.0f);

    auto hz_to_mel = [](float hz) noexcept {
        return 2595.0f * std::log10(1.0f + hz / 700.0f);
    };
    auto mel_to_hz = [](float mel) noexcept {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    };

    float min_mel = hz_to_mel(0.0f);
    float max_mel = hz_to_mel(8000.0f);

    luisa::vector<float> mel_points(kNumFilters + 2);
    for (uint32_t i = 0; i < kNumFilters + 2; i++) {
        mel_points[i] = min_mel + (max_mel - min_mel) * (static_cast<float>(i) / static_cast<float>(kNumFilters + 1));
    }

    luisa::vector<float> bin_points(kNumFilters + 2);
    for (uint32_t i = 0; i < kNumFilters + 2; i++) {
        float hz = mel_to_hz(mel_points[i]);
        bin_points[i] = std::floor(static_cast<float>(kFftLength + 1) * hz / static_cast<float>(kTargetSampleRate));
    }

    for (uint32_t m = 0; m < kNumFilters; m++) {
        float left = bin_points[m];
        float center = bin_points[m + 1];
        float right = bin_points[m + 2];

        for (uint32_t k = 0; k < kNumBins; k++) {
            float f_k = static_cast<float>(k);
            if (f_k >= left && f_k <= center && center > left) {
                weights[k * kNumFilters + m] = (f_k - left) / (center - left);
            } else if (f_k >= center && f_k <= right && right > center) {
                weights[k * kNumFilters + m] = (right - f_k) / (right - center);
            }
        }
    }
    return weights;
}

} // namespace

uint32_t AudioProcessor::compute_num_soft_tokens(uint32_t num_frames) noexcept {
    if (num_frames < 3u) return 0u;
    uint32_t t1 = (num_frames + 2u - 3u) / 2u + 1u;
    if (t1 < 3u) return 0u;
    uint32_t t2 = (t1 + 2u - 3u) / 2u + 1u;
    return t2;
}

luisa::vector<float> AudioProcessor::resample_to_16k(
    luisa::span<const float> in_samples,
    uint32_t in_sr) {

    if (in_samples.empty()) return {};
    if (in_sr == kTargetSampleRate) {
        return luisa::vector<float>(in_samples.begin(), in_samples.end());
    }

    // Specialized high-quality sinc decimation for 48 kHz -> 16 kHz (exact 3:1 integer downsampling)
    if (in_sr == 48000u) {
        size_t out_len = in_samples.size() / 3u;
        luisa::vector<float> out(out_len);

        // Kaiser-windowed sinc FIR filter for 3:1 downsampling (cutoff = pi/3, length = 31)
        constexpr int half_filter = 15;
        luisa::vector<float> fir(2 * half_filter + 1);
        float sum_fir = 0.0f;
        for (int i = -half_filter; i <= half_filter; i++) {
            float x = static_cast<float>(i);
            float sinc = (i == 0) ? 1.0f : (std::sin(kPi * x / 3.0f) / (kPi * x));
            // Blackman-Harris window
            float a0 = 0.35875f, a1 = 0.48829f, a2 = 0.14128f, a3 = 0.01168f;
            float w_arg = kPi * (x + static_cast<float>(half_filter)) / static_cast<float>(2 * half_filter);
            float w = a0 - a1 * std::cos(2.0f * w_arg) + a2 * std::cos(4.0f * w_arg) - a3 * std::cos(6.0f * w_arg);
            float val = sinc * w;
            fir[i + half_filter] = val;
            sum_fir += val;
        }
        for (auto &w : fir) w /= sum_fir;

        for (size_t i = 0; i < out_len; i++) {
            int center_idx = static_cast<int>(i * 3u);
            float acc = 0.0f;
            for (int k = -half_filter; k <= half_filter; k++) {
                int src_idx = center_idx + k;
                if (src_idx >= 0 && src_idx < static_cast<int>(in_samples.size())) {
                    acc += in_samples[src_idx] * fir[k + half_filter];
                }
            }
            out[i] = acc;
        }
        return out;
    }

    // General rational resampling using polyphase windowed-sinc interpolation
    double ratio = static_cast<double>(kTargetSampleRate) / static_cast<double>(in_sr);
    size_t out_len = static_cast<size_t>(std::ceil(in_samples.size() * ratio));
    luisa::vector<float> out(out_len);

    constexpr int sinc_radius = 8;
    for (size_t i = 0; i < out_len; i++) {
        double src_pos = static_cast<double>(i) / ratio;
        int center = static_cast<int>(std::floor(src_pos));
        float acc = 0.0f;
        float w_sum = 0.0f;

        for (int k = -sinc_radius; k <= sinc_radius; k++) {
            int src_idx = center + k;
            if (src_idx >= 0 && src_idx < static_cast<int>(in_samples.size())) {
                double diff = src_pos - static_cast<double>(src_idx);
                double arg = diff * std::min(1.0, ratio);
                float sinc = (std::abs(arg) < 1e-6) ? 1.0f : static_cast<float>(std::sin(kPi * arg) / (kPi * arg));
                float w = 0.5f + 0.5f * static_cast<float>(std::cos(kPi * diff / static_cast<double>(sinc_radius)));
                float weight = sinc * w;
                acc += in_samples[src_idx] * weight;
                w_sum += weight;
            }
        }
        out[i] = (w_sum > 1e-5f) ? (acc / w_sum) : 0.0f;
    }
    return out;
}

ProcessedAudio AudioProcessor::process_memory(
    const float *samples,
    size_t num_samples,
    uint32_t input_sample_rate,
    const Package *package) {

    ProcessedAudio result{};
    if (!samples || num_samples == 0) return result;

    luisa::span<const float> in_span{samples, num_samples};
    luisa::vector<float> mono_16k = resample_to_16k(in_span, input_sample_rate);
    result.raw_samples_16k = mono_16k;

    // Semicausal padding: prepend 160 zeros
    constexpr size_t pad_left = 160;
    luisa::vector<float> padded(pad_left + mono_16k.size(), 0.0f);
    std::memcpy(padded.data() + pad_left, mono_16k.data(), mono_16k.size() * sizeof(float));

    if (padded.size() < 321) {
        return result;
    }

    uint32_t num_frames = static_cast<uint32_t>((padded.size() - 321) / kHopLength + 1);
    result.num_frames = num_frames;
    result.num_soft_tokens = compute_num_soft_tokens(num_frames);
    result.features.resize(num_frames * kNumFilters);

    // Periodic Hann window (320 samples)
    luisa::vector<float> window(kFrameLength);
    for (uint32_t n = 0; n < kFrameLength; n++) {
        window[n] = 0.5f - 0.5f * std::cos(2.0f * kPi * static_cast<float>(n) / static_cast<float>(kFrameLength));
    }

    // Mel filterbank [257, 128]
    const float *mel_filters_data = nullptr;
    luisa::vector<float> generated_filters;
    if (package != nullptr) {
        if (const auto *t = package->find("audio.frontend.mel_filters")) {
            mel_filters_data = t->data;
        }
    }
    if (mel_filters_data == nullptr) {
        generated_filters = generate_htk_mel_filters();
        mel_filters_data = generated_filters.data();
    }

    // FFT workspace buffers
    float fft_r[kFftLength];
    float fft_i[kFftLength];
    float mag[kNumBins];

    for (uint32_t m = 0; m < num_frames; m++) {
        const float *frame_src = padded.data() + m * kHopLength;
        for (uint32_t n = 0; n < kFrameLength; n++) {
            fft_r[n] = frame_src[n] * window[n];
            fft_i[n] = 0.0f;
        }
        for (uint32_t n = kFrameLength; n < kFftLength; n++) {
            fft_r[n] = 0.0f;
            fft_i[n] = 0.0f;
        }

        fft_512(fft_r, fft_i);

        for (uint32_t k = 0; k < kNumBins; k++) {
            mag[k] = std::sqrt(fft_r[k] * fft_r[k] + fft_i[k] * fft_i[k]);
        }

        float *out_feat = result.features.data() + m * kNumFilters;
        for (uint32_t f = 0; f < kNumFilters; f++) {
            float mel_acc = 0.0f;
            for (uint32_t k = 0; k < kNumBins; k++) {
                mel_acc += mag[k] * mel_filters_data[k * kNumFilters + f];
            }
            out_feat[f] = std::log(mel_acc + 0.001f);
        }
    }

    return result;
}

ProcessedAudio AudioProcessor::process_wav_bytes(
    const uint8_t *bytes,
    size_t size_bytes,
    const Package *package) {

    if (!bytes || size_bytes < 44) {
        LUISA_ERROR("AudioProcessor: invalid WAV bytes (size < 44)");
    }

    // Check 'RIFF' and 'WAVE'
    if (std::memcmp(bytes, "RIFF", 4) != 0 || std::memcmp(bytes + 8, "WAVE", 4) != 0) {
        LUISA_ERROR("AudioProcessor: not a valid RIFF/WAVE file");
    }

    uint16_t audio_format = 1;
    uint16_t num_channels = 1;
    uint32_t sample_rate = 16000;
    uint16_t bits_per_sample = 16;
    const uint8_t *data_ptr = nullptr;
    size_t data_bytes = 0;

    // Scan chunks
    size_t offset = 12;
    while (offset + 8 <= size_bytes) {
        char chunk_id[5] = {0};
        std::memcpy(chunk_id, bytes + offset, 4);
        uint32_t chunk_size = *reinterpret_cast<const uint32_t *>(bytes + offset + 4);
        offset += 8;

        if (std::strcmp(chunk_id, "fmt ") == 0 && chunk_size >= 16) {
            audio_format = *reinterpret_cast<const uint16_t *>(bytes + offset);
            num_channels = *reinterpret_cast<const uint16_t *>(bytes + offset + 2);
            sample_rate = *reinterpret_cast<const uint32_t *>(bytes + offset + 4);
            bits_per_sample = *reinterpret_cast<const uint16_t *>(bytes + offset + 14);
        } else if (std::strcmp(chunk_id, "data") == 0) {
            data_ptr = bytes + offset;
            data_bytes = std::min<size_t>(chunk_size, size_bytes - offset);
        }
        offset += chunk_size;
    }

    if (!data_ptr || data_bytes == 0) {
        LUISA_ERROR("AudioProcessor: no 'data' chunk found in WAV");
    }

    luisa::vector<float> mono_samples;
    if (audio_format == 1) { // PCM integer
        if (bits_per_sample == 16) {
            size_t total_samples = data_bytes / 2;
            size_t frame_count = total_samples / num_channels;
            mono_samples.resize(frame_count);
            const auto *p16 = reinterpret_cast<const int16_t *>(data_ptr);
            for (size_t i = 0; i < frame_count; i++) {
                float sum = 0.0f;
                for (uint16_t c = 0; c < num_channels; c++) {
                    sum += static_cast<float>(p16[i * num_channels + c]) / 32768.0f;
                }
                mono_samples[i] = sum / static_cast<float>(num_channels);
            }
        } else if (bits_per_sample == 24) {
            size_t total_samples = data_bytes / 3;
            size_t frame_count = total_samples / num_channels;
            mono_samples.resize(frame_count);
            for (size_t i = 0; i < frame_count; i++) {
                float sum = 0.0f;
                for (uint16_t c = 0; c < num_channels; c++) {
                    size_t idx = (i * num_channels + c) * 3;
                    int32_t val = static_cast<int32_t>(data_ptr[idx]) |
                                  (static_cast<int32_t>(data_ptr[idx + 1]) << 8) |
                                  (static_cast<int32_t>(static_cast<int8_t>(data_ptr[idx + 2])) << 16);
                    sum += static_cast<float>(val) / 8388608.0f;
                }
                mono_samples[i] = sum / static_cast<float>(num_channels);
            }
        } else if (bits_per_sample == 8) {
            size_t frame_count = data_bytes / num_channels;
            mono_samples.resize(frame_count);
            for (size_t i = 0; i < frame_count; i++) {
                float sum = 0.0f;
                for (uint16_t c = 0; c < num_channels; c++) {
                    float s = (static_cast<float>(data_ptr[i * num_channels + c]) - 128.0f) / 128.0f;
                    sum += s;
                }
                mono_samples[i] = sum / static_cast<float>(num_channels);
            }
        } else {
            LUISA_ERROR("AudioProcessor: unsupported PCM bits per sample: {}", bits_per_sample);
        }
    } else if (audio_format == 3) { // IEEE float
        if (bits_per_sample == 32) {
            size_t total_samples = data_bytes / 4;
            size_t frame_count = total_samples / num_channels;
            mono_samples.resize(frame_count);
            const auto *pf = reinterpret_cast<const float *>(data_ptr);
            for (size_t i = 0; i < frame_count; i++) {
                float sum = 0.0f;
                for (uint16_t c = 0; c < num_channels; c++) {
                    sum += pf[i * num_channels + c];
                }
                mono_samples[i] = sum / static_cast<float>(num_channels);
            }
        } else {
            LUISA_ERROR("AudioProcessor: unsupported IEEE float bits per sample: {}", bits_per_sample);
        }
    } else {
        LUISA_ERROR("AudioProcessor: unsupported WAV audio format: {}", audio_format);
    }

    return process_memory(mono_samples.data(), mono_samples.size(), sample_rate, package);
}

ProcessedAudio AudioProcessor::process_file(
    const std::filesystem::path &wav_path,
    const Package *package) {

    std::ifstream file(wav_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LUISA_ERROR("AudioProcessor::process_file: failed to open '{}'", wav_path.string());
    }

    auto file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    luisa::vector<uint8_t> buffer(static_cast<size_t>(file_size));
    if (!file.read(reinterpret_cast<char *>(buffer.data()), file_size)) {
        LUISA_ERROR("AudioProcessor::process_file: failed to read '{}'", wav_path.string());
    }

    return process_wav_bytes(buffer.data(), buffer.size(), package);
}

}// namespace luisa::embed_mm
