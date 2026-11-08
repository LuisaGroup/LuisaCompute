/**
 * LuisaCompute NTC (RTX Neural Texture Compression)
 * Common definitions, constants, and layout specifications.
 * Binary and mathematical parity with NVIDIA RTX NTC SDK (NTEX v3).
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <optional>
#include <algorithm>
#include <cstring>
#include <memory>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/struct.h>

namespace luisa::ntc {

using namespace luisa::compute;

// Container header constants
static constexpr uint32_t NTC_SIGNATURE_VALUE = 0x5845544Eu; // "NTEX"
static constexpr uint32_t NTC_CONTAINER_VERSION = 0x100u;    // 256
static constexpr uint32_t NTC_SCHEMA_VERSION = 3u;          // NTEX schema v3

// Neural architecture dimension constants
static constexpr uint32_t NTC_MAX_MIPS = 16u;
static constexpr uint32_t NTC_MAX_CHANNELS = 16u;
static constexpr uint32_t NTC_MAX_NEURAL_MIPS = 8u;

static constexpr uint32_t NTC_FEATURES_PER_LAYER = 4u; // BGRA4 format: 4 features per layer
static constexpr uint32_t NTC_MLP_LAYERS = 4u;         // 4-layer MLP
static constexpr uint32_t NTC_MLP_POS_ENC_WAVES = 3u;  // 3 octaves: frac(2^k * p)
static constexpr uint32_t NTC_MLP_SUPPLEMENTAL_INPUTS = 14u; // 12 wave inputs + 2 lod inputs

static constexpr uint32_t NTC_MLP_INPUT_CHANNELS   = 48u; // 16 (high) + 16 (low) + 12 (pos) + 2 (lod) + 2 pad = 48
static constexpr uint32_t NTC_MLP_HIDDEN0_CHANNELS = 64u;
static constexpr uint32_t NTC_MLP_HIDDEN1_CHANNELS = 48u;
static constexpr uint32_t NTC_MLP_HIDDEN2_CHANNELS = 32u;
static constexpr uint32_t NTC_MLP_OUTPUT_CHANNELS  = 16u;

// Total network parameters
static constexpr uint32_t NTC_TOTAL_WEIGHTS = 
    NTC_MLP_INPUT_CHANNELS * NTC_MLP_HIDDEN0_CHANNELS +
    NTC_MLP_HIDDEN0_CHANNELS * NTC_MLP_HIDDEN1_CHANNELS +
    NTC_MLP_HIDDEN1_CHANNELS * NTC_MLP_HIDDEN2_CHANNELS +
    NTC_MLP_HIDDEN2_CHANNELS * NTC_MLP_OUTPUT_CHANNELS; // 3072 + 3072 + 1536 + 512 = 8192

static constexpr uint32_t NTC_TOTAL_BIASES = 
    NTC_MLP_HIDDEN0_CHANNELS + NTC_MLP_HIDDEN1_CHANNELS +
    NTC_MLP_HIDDEN2_CHANNELS + NTC_MLP_OUTPUT_CHANNELS; // 64 + 48 + 32 + 16 = 160

// Normalization & Quantization constants
static constexpr float NTC_INPUT_SCALE = 127.5f;
static constexpr float NTC_HGELU_MAXVALUE = 3.0f;
static constexpr float NTC_HGELU_INVSTEP = 80.0f;
static constexpr float NTC_HGELU_STEP = 1.0f / 80.0f;
static constexpr float NTC_HGELU_BIAS = -113.0f;

enum class CompressionType : uint32_t {
    None = 0,
    GDeflate = 1
};

enum class ChannelFormat : uint32_t {
    UNKNOWN = 0,
    UNORM8 = 1,
    UNORM16 = 2,
    FLOAT16 = 3,
    FLOAT32 = 4,
    UINT32 = 5
};

enum class BlockCompressedFormat : uint32_t {
    None = 0,
    BC1 = 1,
    BC2 = 2,
    BC3 = 3,
    BC4 = 4,
    BC5 = 5,
    BC6 = 6,
    BC7 = 7
};

enum class ColorSpace : uint32_t {
    Linear = 0,
    sRGB = 1,
    HLG = 2
};

enum class InferenceWeightType : uint32_t {
    Unknown = 0,
    GenericInt8 = 1,
    GenericFP8 = 2,
    CoopVecFP8 = 3
};

enum class MatrixLayout : uint32_t {
    RowMajor = 0,
    ColumnMajor = 1
};

enum class ActivationType : uint32_t {
    HGELUClamp = 0
};

enum class MlpDataType : uint32_t {
    Int8 = 0,
    Int32 = 1,
    FloatE4M3 = 2,
    FloatE5M2 = 3,
    Float16 = 4,
    Float32 = 5
};

inline const char *compression_type_to_string(CompressionType type) noexcept {
    switch (type) {
        case CompressionType::GDeflate: return "GDeflate";
        default: return "None";
    }
}

inline CompressionType string_to_compression_type(std::string_view str) noexcept {
    if (str == "GDeflate") return CompressionType::GDeflate;
    return CompressionType::None;
}

inline const char *channel_format_to_string(ChannelFormat format) noexcept {
    switch (format) {
        case ChannelFormat::UNORM8: return "UNORM8";
        case ChannelFormat::UNORM16: return "UNORM16";
        case ChannelFormat::FLOAT16: return "FLOAT16";
        case ChannelFormat::FLOAT32: return "FLOAT32";
        case ChannelFormat::UINT32: return "UINT32";
        default: return "UNKNOWN";
    }
}

inline ChannelFormat string_to_channel_format(std::string_view str) noexcept {
    if (str == "UNORM8") return ChannelFormat::UNORM8;
    if (str == "UNORM16") return ChannelFormat::UNORM16;
    if (str == "FLOAT16") return ChannelFormat::FLOAT16;
    if (str == "FLOAT32") return ChannelFormat::FLOAT32;
    if (str == "UINT32") return ChannelFormat::UINT32;
    return ChannelFormat::UNKNOWN;
}

inline const char *bc_format_to_string(BlockCompressedFormat format) noexcept {
    switch (format) {
        case BlockCompressedFormat::BC1: return "BC1";
        case BlockCompressedFormat::BC2: return "BC2";
        case BlockCompressedFormat::BC3: return "BC3";
        case BlockCompressedFormat::BC4: return "BC4";
        case BlockCompressedFormat::BC5: return "BC5";
        case BlockCompressedFormat::BC6: return "BC6";
        case BlockCompressedFormat::BC7: return "BC7";
        default: return "None";
    }
}

inline BlockCompressedFormat string_to_bc_format(std::string_view str) noexcept {
    if (str == "BC1") return BlockCompressedFormat::BC1;
    if (str == "BC2") return BlockCompressedFormat::BC2;
    if (str == "BC3") return BlockCompressedFormat::BC3;
    if (str == "BC4") return BlockCompressedFormat::BC4;
    if (str == "BC5") return BlockCompressedFormat::BC5;
    if (str == "BC6") return BlockCompressedFormat::BC6;
    if (str == "BC7") return BlockCompressedFormat::BC7;
    return BlockCompressedFormat::None;
}

inline const char *colorspace_to_string(ColorSpace cs) noexcept {
    switch (cs) {
        case ColorSpace::sRGB: return "sRGB";
        case ColorSpace::HLG: return "HLG";
        default: return "Linear";
    }
}

inline ColorSpace string_to_colorspace(std::string_view str) noexcept {
    if (str == "sRGB") return ColorSpace::sRGB;
    if (str == "HLG") return ColorSpace::HLG;
    return ColorSpace::Linear;
}

inline const char *mlp_data_type_to_string(MlpDataType type) noexcept {
    switch (type) {
        case MlpDataType::Int8: return "Int8";
        case MlpDataType::Int32: return "Int32";
        case MlpDataType::FloatE4M3: return "FloatE4M3";
        case MlpDataType::FloatE5M2: return "FloatE5M2";
        case MlpDataType::Float16: return "Float16";
        case MlpDataType::Float32: return "Float32";
        default: return "Int8";
    }
}

inline MlpDataType string_to_mlp_data_type(std::string_view str) noexcept {
    if (str == "Int8") return MlpDataType::Int8;
    if (str == "Int32") return MlpDataType::Int32;
    if (str == "FloatE4M3") return MlpDataType::FloatE4M3;
    if (str == "FloatE5M2") return MlpDataType::FloatE5M2;
    if (str == "Float16") return MlpDataType::Float16;
    if (str == "Float32") return MlpDataType::Float32;
    return MlpDataType::Int8;
}

// Binary header at the start of NTEX files
#pragma pack(push, 1)
struct FileHeader {
    uint32_t signature{NTC_SIGNATURE_VALUE};
    uint32_t version{NTC_CONTAINER_VERSION};
    uint64_t jsonChunkOffset{0};
    uint64_t jsonChunkSize{0};
    uint64_t binaryChunkOffset{0};
    uint64_t binaryChunkSize{0};
};
#pragma pack(pop)

// Structure describing per-color-mip sampling parameters
struct ColorMipConstants {
    int32_t neuralMip{0};
    float positionLod{0.0f};
    float positionScale{1.0f};
    int32_t pad{0};
};

// Texture set constant buffer structure matching NTC GAPI layout
struct alignas(16) TextureSetConstants {
    ColorMipConstants colorMips[NTC_MAX_MIPS];
    int32_t networkWeightOffsets[4];
    int32_t networkBiasOffsets[4];
    int32_t networkScaleOffsets[4];
    int32_t imageWidth;
    int32_t imageHeight;
    int32_t imageMips;
    int32_t pad0;
    uint32_t validChannelMask;
    uint32_t channelColorSpaces; // 2 bits per channel
    int32_t pad1;
    int32_t pad2;
};

// Math helpers
inline size_t round_up_4(size_t size) noexcept {
    return (size + 3u) & ~size_t{3u};
}

inline size_t round_up_16(size_t size) noexcept {
    return (size + 15u) & ~size_t{15u};
}

// HGELU clamp activation
inline float hgelu_clamp(float x) noexcept {
    float clamped = std::min(x, NTC_HGELU_MAXVALUE);
    float t = std::clamp(x * 0.33333333f + 0.5f, 0.0f, 1.0f);
    return clamped * t;
}

inline int8_t hgelu_quantize(float x) noexcept {
    float val = std::round(x * NTC_HGELU_INVSTEP) + NTC_HGELU_BIAS;
    val = std::clamp(val, -128.0f, 127.0f);
    return static_cast<int8_t>(val);
}

inline float srgb_to_linear(float c) noexcept {
    c = std::clamp(c, 0.0f, 1.0f);
    return (c <= 0.04045f) ? (c / 12.92f) : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

inline float linear_to_srgb(float c) noexcept {
    c = std::clamp(c, 0.0f, 1.0f);
    return (c <= 0.0031308f) ? (c * 12.92f) : (1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f);
}

// Convert FP8 (E4M3) to float and vice versa
inline float fp8_e4m3_to_float(uint8_t byte_val) noexcept {
    uint8_t sign = (byte_val >> 7) & 1;
    uint8_t exp = (byte_val >> 3) & 0x0F;
    uint8_t mant = byte_val & 0x07;
    if (exp == 0x0F && mant == 0x07) {
        return sign ? -NAN : NAN;
    }
    float fsign = sign ? -1.0f : 1.0f;
    if (exp == 0) {
        return fsign * std::ldexp(static_cast<float>(mant) / 8.0f, -6);
    }
    return fsign * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, static_cast<int>(exp) - 7);
}

inline uint8_t float_to_fp8_e4m3(float f) noexcept {
    if (std::isnan(f)) return 0x7F;
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp > 8) { // overflow -> max normal 448
        return static_cast<uint8_t>((sign << 7) | 0x7E);
    }
    if (exp < -9) { // underflow to 0
        return static_cast<uint8_t>(sign << 7);
    }

    if (exp < -6) { // subnormal
        int shift = -6 - exp;
        uint32_t full_mant = (1u << 23) | mant;
        uint32_t m = (full_mant >> (20 + shift));
        // rounding
        if ((full_mant >> (19 + shift)) & 1) m++;
        if (m > 7) m = 7;
        return static_cast<uint8_t>((sign << 7) | m);
    }

    uint8_t e = static_cast<uint8_t>(exp + 7);
    uint32_t m = mant >> 20;
    if ((mant >> 19) & 1) { // round to nearest even
        m++;
        if (m > 7) {
            m = 0;
            e++;
            if (e >= 15) return static_cast<uint8_t>((sign << 7) | 0x7E);
        }
    }
    return static_cast<uint8_t>((sign << 7) | (e << 3) | m);
}

// Convert FP16 to float and vice versa
inline float fp16_to_float(uint16_t h) noexcept {
    uint32_t sign = (h >> 15) & 0x0001;
    uint32_t exp = (h >> 10) & 0x001f;
    uint32_t mant = h & 0x03ff;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t res = sign << 31;
            float out;
            std::memcpy(&out, &res, sizeof(float));
            return out;
        }
        while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        uint32_t res = (sign << 31) | 0x7f800000 | (mant << 13);
        float out;
        std::memcpy(&out, &res, sizeof(float));
        return out;
    }

    exp = exp + (127 - 15);
    mant = mant << 13;
    uint32_t res = (sign << 31) | (exp << 23) | mant;
    float out;
    std::memcpy(&out, &res, sizeof(float));
    return out;
}

inline uint16_t float_to_fp16(float f) noexcept {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127;
    uint32_t mant = x & 0x7fffff;

    if (exp > 15) {
        return static_cast<uint16_t>(sign | 0x7c00);
    }
    if (exp < -14) {
        if (exp < -24) return static_cast<uint16_t>(sign);
        mant |= 0x800000;
        int shift = -14 - exp;
        uint32_t half_mant = mant >> (shift + 13);
        return static_cast<uint16_t>(sign | half_mant);
    }

    uint16_t h_exp = static_cast<uint16_t>((exp + 15) << 10);
    uint16_t h_mant = static_cast<uint16_t>(mant >> 13);
    return static_cast<uint16_t>(sign | h_exp | h_mant);
}

} // namespace luisa::ntc

namespace ntc = luisa::ntc;
