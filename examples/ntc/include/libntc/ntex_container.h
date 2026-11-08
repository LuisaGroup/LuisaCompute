/**
 * LuisaCompute NTC (RTX Neural Texture Compression)
 * NTEX v3 container parser and writer using yyjson.
 * 100% binary and schema compatible with official NVIDIA ntc-cli and ntc-explorer.
 */

#pragma once

#include <libntc/common.h>
#include <filesystem>
#include <string>
#include <vector>
#include <optional>
#include <memory>

namespace luisa::ntc {

struct BufferView {
    uint64_t offset{0};
    uint64_t storedSize{0};
    std::optional<CompressionType> compression;
    std::optional<uint64_t> uncompressedSize;
    std::optional<uint32_t> crc32;
};

struct LatentShape {
    int32_t gridSizeScale{4};
    int32_t numFeatures{16};
};

struct MLPLayer {
    uint32_t inputChannels{0};
    uint32_t outputChannels{0};
    uint32_t weightView{0};
    std::optional<uint32_t> scaleView;
    uint32_t biasView{0};
    MlpDataType weightType{MlpDataType::Int8};
    MlpDataType biasType{MlpDataType::Int8};
    std::optional<MlpDataType> scaleType;
};

struct MLP {
    std::vector<MLPLayer> layers;
    std::optional<ActivationType> activation{ActivationType::HGELUClamp};
    std::optional<MatrixLayout> weightLayout{MatrixLayout::RowMajor};
};

struct BCModeBuffer {
    uint32_t mipLevel{0};
    uint32_t view{0};
};

struct TextureDesc {
    std::string name;
    uint32_t firstChannel{0};
    uint32_t numChannels{0};
    std::optional<ChannelFormat> channelFormat{ChannelFormat::UNORM8};
    std::optional<ColorSpace> rgbColorSpace{ColorSpace::Linear};
    std::optional<ColorSpace> alphaColorSpace{ColorSpace::Linear};
    std::optional<BlockCompressedFormat> bcFormat{BlockCompressedFormat::None};
    std::vector<BCModeBuffer> bcModeBuffers;
};

struct ChannelDesc {
    std::optional<ColorSpace> colorSpace{ColorSpace::Linear};
};

struct LatentImageDesc {
    uint32_t width{0};
    uint32_t height{0};
    std::vector<uint32_t> layerViews;
};

struct ColorImageData {
    uint32_t view{0};
    std::optional<ChannelFormat> uncompressedFormat;
    std::optional<BlockCompressedFormat> bcFormat;
    std::optional<uint32_t> rowPitch;
    std::optional<uint32_t> pixelStride;
    std::optional<uint32_t> numChannels;
};

struct ColorMipDesc {
    uint32_t width{0};
    uint32_t height{0};
    int32_t latentMip{0};
    float positionLod{0.0f};
    float positionScale{1.0f};
    std::optional<ColorImageData> combinedColorData;
    std::vector<ColorImageData> perTextureColorData;
};

class NtexContainer {
public:
    uint32_t schemaVersion{NTC_SCHEMA_VERSION};
    uint32_t width{0};
    uint32_t height{0};
    uint32_t numChannels{0};
    uint32_t numColorMips{0};
    LatentShape latentShape;
    std::vector<MLP> mlpVersions;
    std::vector<TextureDesc> textures;
    std::vector<ChannelDesc> channels;
    std::vector<LatentImageDesc> latents;
    std::vector<ColorMipDesc> colorMips;
    std::vector<BufferView> views;

    // Binary payload storage
    std::vector<uint8_t> binaryChunk;

public:
    NtexContainer() = default;

    // Deserialization
    bool load_from_file(const std::filesystem::path &path, std::string &error);
    bool load_from_memory(luisa::span<const uint8_t> data, std::string &error);

    // Serialization
    bool save_to_file(const std::filesystem::path &path, std::string &error) const;
    std::vector<uint8_t> save_to_memory(std::string &error) const;

    // View data access
    [[nodiscard]] luisa::span<const uint8_t> get_view_data(uint32_t view_idx) const;
    [[nodiscard]] std::vector<uint8_t> decompress_view_data(uint32_t view_idx) const;

    // Helper queries
    [[nodiscard]] const MLP *find_mlp(InferenceWeightType type) const;
    [[nodiscard]] bool has_weight_type(InferenceWeightType type) const;
    [[nodiscard]] uint32_t get_valid_channel_mask() const;
    [[nodiscard]] uint32_t get_channel_colorspaces_mask() const;

    // Builder helpers for saving
    uint32_t allocate_view_and_append_data(const void *data, size_t size, CompressionType comp = CompressionType::None);
};

} // namespace luisa::ntc
