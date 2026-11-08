/**
 * LuisaCompute NTC (RTX Neural Texture Compression)
 * Material Manifest Parser & Multi-Mip Image Loader.
 */

#pragma once

#include <libntc/common.h>
#include <filesystem>
#include <string>
#include <vector>
#include <map>

namespace luisa::ntc {

struct ManifestTexture {
    std::string fileName;
    std::string name;
    bool isSRGB{false};
    BlockCompressedFormat bcFormat{BlockCompressedFormat::None};
    std::map<std::string, std::string> semantics;

    // Loaded host image data (Mip 0)
    uint32_t width{0};
    uint32_t height{0};
    uint32_t channels{0};
    uint32_t firstChannel{0};
    std::vector<float> floatPixels; // normalized [0, 1] (or linear)
};

class MaterialManifest {
public:
    std::filesystem::path manifestPath;
    uint32_t width{0};
    uint32_t height{0};
    uint32_t totalChannels{0};
    uint32_t numMips{1};
    std::vector<ManifestTexture> textures;

    // Flattened multi-mip reference data:
    // For each mip: vector of floats of size (mip_w * mip_h * 16)
    std::vector<std::vector<float>> mipLevelData;

public:
    MaterialManifest() = default;

    // Load from Manifest.json file
    bool load(const std::filesystem::path &manifest_path, bool generate_mips = true);

    // Save manifest JSON
    bool save(const std::filesystem::path &manifest_path) const;

    [[nodiscard]] uint32_t get_valid_channel_mask() const noexcept;
    [[nodiscard]] uint32_t get_channel_colorspaces_mask() const noexcept;

private:
    void _generate_mipmaps();
};

} // namespace luisa::ntc
