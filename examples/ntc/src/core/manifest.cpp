/**
 * LuisaCompute NTC (RTX Neural Texture Compression)
 * Material Manifest Parser Implementation.
 */

#include "manifest.h"
#include <luisa/core/logging.h>
#include <yyjson.h>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include <fstream>
#include <cmath>

namespace luisa::ntc {

static BlockCompressedFormat parse_bcn(std::string_view str) {
    if (str == "BC1") return BlockCompressedFormat::BC1;
    if (str == "BC3") return BlockCompressedFormat::BC3;
    if (str == "BC4") return BlockCompressedFormat::BC4;
    if (str == "BC5") return BlockCompressedFormat::BC5;
    if (str == "BC7") return BlockCompressedFormat::BC7;
    return BlockCompressedFormat::None;
}

bool MaterialManifest::load(const std::filesystem::path &manifest_path, bool generate_mips) {
    manifestPath = std::filesystem::absolute(manifest_path);
    auto parent_dir = manifestPath.parent_path();

    std::ifstream file(manifestPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LUISA_ERROR_WITH_LOCATION("Failed to open Manifest file: {}", manifestPath.string());
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::string json_text(size, '\0');
    if (!file.read(json_text.data(), size)) {
        LUISA_ERROR_WITH_LOCATION("Failed to read Manifest file: {}", manifestPath.string());
        return false;
    }

    yyjson_read_err jerr;
    yyjson_doc *doc = yyjson_read_opts(json_text.data(), json_text.size(), 0, nullptr, &jerr);
    if (!doc) {
        LUISA_ERROR_WITH_LOCATION("Failed to parse Manifest JSON: {}", jerr.msg);
        return false;
    }

    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *val_texs = yyjson_obj_get(root, "textures");
    if (!val_texs || !yyjson_is_arr(val_texs)) {
        yyjson_doc_free(doc);
        LUISA_ERROR_WITH_LOCATION("Missing or invalid 'textures' array in Manifest.");
        return false;
    }

    textures.clear();
    totalChannels = 0;
    width = 0;
    height = 0;

    size_t idx, max;
    yyjson_val *t_item;
    yyjson_arr_foreach(val_texs, idx, max, t_item) {
        ManifestTexture tex;
        yyjson_val *v_fn = yyjson_obj_get(t_item, "fileName");
        if (v_fn && yyjson_is_str(v_fn)) tex.fileName = yyjson_get_str(v_fn);

        yyjson_val *v_name = yyjson_obj_get(t_item, "name");
        if (v_name && yyjson_is_str(v_name)) tex.name = yyjson_get_str(v_name);

        yyjson_val *v_srgb = yyjson_obj_get(t_item, "isSRGB");
        if (v_srgb && yyjson_is_bool(v_srgb)) tex.isSRGB = yyjson_get_bool(v_srgb);

        yyjson_val *v_bc = yyjson_obj_get(t_item, "bcFormat");
        if (v_bc && yyjson_is_str(v_bc)) tex.bcFormat = parse_bcn(yyjson_get_str(v_bc));

        yyjson_val *v_sem = yyjson_obj_get(t_item, "semantics");
        if (v_sem && yyjson_is_obj(v_sem)) {
            size_t s_idx, s_max;
            yyjson_val *key, *val;
            yyjson_obj_foreach(v_sem, s_idx, s_max, key, val) {
                if (yyjson_is_str(key) && yyjson_is_str(val)) {
                    tex.semantics[yyjson_get_str(key)] = yyjson_get_str(val);
                }
            }
        }

        // Load image using stb
        auto img_path = parent_dir / tex.fileName;
        int img_w = 0, img_h = 0, img_c = 0;
        stbi_uc *raw_bytes = stbi_load(img_path.string().c_str(), &img_w, &img_h, &img_c, 0);
        if (!raw_bytes) {
            LUISA_ERROR_WITH_LOCATION("Failed to load texture image: {}", img_path.string());
            yyjson_doc_free(doc);
            return false;
        }

        if (width == 0 && height == 0) {
            width = static_cast<uint32_t>(img_w);
            height = static_cast<uint32_t>(img_h);
        } else if (width != static_cast<uint32_t>(img_w) || height != static_cast<uint32_t>(img_h)) {
            LUISA_WARNING_WITH_LOCATION("Image size mismatch for {}: expected {}x{}, got {}x{}",
                                        tex.fileName, width, height, img_w, img_h);
        }

        tex.width = static_cast<uint32_t>(img_w);
        tex.height = static_cast<uint32_t>(img_h);
        tex.channels = static_cast<uint32_t>(img_c);
        tex.firstChannel = totalChannels;
        totalChannels += tex.channels;

        size_t pixel_count = static_cast<size_t>(img_w) * img_h * img_c;
        tex.floatPixels.resize(pixel_count);
        for (size_t i = 0; i < pixel_count; ++i) {
            float val = static_cast<float>(raw_bytes[i]) * (1.0f / 255.0f);
            if (tex.isSRGB && (i % img_c) < 3) {
                val = srgb_to_linear(val);
            }
            tex.floatPixels[i] = val;
        }

        stbi_image_free(raw_bytes);
        textures.push_back(std::move(tex));
    }

    yyjson_doc_free(doc);

    // Compute mip count
    numMips = generate_mips ? (static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1u) : 1u;
    numMips = std::min(numMips, NTC_MAX_MIPS);

    _generate_mipmaps();
    return true;
}

void MaterialManifest::_generate_mipmaps() {
    mipLevelData.resize(numMips);

    for (uint32_t mip = 0; mip < numMips; ++mip) {
        uint32_t mw = std::max(1u, width >> mip);
        uint32_t mh = std::max(1u, height >> mip);
        size_t total_floats = static_cast<size_t>(mw) * mh * NTC_MLP_OUTPUT_CHANNELS;
        mipLevelData[mip].assign(total_floats, 0.0f);
    }

    // Populate Mip 0
    size_t total_p0 = static_cast<size_t>(width) * height;
    for (const auto &tex : textures) {
        for (size_t p = 0; p < total_p0; ++p) {
            for (uint32_t c = 0; c < tex.channels; ++c) {
                float v = tex.floatPixels[p * tex.channels + c];
                mipLevelData[0][p * NTC_MLP_OUTPUT_CHANNELS + tex.firstChannel + c] = v;
            }
        }
    }

    // Downsample subsequent mips with box filtering
    for (uint32_t mip = 1; mip < numMips; ++mip) {
        uint32_t prev_w = std::max(1u, width >> (mip - 1));
        uint32_t prev_h = std::max(1u, height >> (mip - 1));
        uint32_t curr_w = std::max(1u, width >> mip);
        uint32_t curr_h = std::max(1u, height >> mip);

        for (uint32_t cy = 0; cy < curr_h; ++cy) {
            for (uint32_t cx = 0; cx < curr_w; ++cx) {
                uint32_t px0 = std::min(cx * 2, prev_w - 1);
                uint32_t px1 = std::min(cx * 2 + 1, prev_w - 1);
                uint32_t py0 = std::min(cy * 2, prev_h - 1);
                uint32_t py1 = std::min(cy * 2 + 1, prev_h - 1);

                size_t c_idx = (static_cast<size_t>(cy) * curr_w + cx) * NTC_MLP_OUTPUT_CHANNELS;
                size_t p00 = (static_cast<size_t>(py0) * prev_w + px0) * NTC_MLP_OUTPUT_CHANNELS;
                size_t p10 = (static_cast<size_t>(py0) * prev_w + px1) * NTC_MLP_OUTPUT_CHANNELS;
                size_t p01 = (static_cast<size_t>(py1) * prev_w + px0) * NTC_MLP_OUTPUT_CHANNELS;
                size_t p11 = (static_cast<size_t>(py1) * prev_w + px1) * NTC_MLP_OUTPUT_CHANNELS;

                for (uint32_t ch = 0; ch < totalChannels; ++ch) {
                    float v = (mipLevelData[mip - 1][p00 + ch] +
                               mipLevelData[mip - 1][p10 + ch] +
                               mipLevelData[mip - 1][p01 + ch] +
                               mipLevelData[mip - 1][p11 + ch]) * 0.25f;
                    mipLevelData[mip][c_idx + ch] = v;
                }
            }
        }
    }
}

uint32_t MaterialManifest::get_valid_channel_mask() const noexcept {
    return (totalChannels >= 32) ? ~0u : ((1u << totalChannels) - 1u);
}

uint32_t MaterialManifest::get_channel_colorspaces_mask() const noexcept {
    uint32_t mask = 0;
    for (const auto &tex : textures) {
        if (tex.isSRGB) {
            for (uint32_t c = 0; c < std::min(tex.channels, 3u); ++c) {
                uint32_t ch_idx = tex.firstChannel + c;
                if (ch_idx < 16) {
                    mask |= (1u << (ch_idx * 2)); // sRGB = 1
                }
            }
        }
    }
    return mask;
}

bool MaterialManifest::save(const std::filesystem::path &manifest_path) const {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_val *arr_texs = yyjson_mut_arr(doc);
    for (const auto &tex : textures) {
        yyjson_mut_val *t_obj = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_strcpy(doc, t_obj, "fileName", tex.fileName.c_str());
        yyjson_mut_obj_add_strcpy(doc, t_obj, "name", tex.name.c_str());
        yyjson_mut_obj_add_bool(doc, t_obj, "isSRGB", tex.isSRGB);
        yyjson_mut_obj_add_str(doc, t_obj, "bcFormat", bc_format_to_string(tex.bcFormat));

        yyjson_mut_val *s_obj = yyjson_mut_obj(doc);
        for (const auto &[k, v] : tex.semantics) {
            yyjson_mut_obj_add_strcpy(doc, s_obj, k.c_str(), v.c_str());
        }
        yyjson_mut_obj_add_val(doc, t_obj, "semantics", s_obj);
        yyjson_mut_arr_append(arr_texs, t_obj);
    }
    yyjson_mut_obj_add_val(doc, root, "textures", arr_texs);

    size_t len = 0;
    char *json = yyjson_mut_write(doc, YYJSON_WRITE_PRETTY, &len);
    if (!json) {
        yyjson_mut_doc_free(doc);
        return false;
    }

    std::ofstream out(manifest_path);
    out.write(json, len);
    free(json);
    yyjson_mut_doc_free(doc);
    return true;
}

} // namespace luisa::ntc
