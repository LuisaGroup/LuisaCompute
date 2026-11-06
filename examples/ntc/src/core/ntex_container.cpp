/**
 * LuisaCompute NTC (RTX Neural Texture Compression)
 * NTEX v3 container parser and writer implementation using yyjson.
 */

#include "ntex_container.h"
#include "gdeflate_codec.h"
#include <luisa/core/logging.h>
#include <yyjson.h>

#include <fstream>
#include <sstream>

namespace luisa::ntc {



bool NtexContainer::load_from_file(const std::filesystem::path &path, std::string &error) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        error = "Failed to open file: " + path.string();
        return false;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
        error = "Failed to read file: " + path.string();
        return false;
    }
    return load_from_memory(buffer, error);
}

bool NtexContainer::load_from_memory(luisa::span<const uint8_t> data, std::string &error) {
    if (data.size() < sizeof(FileHeader)) {
        error = "Buffer smaller than FileHeader";
        return false;
    }

    FileHeader header;
    std::memcpy(&header, data.data(), sizeof(FileHeader));

    if (header.signature != NTC_SIGNATURE_VALUE) {
        error = "Invalid signature: expected NTEX (0x5845544E)";
        return false;
    }
    if (header.version != NTC_CONTAINER_VERSION) {
        error = "Incompatible container version: expected 0x100";
        return false;
    }

    if (header.jsonChunkOffset + header.jsonChunkSize > data.size()) {
        error = "JSON chunk extends past file boundary";
        return false;
    }
    if (header.binaryChunkOffset + header.binaryChunkSize > data.size()) {
        error = "Binary chunk extends past file boundary";
        return false;
    }

    // Read JSON chunk
    const char *json_bytes = reinterpret_cast<const char *>(data.data() + header.jsonChunkOffset);
    size_t json_len = header.jsonChunkSize;
    if (json_len > 0 && json_bytes[json_len - 1] == '\0') {
        json_len--;
    }

    yyjson_read_err jerr;
    yyjson_doc *doc = yyjson_read_opts(const_cast<char *>(json_bytes), json_len, 0, nullptr, &jerr);
    if (!doc) {
        error = std::string("Failed to parse JSON chunk: ") + jerr.msg;
        return false;
    }

    yyjson_val *root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        error = "JSON root is not an object";
        return false;
    }

    schemaVersion = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(root, "schemaVersion")));
    width = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(root, "width")));
    height = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(root, "height")));
    numChannels = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(root, "numChannels")));
    
    yyjson_val *val_color_mips = yyjson_obj_get(root, "numColorMips");
    numColorMips = val_color_mips ? static_cast<uint32_t>(yyjson_get_uint(val_color_mips)) : 1u;

    // Latent shape
    yyjson_val *val_latent_shape = yyjson_obj_get(root, "latentShape");
    if (val_latent_shape && yyjson_is_obj(val_latent_shape)) {
        yyjson_val *gscale = yyjson_obj_get(val_latent_shape, "gridSizeScale");
        latentShape.gridSizeScale = gscale ? static_cast<int32_t>(yyjson_get_int(gscale)) : 4;
        yyjson_val *nfeat = yyjson_obj_get(val_latent_shape, "numFeatures");
        latentShape.numFeatures = nfeat ? static_cast<int32_t>(yyjson_get_int(nfeat)) : 16;
    }

    // MLP versions
    mlpVersions.clear();
    yyjson_val *val_mlps = yyjson_obj_get(root, "mlpVersions");
    if (val_mlps && yyjson_is_arr(val_mlps)) {
        size_t idx, max;
        yyjson_val *mlp_item;
        yyjson_arr_foreach(val_mlps, idx, max, mlp_item) {
            MLP mlp;
            yyjson_val *act = yyjson_obj_get(mlp_item, "activation");
            if (act && yyjson_is_str(act)) {
                mlp.activation = ActivationType::HGELUClamp;
            }
            yyjson_val *layout = yyjson_obj_get(mlp_item, "weightLayout");
            if (layout && yyjson_is_str(layout)) {
                std::string_view lstr = yyjson_get_str(layout);
                mlp.weightLayout = (lstr == "ColumnMajor") ? MatrixLayout::ColumnMajor : MatrixLayout::RowMajor;
            }
            yyjson_val *layers_arr = yyjson_obj_get(mlp_item, "layers");
            if (layers_arr && yyjson_is_arr(layers_arr)) {
                size_t l_idx, l_max;
                yyjson_val *layer_item;
                yyjson_arr_foreach(layers_arr, l_idx, l_max, layer_item) {
                    MLPLayer layer;
                    layer.inputChannels = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(layer_item, "inputChannels")));
                    layer.outputChannels = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(layer_item, "outputChannels")));
                    layer.weightView = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(layer_item, "weightView")));
                    yyjson_val *sview = yyjson_obj_get(layer_item, "scaleView");
                    if (sview && yyjson_is_uint(sview)) layer.scaleView = static_cast<uint32_t>(yyjson_get_uint(sview));
                    layer.biasView = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(layer_item, "biasView")));

                    yyjson_val *wtype = yyjson_obj_get(layer_item, "weightType");
                    if (wtype && yyjson_is_str(wtype)) layer.weightType = string_to_mlp_data_type(yyjson_get_str(wtype));

                    yyjson_val *btype = yyjson_obj_get(layer_item, "biasType");
                    if (btype && yyjson_is_str(btype)) layer.biasType = string_to_mlp_data_type(yyjson_get_str(btype));

                    yyjson_val *stype = yyjson_obj_get(layer_item, "scaleType");
                    if (stype && yyjson_is_str(stype)) layer.scaleType = string_to_mlp_data_type(yyjson_get_str(stype));

                    mlp.layers.push_back(layer);
                }
            }
            mlpVersions.push_back(mlp);
        }
    }

    // Textures
    textures.clear();
    yyjson_val *val_textures = yyjson_obj_get(root, "textures");
    if (val_textures && yyjson_is_arr(val_textures)) {
        size_t idx, max;
        yyjson_val *tex_item;
        yyjson_arr_foreach(val_textures, idx, max, tex_item) {
            TextureDesc tex;
            yyjson_val *name = yyjson_obj_get(tex_item, "name");
            if (name && yyjson_is_str(name)) tex.name = yyjson_get_str(name);
            tex.firstChannel = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(tex_item, "firstChannel")));
            tex.numChannels = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(tex_item, "numChannels")));

            yyjson_val *fmt = yyjson_obj_get(tex_item, "channelFormat");
            if (fmt && yyjson_is_str(fmt)) tex.channelFormat = string_to_channel_format(yyjson_get_str(fmt));

            yyjson_val *rgb_cs = yyjson_obj_get(tex_item, "rgbColorSpace");
            if (rgb_cs && yyjson_is_str(rgb_cs)) tex.rgbColorSpace = string_to_colorspace(yyjson_get_str(rgb_cs));

            yyjson_val *alpha_cs = yyjson_obj_get(tex_item, "alphaColorSpace");
            if (alpha_cs && yyjson_is_str(alpha_cs)) tex.alphaColorSpace = string_to_colorspace(yyjson_get_str(alpha_cs));

            yyjson_val *bcf = yyjson_obj_get(tex_item, "bcFormat");
            if (bcf && yyjson_is_str(bcf)) tex.bcFormat = string_to_bc_format(yyjson_get_str(bcf));

            yyjson_val *bc_modes = yyjson_obj_get(tex_item, "bcModeBuffers");
            if (bc_modes && yyjson_is_arr(bc_modes)) {
                size_t m_idx, m_max;
                yyjson_val *m_item;
                yyjson_arr_foreach(bc_modes, m_idx, m_max, m_item) {
                    BCModeBuffer mb;
                    mb.mipLevel = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(m_item, "mipLevel")));
                    mb.view = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(m_item, "view")));
                    tex.bcModeBuffers.push_back(mb);
                }
            }
            textures.push_back(tex);
        }
    }

    // Channels
    channels.clear();
    yyjson_val *val_channels = yyjson_obj_get(root, "channels");
    if (val_channels && yyjson_is_arr(val_channels)) {
        size_t idx, max;
        yyjson_val *ch_item;
        yyjson_arr_foreach(val_channels, idx, max, ch_item) {
            ChannelDesc ch;
            yyjson_val *cs = yyjson_obj_get(ch_item, "colorSpace");
            if (cs && yyjson_is_str(cs)) ch.colorSpace = string_to_colorspace(yyjson_get_str(cs));
            channels.push_back(ch);
        }
    }

    // Latents
    latents.clear();
    yyjson_val *val_latents = yyjson_obj_get(root, "latents");
    if (val_latents && yyjson_is_arr(val_latents)) {
        size_t idx, max;
        yyjson_val *lat_item;
        yyjson_arr_foreach(val_latents, idx, max, lat_item) {
            LatentImageDesc lat;
            lat.width = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(lat_item, "width")));
            lat.height = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(lat_item, "height")));
            yyjson_val *lviews = yyjson_obj_get(lat_item, "layerViews");
            if (lviews && yyjson_is_arr(lviews)) {
                size_t v_idx, v_max;
                yyjson_val *v_item;
                yyjson_arr_foreach(lviews, v_idx, v_max, v_item) {
                    lat.layerViews.push_back(static_cast<uint32_t>(yyjson_get_uint(v_item)));
                }
            }
            latents.push_back(lat);
        }
    }

    // Color Mips
    colorMips.clear();
    yyjson_val *val_mips = yyjson_obj_get(root, "colorMips");
    if (val_mips && yyjson_is_arr(val_mips)) {
        size_t idx, max;
        yyjson_val *mip_item;
        yyjson_arr_foreach(val_mips, idx, max, mip_item) {
            ColorMipDesc mip;
            yyjson_val *w = yyjson_obj_get(mip_item, "width");
            mip.width = w ? static_cast<uint32_t>(yyjson_get_uint(w)) : 0;
            yyjson_val *h = yyjson_obj_get(mip_item, "height");
            mip.height = h ? static_cast<uint32_t>(yyjson_get_uint(h)) : 0;
            yyjson_val *lm = yyjson_obj_get(mip_item, "latentMip");
            mip.latentMip = lm ? static_cast<int32_t>(yyjson_get_int(lm)) : 0;
            yyjson_val *plod = yyjson_obj_get(mip_item, "positionLod");
            mip.positionLod = plod ? static_cast<float>(yyjson_get_num(plod)) : 0.0f;
            yyjson_val *pscale = yyjson_obj_get(mip_item, "positionScale");
            mip.positionScale = pscale ? static_cast<float>(yyjson_get_num(pscale)) : 1.0f;
            colorMips.push_back(mip);
        }
    }

    // Buffer Views
    views.clear();
    yyjson_val *val_views = yyjson_obj_get(root, "views");
    if (val_views && yyjson_is_arr(val_views)) {
        size_t idx, max;
        yyjson_val *view_item;
        yyjson_arr_foreach(val_views, idx, max, view_item) {
            BufferView bv;
            bv.offset = static_cast<uint64_t>(yyjson_get_uint(yyjson_obj_get(view_item, "offset")));
            bv.storedSize = static_cast<uint64_t>(yyjson_get_uint(yyjson_obj_get(view_item, "storedSize")));
            yyjson_val *comp = yyjson_obj_get(view_item, "compression");
            if (comp && yyjson_is_str(comp)) bv.compression = string_to_compression_type(yyjson_get_str(comp));
            yyjson_val *unc_size = yyjson_obj_get(view_item, "uncompressedSize");
            if (unc_size && yyjson_is_uint(unc_size)) bv.uncompressedSize = static_cast<uint64_t>(yyjson_get_uint(unc_size));
            yyjson_val *crc = yyjson_obj_get(view_item, "crc32");
            if (crc && yyjson_is_uint(crc)) bv.crc32 = static_cast<uint32_t>(yyjson_get_uint(crc));
            views.push_back(bv);
        }
    }

    yyjson_doc_free(doc);

    // Read binary chunk
    binaryChunk.resize(header.binaryChunkSize);
    std::memcpy(binaryChunk.data(), data.data() + header.binaryChunkOffset, header.binaryChunkSize);

    return true;
}

std::vector<uint8_t> NtexContainer::save_to_memory(std::string &error) const {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_obj_add_uint(doc, root, "schemaVersion", schemaVersion);
    yyjson_mut_obj_add_uint(doc, root, "width", width);
    yyjson_mut_obj_add_uint(doc, root, "height", height);
    yyjson_mut_obj_add_uint(doc, root, "numChannels", numChannels);
    yyjson_mut_obj_add_uint(doc, root, "numColorMips", numColorMips);

    // Latent shape
    yyjson_mut_val *val_ls = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_uint(doc, val_ls, "gridSizeScale", latentShape.gridSizeScale);
    yyjson_mut_obj_add_uint(doc, val_ls, "numFeatures", latentShape.numFeatures);
    yyjson_mut_obj_add_val(doc, root, "latentShape", val_ls);

    // MLP versions
    yyjson_mut_val *arr_mlps = yyjson_mut_arr(doc);
    for (const auto &mlp : mlpVersions) {
        yyjson_mut_val *mlp_obj = yyjson_mut_obj(doc);
        if (mlp.activation) {
            yyjson_mut_obj_add_str(doc, mlp_obj, "activation", "HGELUClamp");
        }
        if (mlp.weightLayout) {
            yyjson_mut_obj_add_str(doc, mlp_obj, "weightLayout",
                (*mlp.weightLayout == MatrixLayout::ColumnMajor) ? "ColumnMajor" : "RowMajor");
        }
        yyjson_mut_val *arr_layers = yyjson_mut_arr(doc);
        for (const auto &layer : mlp.layers) {
            yyjson_mut_val *layer_obj = yyjson_mut_obj(doc);
            yyjson_mut_obj_add_uint(doc, layer_obj, "inputChannels", layer.inputChannels);
            yyjson_mut_obj_add_uint(doc, layer_obj, "outputChannels", layer.outputChannels);
            yyjson_mut_obj_add_uint(doc, layer_obj, "weightView", layer.weightView);
            if (layer.scaleView) {
                yyjson_mut_obj_add_uint(doc, layer_obj, "scaleView", *layer.scaleView);
            }
            yyjson_mut_obj_add_uint(doc, layer_obj, "biasView", layer.biasView);
            yyjson_mut_obj_add_str(doc, layer_obj, "weightType", mlp_data_type_to_string(layer.weightType));
            yyjson_mut_obj_add_str(doc, layer_obj, "biasType", mlp_data_type_to_string(layer.biasType));
            if (layer.scaleType) {
                yyjson_mut_obj_add_str(doc, layer_obj, "scaleType", mlp_data_type_to_string(*layer.scaleType));
            }
            yyjson_mut_arr_append(arr_layers, layer_obj);
        }
        yyjson_mut_obj_add_val(doc, mlp_obj, "layers", arr_layers);
        yyjson_mut_arr_append(arr_mlps, mlp_obj);
    }
    yyjson_mut_obj_add_val(doc, root, "mlpVersions", arr_mlps);

    // Textures
    yyjson_mut_val *arr_textures = yyjson_mut_arr(doc);
    for (const auto &tex : textures) {
        yyjson_mut_val *tex_obj = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_strcpy(doc, tex_obj, "name", tex.name.c_str());
        yyjson_mut_obj_add_uint(doc, tex_obj, "firstChannel", tex.firstChannel);
        yyjson_mut_obj_add_uint(doc, tex_obj, "numChannels", tex.numChannels);
        if (tex.channelFormat) {
            yyjson_mut_obj_add_str(doc, tex_obj, "channelFormat", channel_format_to_string(*tex.channelFormat));
        }
        if (tex.rgbColorSpace && *tex.rgbColorSpace != ColorSpace::Linear) {
            yyjson_mut_obj_add_str(doc, tex_obj, "rgbColorSpace", colorspace_to_string(*tex.rgbColorSpace));
        }
        if (tex.alphaColorSpace && *tex.alphaColorSpace != ColorSpace::Linear) {
            yyjson_mut_obj_add_str(doc, tex_obj, "alphaColorSpace", colorspace_to_string(*tex.alphaColorSpace));
        }
        if (tex.bcFormat && *tex.bcFormat != BlockCompressedFormat::None) {
            yyjson_mut_obj_add_str(doc, tex_obj, "bcFormat", bc_format_to_string(*tex.bcFormat));
        }
        if (!tex.bcModeBuffers.empty()) {
            yyjson_mut_val *arr_modes = yyjson_mut_arr(doc);
            for (const auto &mb : tex.bcModeBuffers) {
                yyjson_mut_val *mb_obj = yyjson_mut_obj(doc);
                yyjson_mut_obj_add_uint(doc, mb_obj, "mipLevel", mb.mipLevel);
                yyjson_mut_obj_add_uint(doc, mb_obj, "view", mb.view);
                yyjson_mut_arr_append(arr_modes, mb_obj);
            }
            yyjson_mut_obj_add_val(doc, tex_obj, "bcModeBuffers", arr_modes);
        }
        yyjson_mut_arr_append(arr_textures, tex_obj);
    }
    yyjson_mut_obj_add_val(doc, root, "textures", arr_textures);

    // Channels
    yyjson_mut_val *arr_channels = yyjson_mut_arr(doc);
    for (const auto &ch : channels) {
        yyjson_mut_val *ch_obj = yyjson_mut_obj(doc);
        if (ch.colorSpace && *ch.colorSpace != ColorSpace::Linear) {
            yyjson_mut_obj_add_str(doc, ch_obj, "colorSpace", colorspace_to_string(*ch.colorSpace));
        }
        yyjson_mut_arr_append(arr_channels, ch_obj);
    }
    yyjson_mut_obj_add_val(doc, root, "channels", arr_channels);

    // Latents
    yyjson_mut_val *arr_latents = yyjson_mut_arr(doc);
    for (const auto &lat : latents) {
        yyjson_mut_val *lat_obj = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_uint(doc, lat_obj, "width", lat.width);
        yyjson_mut_obj_add_uint(doc, lat_obj, "height", lat.height);
        yyjson_mut_val *arr_lviews = yyjson_mut_arr(doc);
        for (uint32_t lv : lat.layerViews) {
            yyjson_mut_arr_add_uint(doc, arr_lviews, lv);
        }
        yyjson_mut_obj_add_val(doc, lat_obj, "layerViews", arr_lviews);
        yyjson_mut_arr_append(arr_latents, lat_obj);
    }
    yyjson_mut_obj_add_val(doc, root, "latents", arr_latents);

    // Color Mips
    yyjson_mut_val *arr_colormips = yyjson_mut_arr(doc);
    for (const auto &cm : colorMips) {
        yyjson_mut_val *cm_obj = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_uint(doc, cm_obj, "width", cm.width);
        yyjson_mut_obj_add_uint(doc, cm_obj, "height", cm.height);
        yyjson_mut_obj_add_int(doc, cm_obj, "latentMip", cm.latentMip);
        yyjson_mut_obj_add_real(doc, cm_obj, "positionLod", cm.positionLod);
        yyjson_mut_obj_add_real(doc, cm_obj, "positionScale", cm.positionScale);
        yyjson_mut_arr_append(arr_colormips, cm_obj);
    }
    yyjson_mut_obj_add_val(doc, root, "colorMips", arr_colormips);

    // Buffer Views
    yyjson_mut_val *arr_views = yyjson_mut_arr(doc);
    for (const auto &v : views) {
        yyjson_mut_val *v_obj = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_uint(doc, v_obj, "offset", v.offset);
        yyjson_mut_obj_add_uint(doc, v_obj, "storedSize", v.storedSize);
        if (v.compression && *v.compression != CompressionType::None) {
            yyjson_mut_obj_add_str(doc, v_obj, "compression", compression_type_to_string(*v.compression));
        }
        if (v.uncompressedSize) {
            yyjson_mut_obj_add_uint(doc, v_obj, "uncompressedSize", *v.uncompressedSize);
        }
        if (v.crc32) {
            yyjson_mut_obj_add_uint(doc, v_obj, "crc32", *v.crc32);
        }
        yyjson_mut_arr_append(arr_views, v_obj);
    }
    yyjson_mut_obj_add_val(doc, root, "views", arr_views);

    // Write JSON string
    size_t json_len = 0;
    char *json_str = yyjson_mut_write(doc, 0, &json_len);
    if (!json_str) {
        yyjson_mut_doc_free(doc);
        error = "Failed to serialize JSON document";
        return {};
    }

    // Prepare container output
    FileHeader header{};
    header.signature = NTC_SIGNATURE_VALUE;
    header.version = NTC_CONTAINER_VERSION;
    header.jsonChunkOffset = sizeof(FileHeader);
    header.jsonChunkSize = json_len + 1; // including null terminator
    header.binaryChunkOffset = round_up_4(header.jsonChunkOffset + header.jsonChunkSize);
    header.binaryChunkSize = binaryChunk.size();

    size_t total_size = header.binaryChunkOffset + header.binaryChunkSize;
    std::vector<uint8_t> output(total_size, 0);

    // Copy FileHeader
    std::memcpy(output.data(), &header, sizeof(FileHeader));

    // Copy JSON chunk (including null byte)
    std::memcpy(output.data() + header.jsonChunkOffset, json_str, json_len);
    output[header.jsonChunkOffset + json_len] = '\0';

    // Copy Binary chunk
    if (!binaryChunk.empty()) {
        std::memcpy(output.data() + header.binaryChunkOffset, binaryChunk.data(), binaryChunk.size());
    }

    free(json_str);
    yyjson_mut_doc_free(doc);
    return output;
}

bool NtexContainer::save_to_file(const std::filesystem::path &path, std::string &error) const {
    auto bytes = save_to_memory(error);
    if (bytes.empty() && !error.empty()) {
        return false;
    }
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        error = "Failed to open output file: " + path.string();
        return false;
    }
    file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    return true;
}

luisa::span<const uint8_t> NtexContainer::get_view_data(uint32_t view_idx) const {
    if (view_idx >= views.size()) return {};
    const auto &view = views[view_idx];
    if (view.offset + view.storedSize > binaryChunk.size()) return {};
    return {binaryChunk.data() + view.offset, view.storedSize};
}

std::vector<uint8_t> NtexContainer::decompress_view_data(uint32_t view_idx) const {
    auto raw = get_view_data(view_idx);
    if (raw.empty()) return {};
    const auto &view = views[view_idx];
    if (!view.compression || *view.compression == CompressionType::None) {
        return {raw.begin(), raw.end()};
    }
    if (*view.compression == CompressionType::GDeflate) {
        size_t unc_size = view.uncompressedSize.value_or(raw.size());
        std::vector<uint8_t> decomp(unc_size);
        if (decompress_gdeflate(raw.data(), raw.size(), decomp.data(), unc_size, view.crc32.value_or(0))) {
            return decomp;
        }
    }
    return {raw.begin(), raw.end()};
}

const MLP *NtexContainer::find_mlp(InferenceWeightType type) const {
    for (const auto &mlp : mlpVersions) {
        if (mlp.layers.empty()) continue;
        if (type == InferenceWeightType::GenericInt8 && mlp.layers[0].weightType == MlpDataType::Int8) {
            return &mlp;
        }
        if (type == InferenceWeightType::GenericFP8 && mlp.layers[0].weightType == MlpDataType::FloatE4M3) {
            return &mlp;
        }
    }
    return mlpVersions.empty() ? nullptr : &mlpVersions[0];
}

bool NtexContainer::has_weight_type(InferenceWeightType type) const {
    return find_mlp(type) != nullptr;
}

uint32_t NtexContainer::get_valid_channel_mask() const {
    uint32_t mask = 0;
    for (const auto &tex : textures) {
        uint32_t tex_mask = ((1u << tex.numChannels) - 1u) << tex.firstChannel;
        mask |= tex_mask;
    }
    return mask;
}

uint32_t NtexContainer::get_channel_colorspaces_mask() const {
    uint32_t mask = 0;
    for (size_t i = 0; i < channels.size() && i < NTC_MAX_CHANNELS; ++i) {
        ColorSpace cs = channels[i].colorSpace.value_or(ColorSpace::Linear);
        mask |= (static_cast<uint32_t>(cs) & 3u) << (i * 2);
    }
    return mask;
}

uint32_t NtexContainer::allocate_view_and_append_data(const void *data, size_t size, CompressionType comp) {
    BufferView bv;
    bv.offset = round_up_4(binaryChunk.size());
    binaryChunk.resize(bv.offset); // pad with 0s

    if (comp == CompressionType::GDeflate && size > 0) {
        std::vector<uint8_t> compressed;
        uint32_t crc = 0;
        if (compress_gdeflate(data, size, compressed, 9, &crc) && compressed.size() < size) {
            bv.compression = CompressionType::GDeflate;
            bv.storedSize = compressed.size();
            bv.uncompressedSize = size;
            bv.crc32 = crc;
            binaryChunk.insert(binaryChunk.end(), compressed.begin(), compressed.end());
            uint32_t idx = static_cast<uint32_t>(views.size());
            views.push_back(bv);
            return idx;
        }
    }

    // Default: uncompressed
    bv.compression = CompressionType::None;
    bv.storedSize = size;
    const auto *bytes = static_cast<const uint8_t *>(data);
    binaryChunk.insert(binaryChunk.end(), bytes, bytes + size);

    uint32_t idx = static_cast<uint32_t>(views.size());
    views.push_back(bv);
    return idx;
}

} // namespace luisa::ntc
