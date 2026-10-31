// embedding_mm engine -- Preprocessed Multimodal Feature Tensor Bundle Loader.
#include "embedding_mm/embed/feature_loader.h"

#include <cstdio>
#include <fstream>

#include <luisa/core/logging.h>

namespace luisa::embed_mm {

namespace {

constexpr uint32_t kFeatureMagic = 0x54464D4Du;// "MMFT"

FeatureBundleHeader read_header(std::ifstream &fs, const std::filesystem::path &path) {
    FeatureBundleHeader hdr{};
    fs.read(reinterpret_cast<char *>(&hdr), sizeof(FeatureBundleHeader));
    if (!fs || hdr.magic != kFeatureMagic) {
        LUISA_ERROR("FeatureLoader: invalid or corrupted .mmfeat header in '{}'. "
                    "Run 'python examples/embedding_mm/tools/preprocess.py' first.",
                    path.string());
    }
    if (hdr.version != 1u) {
        LUISA_ERROR("FeatureLoader: unsupported .mmfeat version {} in '{}'.",
                    hdr.version, path.string());
    }
    return hdr;
}

ImageFeatures read_image_payload(std::ifstream &fs, const FeatureBundleHeader &hdr, const std::filesystem::path &path) {
    ImageFeatures out{};
    out.max_patches = hdr.max_patches;
    out.num_patches_w = hdr.num_patches_w;
    out.num_patches_h = hdr.num_patches_h;
    out.num_soft_tokens = hdr.num_soft_tokens;

    size_t px_count = static_cast<size_t>(hdr.max_patches) * hdr.feat_dim;
    size_t pos_count = static_cast<size_t>(hdr.max_patches) * 2u;

    out.pixel_values.resize(px_count);
    out.position_ids.resize(pos_count);

    fs.read(reinterpret_cast<char *>(out.pixel_values.data()), px_count * sizeof(float));
    fs.read(reinterpret_cast<char *>(out.position_ids.data()), pos_count * sizeof(int32_t));
    if (!fs) {
        LUISA_ERROR("FeatureLoader: truncated Image payload in '{}'.", path.string());
    }
    return out;
}

AudioFeatures read_audio_payload(std::ifstream &fs, const FeatureBundleHeader &hdr, const std::filesystem::path &path) {
    AudioFeatures out{};
    out.num_frames = hdr.num_frames;
    out.num_features = hdr.feat_dim;
    out.num_soft_tokens = hdr.num_soft_tokens;

    size_t feat_count = static_cast<size_t>(hdr.num_frames) * hdr.feat_dim;
    out.features.resize(feat_count);
    fs.read(reinterpret_cast<char *>(out.features.data()), feat_count * sizeof(float));
    if (!fs) {
        LUISA_ERROR("FeatureLoader: truncated Audio payload in '{}'.", path.string());
    }
    return out;
}

VideoFeatures read_video_payload(std::ifstream &fs, const FeatureBundleHeader &hdr, const std::filesystem::path &path) {
    VideoFeatures out{};
    out.num_frames = hdr.num_frames;
    out.max_patches = hdr.max_patches;
    out.num_patches_w = hdr.num_patches_w;
    out.num_patches_h = hdr.num_patches_h;
    out.num_soft_tokens_per_frame = hdr.num_soft_tokens;
    out.total_soft_tokens = hdr.num_frames * hdr.num_soft_tokens;

    size_t frame_px_count = static_cast<size_t>(hdr.max_patches) * hdr.feat_dim;
    size_t frame_pos_count = static_cast<size_t>(hdr.max_patches) * 2u;

    luisa::vector<float> all_px(static_cast<size_t>(hdr.num_frames) * frame_px_count);
    luisa::vector<int32_t> all_pos(static_cast<size_t>(hdr.num_frames) * frame_pos_count);

    fs.read(reinterpret_cast<char *>(all_px.data()), all_px.size() * sizeof(float));
    fs.read(reinterpret_cast<char *>(all_pos.data()), all_pos.size() * sizeof(int32_t));
    if (hdr.reserved[0] > 0u) {
        out.input_ids.resize(hdr.reserved[0]);
        fs.read(reinterpret_cast<char *>(out.input_ids.data()), hdr.reserved[0] * sizeof(uint32_t));
    }
    if (!fs) {
        LUISA_ERROR("FeatureLoader: truncated Video payload in '{}'.", path.string());
    }

    out.frames.resize(hdr.num_frames);
    for (uint32_t f = 0u; f < hdr.num_frames; f++) {
        out.frames[f].pixel_values.assign(
            all_px.begin() + f * frame_px_count,
            all_px.begin() + (f + 1u) * frame_px_count);
        out.frames[f].position_ids.assign(
            all_pos.begin() + f * frame_pos_count,
            all_pos.begin() + (f + 1u) * frame_pos_count);
    }
    return out;
}

}// namespace

std::filesystem::path FeatureLoader::resolve_feat_path(const std::filesystem::path &path) {
    if (std::filesystem::exists(path) && !std::filesystem::is_directory(path)) {
        if (path.extension() == ".mmfeat") {
            return path;
        }
        auto sidecar1 = std::filesystem::path(path.string() + ".mmfeat");
        if (std::filesystem::exists(sidecar1)) {
            return sidecar1;
        }
        auto sidecar2 = path;
        sidecar2.replace_extension(".mmfeat");
        if (std::filesystem::exists(sidecar2)) {
            return sidecar2;
        }
        return path;
    }

    auto cand1 = std::filesystem::path(path.string() + ".mmfeat");
    if (std::filesystem::exists(cand1)) {
        return cand1;
    }
    auto cand2 = path;
    cand2.replace_extension(".mmfeat");
    if (std::filesystem::exists(cand2)) {
        return cand2;
    }
    return path;
}

ModalityKind FeatureLoader::peek_modality(const std::filesystem::path &path) {
    auto resolved = resolve_feat_path(path);
    std::ifstream fs(resolved, std::ios::binary);
    if (!fs) {
        LUISA_ERROR("FeatureLoader::peek_modality: cannot open '{}'. "
                    "Preprocess raw media first using tools/preprocess.py.",
                    resolved.string());
    }
    auto hdr = read_header(fs, resolved);
    return static_cast<ModalityKind>(hdr.modality);
}

ImageFeatures FeatureLoader::load_image(const std::filesystem::path &path) {
    auto resolved = resolve_feat_path(path);
    std::ifstream fs(resolved, std::ios::binary);
    if (!fs) {
        LUISA_ERROR("FeatureLoader::load_image: cannot open '{}'. "
                    "Preprocess raw image first via tools/preprocess.py.",
                    resolved.string());
    }
    auto hdr = read_header(fs, resolved);
    if (static_cast<ModalityKind>(hdr.modality) != ModalityKind::Image) {
        LUISA_ERROR("FeatureLoader::load_image: '{}' is not an Image feature bundle (modality={}).",
                    resolved.string(), hdr.modality);
    }
    return read_image_payload(fs, hdr, resolved);
}

AudioFeatures FeatureLoader::load_audio(const std::filesystem::path &path) {
    auto resolved = resolve_feat_path(path);
    std::ifstream fs(resolved, std::ios::binary);
    if (!fs) {
        LUISA_ERROR("FeatureLoader::load_audio: cannot open '{}'. "
                    "Preprocess raw audio first via tools/preprocess.py.",
                    resolved.string());
    }
    auto hdr = read_header(fs, resolved);
    if (static_cast<ModalityKind>(hdr.modality) != ModalityKind::Audio) {
        LUISA_ERROR("FeatureLoader::load_audio: '{}' is not an Audio feature bundle (modality={}).",
                    resolved.string(), hdr.modality);
    }
    return read_audio_payload(fs, hdr, resolved);
}

VideoFeatures FeatureLoader::load_video(const std::filesystem::path &path) {
    auto resolved = resolve_feat_path(path);
    std::ifstream fs(resolved, std::ios::binary);
    if (!fs) {
        LUISA_ERROR("FeatureLoader::load_video: cannot open '{}'. "
                    "Preprocess raw video first via tools/preprocess.py.",
                    resolved.string());
    }
    auto hdr = read_header(fs, resolved);
    if (static_cast<ModalityKind>(hdr.modality) != ModalityKind::Video) {
        LUISA_ERROR("FeatureLoader::load_video: '{}' is not a Video feature bundle (modality={}).",
                    resolved.string(), hdr.modality);
    }
    return read_video_payload(fs, hdr, resolved);
}

MultimodalFeatures FeatureLoader::load_multimodal(const std::filesystem::path &path) {
    auto resolved = resolve_feat_path(path);
    std::ifstream fs(resolved, std::ios::binary);
    if (!fs) {
        LUISA_ERROR("FeatureLoader::load_multimodal: cannot open '{}'. "
                    "Preprocess multimodal inputs first via tools/preprocess.py.",
                    resolved.string());
    }
    auto hdr = read_header(fs, resolved);
    if (static_cast<ModalityKind>(hdr.modality) != ModalityKind::Multimodal) {
        LUISA_ERROR("FeatureLoader::load_multimodal: '{}' is not a Multimodal bundle (modality={}).",
                    resolved.string(), hdr.modality);
    }

    uint32_t seq_len = hdr.num_frames;
    uint32_t num_images = hdr.max_patches;
    uint32_t num_audios = hdr.num_patches_w;
    uint32_t num_videos = hdr.num_patches_h;

    MultimodalFeatures out{};
    out.input_ids.resize(seq_len);
    fs.read(reinterpret_cast<char *>(out.input_ids.data()), seq_len * sizeof(uint32_t));
    if (!fs) {
        LUISA_ERROR("FeatureLoader::load_multimodal: truncated input_ids in '{}'.", resolved.string());
    }

    out.images.reserve(num_images);
    for (uint32_t i = 0u; i < num_images; i++) {
        auto ihdr = read_header(fs, resolved);
        out.images.push_back(read_image_payload(fs, ihdr, resolved));
    }

    out.audios.reserve(num_audios);
    for (uint32_t a = 0u; a < num_audios; a++) {
        auto ahdr = read_header(fs, resolved);
        out.audios.push_back(read_audio_payload(fs, ahdr, resolved));
    }

    out.videos.reserve(num_videos);
    for (uint32_t v = 0u; v < num_videos; v++) {
        auto vhdr = read_header(fs, resolved);
        out.videos.push_back(read_video_payload(fs, vhdr, resolved));
    }

    return out;
}

}// namespace luisa::embed_mm
