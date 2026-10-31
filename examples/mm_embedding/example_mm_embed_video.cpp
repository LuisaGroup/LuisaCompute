// mm_embedding engine -- interactive video & multimodal video embedding demo.
//
// Usage:
//   example_mm_embed_video [backend: dx|vk] [model.mmegm] [tokenizer.json] [video.mp4 or frame_dir] [optional_prompt]

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>

#include "mm_embedding/embed/session.h"
#include "mm_embedding/embed/video_processor.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::mm_embed;

int main(int argc, char **argv) {
    luisa::string backend = "dx";
    std::filesystem::path egm_path = "model.mmegm";
    std::filesystem::path tok_path = "tokenizer.json";
    std::filesystem::path vid_path = "examples/mm_embedding/data/video/sample_01_nature_ocean.mp4";
    luisa::string prompt = "";

    // Parse named flags or positional args
    for (int i = 1; i < argc; ++i) {
        luisa::string_view arg = argv[i];
        if ((arg == "--backend" || arg == "-b") && i + 1 < argc) {
            backend = argv[++i];
        } else if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            egm_path = argv[++i];
        } else if ((arg == "--tokenizer" || arg == "-t") && i + 1 < argc) {
            tok_path = argv[++i];
        } else if ((arg == "--video" || arg == "-v") && i + 1 < argc) {
            vid_path = argv[++i];
        } else if ((arg == "--prompt" || arg == "-p") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (!arg.starts_with("-")) {
            // Positional fallback
            if (backend == "dx" && i == 1) {
                backend = argv[1];
            } else if (i == 2) {
                egm_path = argv[2];
            } else if (i == 3) {
                tok_path = argv[3];
            } else if (i == 4) {
                vid_path = argv[4];
            } else if (i == 5) {
                prompt = argv[5];
            }
        }
    }

    auto resolve_path = [](std::filesystem::path p) -> std::filesystem::path {
        if (std::filesystem::exists(p)) return p;
        if (std::filesystem::exists(std::filesystem::path("..") / p)) return std::filesystem::path("..") / p;
        if (std::filesystem::exists(std::filesystem::path("../..") / p)) return std::filesystem::path("../..") / p;
        if (std::filesystem::exists(std::filesystem::path("../../..") / p)) return std::filesystem::path("../../..") / p;
        return p;
    };

    egm_path = resolve_path(egm_path);
    tok_path = resolve_path(tok_path);
    vid_path = resolve_path(vid_path);

    if (!std::filesystem::exists(egm_path)) {
        if (std::filesystem::exists("models/embeddinggemma-2/model.mmegm")) {
            egm_path = "models/embeddinggemma-2/model.mmegm";
        } else if (std::filesystem::exists("../../model.mmegm")) {
            egm_path = "../../model.mmegm";
        }
    }
    if (!std::filesystem::exists(tok_path)) {
        if (std::filesystem::exists("models/embeddinggemma-2/tokenizer.json")) {
            tok_path = "models/embeddinggemma-2/tokenizer.json";
        } else if (std::filesystem::exists("../../models/embeddinggemma-2/tokenizer.json")) {
            tok_path = "../../models/embeddinggemma-2/tokenizer.json";
        }
    }
    if (!std::filesystem::exists(vid_path)) {
        // Check frame directory fallback or fixtures
        auto stem_dir = vid_path.parent_path() / "fixtures" / vid_path.stem();
        if (std::filesystem::exists(stem_dir)) {
            vid_path = stem_dir;
        } else {
            auto parent_stem = vid_path.parent_path() / vid_path.stem();
            if (std::filesystem::exists(parent_stem)) {
                vid_path = parent_stem;
            }
        }
    }

    std::cout << "========================================================\n"
              << " LuisaCompute EmbeddingGemma-2 Video Embedding Demo (Phase 4)\n"
              << " Backend: " << backend << "\n"
              << " Model:   " << egm_path.string() << "\n"
              << " Video:   " << vid_path.string() << "\n"
              << " Prompt:  " << (prompt.empty() ? "(none - standalone video)" : prompt.c_str()) << "\n"
              << "========================================================\n"
              << std::endl;

    Context context{argv[0]};
    auto device = context.create_device(backend);

    SessionConfig config{};
    config.enable_vision = true;// Video modality utilizes Vision Tower weights

    auto session = Session::create(device, egm_path, tok_path, config);
    if (!session->has_vision()) {
        std::cerr << "Error: loaded package does not contain Vision Tower weights.\n";
        return 1;
    }

    VideoConfig vcfg{};
    vcfg.fps = 1.0f;
    vcfg.max_frames = 32u;
    vcfg.overflow_strategy = VideoOverflowStrategy::Uniform;

    std::cout << "Encoding video '" << vid_path.filename().string() << "'..." << std::endl;
    auto vid_emb = session->encode_video(vid_path, vcfg);

    float norm = 0.0f;
    for (float v : vid_emb.values) norm += v * v;
    norm = std::sqrt(norm);

    std::cout << "  Embedding Dimension: " << vid_emb.dimension << "\n"
              << "  L2 Norm:             " << norm << "\n"
              << "  First 8 values:      [";
    for (size_t i = 0; i < 8 && i < vid_emb.values.size(); i++) {
        std::cout << (i ? ", " : "") << std::fixed << std::setprecision(5) << vid_emb.values[i];
    }
    std::cout << ", ...]\n" << std::endl;

    // Matryoshka Representation Learning (MRL) dimension slices
    for (uint32_t dim : {512u, 256u, 128u}) {
        EncodeOptions mrl_opt{};
        mrl_opt.target_dimension = dim;
        auto mrl_emb = session->encode_video(vid_path, vcfg, mrl_opt);
        float mrl_norm = 0.0f;
        for (float v : mrl_emb.values) mrl_norm += v * v;
        std::cout << "  MRL Dimension " << dim << " Norm: " << std::sqrt(mrl_norm) << std::endl;
    }
    std::cout << std::endl;

    if (!prompt.empty()) {
        std::cout << "Encoding text prompt: \"" << prompt << "\"..." << std::endl;
        auto txt_emb = session->encode(prompt);

        float sim = 0.0f;
        for (uint32_t d = 0; d < 768u; d++) {
            sim += vid_emb.values[d] * txt_emb.values[d];
        }
        std::cout << "  Cosine Similarity(Video, Text): " << std::fixed << std::setprecision(4)
                  << sim << std::endl;

        std::cout << "\nEncoding multimodal query (Video + Text prompt)..." << std::endl;
        auto mm_emb = session->encode_multimodal_video(vid_path, prompt, vcfg);
        float mm_norm = 0.0f;
        for (float v : mm_emb.values) mm_norm += v * v;
        std::cout << "  Multimodal Embedding Norm: " << std::sqrt(mm_norm) << std::endl;
    }

    std::cout << "\nDemo completed successfully." << std::endl;
    return 0;
}
