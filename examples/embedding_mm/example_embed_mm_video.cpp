// embedding_mm engine -- Official Video Embedding Demo.
//
// Mirrors Google's official EmbeddingGemma-2 tutorial:
//   - Section "Working with Video":
//     Input: cat_dozing.mp4 (default fps=1, max_frames=32)
//     Candidates:
//       1. "a cat dozing"
//       2. "a cat playing with a small lizard"
//       3. "a sports car racing on a highway"
//   - Section "Customizing Video Processing":
//     Input: cat_dozing_16frames.mp4 (max_frames=16, fps=None, overflow_strategy="uniform", add_timestamps=True)
//
// Usage:
//   example_embed_mm_video [backend: dx|cuda|vk] [model.mmegm] [tokenizer.json]
//   example_embed_mm_video --backend dx --video path/to/video.mp4.mmfeat --prompt "Summary of <|video|>"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>

#include "embedding_mm/embed/session.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::embed_mm;

static std::filesystem::path resolve_path(const std::filesystem::path &p) {
    if (std::filesystem::exists(p)) return p;
    for (const char *prefix : {"..", "../..", "../../.."}) {
        auto cand = std::filesystem::path(prefix) / p;
        if (std::filesystem::exists(cand)) return cand;
    }
    return p;
}

int main(int argc, char **argv) {
    luisa::string backend = "dx";
    std::filesystem::path egm_path = "model.mmegm";
    std::filesystem::path tok_path = "tokenizer.json";
    std::filesystem::path data_dir = "examples/embedding_mm/data";
    std::filesystem::path custom_video;
    luisa::string prompt;

    int pos = 0;
    for (int i = 1; i < argc; ++i) {
        luisa::string_view arg = argv[i];
        if ((arg == "--backend" || arg == "-b") && i + 1 < argc) {
            backend = argv[++i];
        } else if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            egm_path = argv[++i];
        } else if ((arg == "--tokenizer" || arg == "-t") && i + 1 < argc) {
            tok_path = argv[++i];
        } else if ((arg == "--data-dir" || arg == "-d") && i + 1 < argc) {
            data_dir = argv[++i];
        } else if ((arg == "--video" || arg == "-v") && i + 1 < argc) {
            custom_video = argv[++i];
        } else if ((arg == "--prompt" || arg == "-p") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (!arg.starts_with("-")) {
            if (pos == 0) backend = argv[i];
            else if (pos == 1) egm_path = argv[i];
            else if (pos == 2) tok_path = argv[i];
            else if (pos == 3) custom_video = argv[i];
            else if (pos == 4) prompt = argv[i];
            pos++;
        }
    }

    egm_path = resolve_path(egm_path);
    tok_path = resolve_path(tok_path);
    data_dir = resolve_path(data_dir);

    std::cout << "================================================================\n"
              << " LuisaCompute EmbeddingGemma-2: Working with Video Demo\n"
              << " Selective Encoder Loading: Text + Vision (~440M Parameters)\n"
              << " Backend: " << backend << "\n"
              << "================================================================\n"
              << std::endl;

    if (!std::filesystem::exists(egm_path) || !std::filesystem::exists(tok_path)) {
        std::cerr << "Error: model or tokenizer not found.\n";
        return 1;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);

    SessionConfig config{};
    config.enable_vision = true;
    config.enable_audio = false;// Skip Audio Conformer (~300M params)

    auto session = Session::create(device, egm_path, tok_path, config);
    if (!session->has_vision()) {
        std::cerr << "Error: loaded package does not contain Vision Tower weights.\n";
        return 1;
    }

    if (!custom_video.empty()) {
        custom_video = FeatureLoader::resolve_feat_path(resolve_path(custom_video));
        std::cout << "Encoding custom video '" << custom_video.string() << "'..." << std::endl;
        auto vid_emb = session->encode_video(custom_video);
        std::printf("  Dimension: %u, First 4 values: [%.5f, %.5f, %.5f, %.5f]\n",
                    vid_emb.dimension, vid_emb.values[0], vid_emb.values[1],
                    vid_emb.values[2], vid_emb.values[3]);
        if (!prompt.empty()) {
            auto mm_emb = session->encode_multimodal_video(custom_video, prompt);
            std::printf("  Multimodal Video+Prompt Similarity: %.4f\n",
                        Session::similarity(vid_emb, mm_emb));
        }
        return 0;
    }

    auto vid_default = data_dir / "cat_dozing.mp4.mmfeat";
    auto vid_16f = data_dir / "cat_dozing_16frames.mp4.mmfeat";
    if (!std::filesystem::exists(vid_default)) {
        std::cerr << "Error: preprocessed video bundle not found in '" << data_dir.string() << "'.\n"
                  << "Run: python examples/embedding_mm/tools/preprocess.py\n";
        return 1;
    }

    const luisa::string_view candidates[] = {
        "a cat dozing",
        "a cat playing with a small lizard",
        "a sports car racing on a highway"
    };
    auto cand_embs = session->encode_batch(luisa::span{candidates, 3u});

    // 1. Official Section: Working with Video (default fps=1, max_frames=32)
    std::cout << "[1] Working with Video (Default: fps=1, max_frames=32)\n"
              << "  Input Video: cat_dozing.mp4\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    auto emb_def = session->encode_video(vid_default);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_def = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("  Encoded on GPU in %.2f ms:\n", ms_def);
    for (size_t i = 0; i < 3; i++) {
        std::printf("    vs \"%-33s\" : %.4f%s\n",
                    candidates[i].data(), Session::similarity(emb_def, cand_embs[i]),
                    (i == 0 ? "  <-- Top-1 Match" : ""));
    }
    std::cout << std::endl;

    // 2. Official Section: Customizing Video Processing (16 frames, uniform, add_timestamps=True)
    if (std::filesystem::exists(vid_16f)) {
        std::cout << "[2] Customizing Video Processing (max_frames=16, uniform, add_timestamps=True)\n"
                  << "  Input Video: cat_dozing.mp4 (16 timestamped frames)\n";
        auto t2 = std::chrono::high_resolution_clock::now();
        auto emb_16f = session->encode_video(vid_16f);
        auto t3 = std::chrono::high_resolution_clock::now();
        double ms_16f = std::chrono::duration<double, std::milli>(t3 - t2).count();

        std::printf("  Encoded on GPU in %.2f ms:\n", ms_16f);
        for (size_t i = 0; i < 3; i++) {
            std::printf("    vs \"%-33s\" : %.4f%s\n",
                        candidates[i].data(), Session::similarity(emb_16f, cand_embs[i]),
                        (i == 0 ? "  <-- Top-1 Match" : ""));
        }
        std::cout << std::endl;
    }

    std::cout << "Video demo completed successfully." << std::endl;
    return 0;
}
