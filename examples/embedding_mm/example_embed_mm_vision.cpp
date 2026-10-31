// embedding_mm engine -- Official Image & Interleaved Multimodal Embedding Demo.
//
// Mirrors Google's official EmbeddingGemma-2 tutorial:
//   - Section "Working with Images":
//     Input: GoldenGate.png (default max_soft_tokens=280)
//     Candidates: [kitchen_painting.jpg, "San Francisco", "a cute cat walking", "a friendly dog sitting in the grass"]
//   - Section "Customizing Image Processing":
//     High-resolution GoldenGate_1120.png (max_soft_tokens=1120)
//   - Section "Interleaved Multimodal Inputs":
//     1. Dict-style multimodal query (Text + 2 Images + Audio)
//     2. Interleaved prompt ("Compare the scene in <|video|> with <|image|> and <|image|>.")
//     3. Media sequence ("<|image|><|audio|><|image|>")
//
// Usage:
//   example_embed_mm_vision [backend: dx|cuda|vk] [model.mmegm] [tokenizer.json]
//   example_embed_mm_vision --backend dx --image path/to/image.mmfeat --prompt "Describe this image"

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
    std::filesystem::path custom_image;
    luisa::string custom_prompt;

    int pos = 0;
    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if ((arg == "--backend" || arg == "-b") && i + 1 < argc) {
            backend = argv[++i];
        } else if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            egm_path = argv[++i];
        } else if ((arg == "--tokenizer" || arg == "-t") && i + 1 < argc) {
            tok_path = argv[++i];
        } else if ((arg == "--data-dir" || arg == "-d") && i + 1 < argc) {
            data_dir = argv[++i];
        } else if ((arg == "--image" || arg == "-i") && i + 1 < argc) {
            custom_image = argv[++i];
        } else if ((arg == "--prompt" || arg == "-p") && i + 1 < argc) {
            custom_prompt = argv[++i];
        } else if (!arg.starts_with("-")) {
            if (pos == 0) backend = argv[i];
            else if (pos == 1) egm_path = argv[i];
            else if (pos == 2) tok_path = argv[i];
            else if (pos == 3) custom_image = argv[i];
            else if (pos == 4) custom_prompt = argv[i];
            pos++;
        }
    }

    egm_path = resolve_path(egm_path);
    tok_path = resolve_path(tok_path);
    data_dir = resolve_path(data_dir);

    std::cout << "================================================================\n"
              << " LuisaCompute EmbeddingGemma-2: Vision & Interleaved Demo\n"
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
    config.enable_audio = true;
    auto session = Session::create(device, egm_path, tok_path, config);

    // Custom single-image mode if --image is specified
    if (!custom_image.empty()) {
        custom_image = resolve_path(custom_image);
        std::cout << "Encoding custom image '" << custom_image.string() << "'..." << std::endl;
        auto img_emb = session->encode_image(custom_image);
        std::printf("  Dimension: %u, First 4 values: [%.5f, %.5f, %.5f, %.5f]\n",
                    img_emb.dimension, img_emb.values[0], img_emb.values[1],
                    img_emb.values[2], img_emb.values[3]);
        if (!custom_prompt.empty()) {
            auto txt_emb = session->encode(custom_prompt);
            std::printf("  Similarity(Image, \"%s\") = %.4f\n",
                        custom_prompt.c_str(), Session::similarity(img_emb, txt_emb));
        }
        return 0;
    }

    auto img1_280 = data_dir / "GoldenGate.png.mmfeat";
    auto img2_280 = data_dir / "kitchen_painting.jpg.mmfeat";
    if (!std::filesystem::exists(img1_280)) {
        std::cerr << "Error: preprocessed official assets not found in '" << data_dir.string() << "'.\n"
                  << "Run: python examples/embedding_mm/tools/preprocess.py\n";
        return 1;
    }

    const luisa::string_view text_cands[] = {
        "San Francisco",
        "a cute cat walking",
        "a friendly dog sitting in the grass"
    };
    auto txt_embs = session->encode_batch(luisa::span{text_cands, 3u});

    // 1. Official Section: Working with Images (default max_soft_tokens=280)
    std::cout << "[1] Working with Images (Default max_soft_tokens=280)\n"
              << "  Query Image: GoldenGate.png\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    auto q_280 = session->encode_image(img1_280);
    auto c0_280 = session->encode_image(img2_280);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_280 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("  Encoded 2 images on GPU in %.2f ms:\n", ms_280);
    std::printf("     vs [Image] kitchen_painting.jpg             : %.4f\n", Session::similarity(q_280, c0_280));
    for (size_t i = 0; i < 3; i++) {
        std::printf("     vs [Text]  \"%-34s\" : %.4f%s\n",
                    text_cands[i].data(), Session::similarity(q_280, txt_embs[i]),
                    (i == 0 ? "  <-- Top-1 Match" : ""));
    }
    std::cout << std::endl;

    // 2. Official Section: Customizing Image Processing (max_soft_tokens=1120)
    auto img1_1120 = data_dir / "GoldenGate_1120.png.mmfeat";
    auto img2_1120 = data_dir / "kitchen_painting_1120.jpg.mmfeat";
    if (std::filesystem::exists(img1_1120) && std::filesystem::exists(img2_1120)) {
        std::cout << "[2] Customizing Image Processing (High-Res max_soft_tokens=1120)\n";
        auto t2 = std::chrono::high_resolution_clock::now();
        auto q_1120 = session->encode_image(img1_1120);
        auto c0_1120 = session->encode_image(img2_1120);
        auto t3 = std::chrono::high_resolution_clock::now();
        double ms_1120 = std::chrono::duration<double, std::milli>(t3 - t2).count();

        std::printf("  Encoded 2 high-res images (1089 soft tokens each) on GPU in %.2f ms:\n", ms_1120);
        std::printf("     vs [Image] kitchen_painting.jpg (1120)      : %.4f\n", Session::similarity(q_1120, c0_1120));
        for (size_t i = 0; i < 3; i++) {
            std::printf("     vs [Text]  \"%-34s\" : %.4f%s\n",
                        text_cands[i].data(), Session::similarity(q_1120, txt_embs[i]),
                        (i == 0 ? "  <-- Top-1 Match" : ""));
        }
        std::cout << std::endl;
    }

    // 3. Official Section: Interleaved Multimodal Inputs
    auto inter_mm = data_dir / "interleaved_multimodal.mmfeat";
    auto inter_pr = data_dir / "interleaved_prompt.mmfeat";
    auto inter_sq = data_dir / "interleaved_media_seq.mmfeat";
    if (std::filesystem::exists(inter_mm) && session->has_audio()) {
        std::cout << "[3] Interleaved Multimodal Inputs (Images + Audio + Video + Text)\n";
        auto emb_mm = session->encode_multimodal(inter_mm);
        auto emb_pr = session->encode_multimodal(inter_pr);
        auto emb_sq = session->encode_multimodal(inter_sq);

        const luisa::string_view inter_cands[] = {
            "San Francisco Golden Gate Bridge and a kitchen painting with a poem about roses",
            "A sleepy cat compared with the Golden Gate Bridge and a kitchen painting",
            "A formula one race car speeding on a wet track"
        };
        auto ic_embs = session->encode_batch(luisa::span{inter_cands, 3u});
        Embedding queries[] = {emb_mm, emb_pr, emb_sq};
        const char *q_names[] = {
            "query_multimodal (Text + 2 Images + Audio)",
            "query_interleaved (Video + 2 Images)",
            "query_media_sequence (<|image|><|audio|><|image|>)"
        };
        auto mat = Session::similarity_matrix(luisa::span{queries, 3u}, ic_embs);
        for (size_t q = 0; q < 3; q++) {
            std::printf("  %s:\n    Similarities: [%.4f, %.4f, %.4f]\n",
                        q_names[q], mat[q][0], mat[q][1], mat[q][2]);
        }
        std::cout << std::endl;
    }

    std::cout << "Vision & Interleaved demo completed successfully." << std::endl;
    return 0;
}
