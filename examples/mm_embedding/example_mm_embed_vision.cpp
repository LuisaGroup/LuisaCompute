// mm_embedding engine -- interactive multimodal image & text embedding demo.
//
// Usage:
//   example_mm_embed_vision [backend: dx|vk] [model.mmegm] [tokenizer.json] [image.png] [optional_prompt]

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>

#include "mm_embedding/embed/session.h"
#include "mm_embedding/embed/image_processor.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::mm_embed;

int main(int argc, char **argv) {
    luisa::string backend = (argc > 1) ? argv[1] : "dx";
    std::filesystem::path egm_path = (argc > 2) ? argv[2] : "model.mmegm";
    std::filesystem::path tok_path = (argc > 3) ? argv[3] : "tokenizer.json";
    std::filesystem::path img_path = (argc > 4) ? argv[4] : "examples/mm_embedding/data/forest.png";
    luisa::string prompt = (argc > 5) ? argv[5] : "";

    std::cout << "========================================================\n"
              << " LuisaCompute EmbeddingGemma-2 Multimodal Demo (Phase 2)\n"
              << " Backend: " << backend << "\n"
              << " Image:   " << img_path.string() << "\n"
              << "========================================================\n"
              << std::endl;

    if (!std::filesystem::exists(egm_path)) {
        std::cerr << "Error: model package not found at '" << egm_path.string() << "'\n";
        return 1;
    }
    if (!std::filesystem::exists(tok_path)) {
        std::cerr << "Error: tokenizer not found at '" << tok_path.string() << "'\n";
        return 1;
    }
    if (!std::filesystem::exists(img_path)) {
        std::cerr << "Error: image not found at '" << img_path.string() << "'\n";
        return 1;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);

    SessionConfig config{};
    config.enable_vision = true; // Selective Encoder Loading

    auto session = Session::create(device, egm_path, tok_path, config);
    if (!session->has_vision()) {
        std::cerr << "Error: loaded package does not contain Vision Tower weights.\n";
        return 1;
    }

    std::cout << "\nEncoding image '" << img_path.filename().string() << "'..." << std::endl;
    auto img_emb = session->encode_image(img_path);

    float norm = 0.0f;
    for (float v : img_emb.values) norm += v * v;
    norm = std::sqrt(norm);

    std::cout << "  Embedding Dimension: " << img_emb.dimension << "\n"
              << "  L2 Norm:             " << norm << "\n"
              << "  First 8 values:      [";
    for (size_t i = 0; i < 8 && i < img_emb.values.size(); i++) {
        std::cout << (i ? ", " : "") << std::fixed << std::setprecision(5) << img_emb.values[i];
    }
    std::cout << ", ...]\n" << std::endl;

    if (!prompt.empty()) {
        std::cout << "Encoding text prompt: \"" << prompt << "\"..." << std::endl;
        auto txt_emb = session->encode(prompt);

        float sim = 0.0f;
        for (uint32_t d = 0; d < 768u; d++) {
            sim += img_emb.values[d] * txt_emb.values[d];
        }
        std::cout << "  Cosine Similarity(Image, Text): " << std::fixed << std::setprecision(4)
                  << sim << std::endl;

        std::cout << "\nEncoding multimodal query (Image + Text prompt)..." << std::endl;
        auto mm_emb = session->encode_multimodal(img_path, prompt);
        float mm_norm = 0.0f;
        for (float v : mm_emb.values) mm_norm += v * v;
        std::cout << "  Multimodal Embedding Norm: " << std::sqrt(mm_norm) << std::endl;
    }

    std::cout << "\nDemo completed successfully." << std::endl;
    return 0;
}
