// embedding_mm engine -- C++ golden intermediate verification tool.
//
// Usage:
//   example_embed_mm_dump_golden <backend> <model.mmegm> <tokenizer.json> <golden_dense.bin>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include "embedding_mm/embed/session.h"

int main(int argc, char **argv) {
    if (argc < 5) {
        std::fprintf(stderr, "Usage: %s <backend: dx|vk|cuda|cpu> <model.mmegm> <tokenizer.json> <golden_dense.bin>\n", argv[0]);
        return 1;
    }

    luisa::string backend = argv[1];
    std::filesystem::path egm_path = argv[2];
    std::filesystem::path tok_path = argv[3];
    std::filesystem::path golden_bin_path = argv[4];

    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(backend);

    std::cout << "Initializing Session on backend '" << backend << "'..." << std::endl;
    auto session = luisa::embed_mm::Session::create(device, egm_path, tok_path);

    const luisa::string_view test_texts[] = {
        "Hello world, welcome to multimodal embeddings with Gemma!",
        "The quick brown fox jumps over the lazy dog.",
        "LuisaCompute is a high-performance, cross-platform rendering and compute framework.",
        "Artificial intelligence models map diverse modalities into a unified latent space.",
        "Matryoshka Representation Learning allows dynamic truncation of embedding dimensions.",
    };
    constexpr size_t num_texts = sizeof(test_texts) / sizeof(test_texts[0]);

    uint32_t target_dim = 768u;
    if (argc >= 6) {
        target_dim = static_cast<uint32_t>(std::atoi(argv[5]));
    }

    std::cout << "Encoding " << num_texts << " reference sentences (dim=" << target_dim << ")..." << std::endl;
    luisa::embed_mm::EncodeOptions opt{};
    opt.target_dimension = target_dim;
    auto embeddings = session->encode_batch(luisa::span{test_texts, num_texts}, opt);

    // Read golden binary
    std::ifstream f_golden(golden_bin_path, std::ios::binary);
    if (!f_golden) {
        std::fprintf(stderr, "Warning: golden binary '%s' could not be opened for comparison.\n",
                     golden_bin_path.string().c_str());
        return 0;
    }

    luisa::vector<float> golden_data(num_texts * target_dim);
    f_golden.read(reinterpret_cast<char *>(golden_data.data()), golden_data.size() * sizeof(float));

    std::cout << "=== Parity Verification Results (dim=" << target_dim << ") ===" << std::endl;
    for (size_t b = 0; b < num_texts; b++) {
        float dot = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        float max_diff = 0.0f;

        for (size_t d = 0; d < target_dim; d++) {
            float v_test = embeddings[b].values[d];
            float v_gold = golden_data[b * target_dim + d];

            dot += v_test * v_gold;
            norm_a += v_test * v_test;
            norm_b += v_gold * v_gold;
            max_diff = std::max(max_diff, std::abs(v_test - v_gold));
        }

        float cosine = dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-12f);
        std::cout << "  Sentence [" << b << "]: cosine=" << cosine
                  << ", max_abs_diff=" << max_diff << std::endl;
        std::cout << "    v_test[:10]: ";
        for (size_t d = 0; d < 10; d++) std::cout << embeddings[b].values[d] << " ";
        std::cout << "\n    v_gold[:10]: ";
        for (size_t d = 0; d < 10; d++) std::cout << golden_data[b * target_dim + d] << " ";
        std::cout << std::endl;
    }

    return 0;
}
