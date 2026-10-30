// mm_embedding engine -- interactive text embedding demo.
//
// Usage:
//   example_mm_embed_text <backend: dx|vk|cuda|cpu> [model.mmegm] [tokenizer.json]

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include "mm_embedding/embed/session.h"
#include "mm_embedding/embed/tokenizer.h"

int main(int argc, char **argv) {
    luisa::string backend = (argc > 1) ? argv[1] : "dx";
    std::filesystem::path egm_path = (argc > 2) ? argv[2] : "model.mmegm";
    std::filesystem::path tok_path = (argc > 3) ? argv[3] : "tokenizer.json";

    std::cout << "========================================================\n"
              << " LuisaCompute Multimodal Embedding Engine (mm_embedding)\n"
              << " Model: Google EmbeddingGemma-2 (~270M Text Backbone)\n"
              << " Backend: " << backend << "\n"
              << "========================================================\n"
              << std::endl;

    if (!std::filesystem::exists(egm_path) || !std::filesystem::exists(tok_path)) {
        std::cout << "Notice: Model files ('" << egm_path.string() << "' or '"
                  << tok_path.string() << "') not found.\n"
                  << "To run real GPU inference, pack the model first:\n"
                  << "  python examples/mm_embedding/tools/pack_model.py google/embeddinggemma-2 model.mmegm\n"
                  << "  " << argv[0] << " " << backend << " model.mmegm tokenizer.json\n"
                  << std::endl;
        return 0;
    }

    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(backend);

    auto session = luisa::mm_embed::Session::create(device, egm_path, tok_path);

    const luisa::string_view query = "How does Matryoshka Representation Learning work?";
    const luisa::string_view documents[] = {
        "Matryoshka Representation Learning trains embeddings that can be truncated to smaller dimensions with minimal loss.",
        "Deep neural networks use gradient descent and backpropagation to optimize loss functions.",
        "Gemma 2 incorporates Grouped-Query Attention and alternating sliding window layers.",
        "Paris is the capital of France and is known for its cultural landmarks."
    };
    constexpr size_t num_docs = sizeof(documents) / sizeof(documents[0]);

    std::cout << "Query: \"" << query << "\"\n" << std::endl;
    std::cout << "Documents:" << std::endl;
    for (size_t i = 0; i < num_docs; i++) {
        std::cout << "  [" << i + 1 << "] " << documents[i] << std::endl;
    }
    auto q_toks = session->tokenizer().encode(query);
    std::cout << "Query Tokens (" << q_toks.size() << "): [";
    for (size_t i = 0; i < q_toks.size(); i++) {
        std::cout << (i > 0 ? ", " : "") << q_toks[i];
    }
    std::cout << "]\n" << std::endl;

    // Test different MRL dimensions
    const uint32_t mrl_dims[] = {768u, 512u, 256u, 128u};

    for (uint32_t dim : mrl_dims) {
        luisa::mm_embed::EncodeOptions opt{};
        opt.target_dimension = dim;

        auto q_emb = session->encode(query, opt);
        auto d_embs = session->encode_batch(luisa::span{documents, num_docs}, opt);

        std::cout << "--- Cosine Similarities at MRL Dimension " << dim << " ---" << std::endl;

        for (size_t i = 0; i < num_docs; i++) {
            float dot = 0.0f;
            for (size_t d = 0; d < dim; d++) {
                dot += q_emb.values[d] * d_embs[i].values[d];
            }
            std::printf("  Doc [%zu]: score = %.4f\n", i + 1, dot);
        }
        std::cout << std::endl;
    }

    std::cout << "Inference completed successfully." << std::endl;
    return 0;
}
