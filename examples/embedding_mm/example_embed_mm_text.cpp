// embedding_mm engine -- Text Embeddings, Task Prefixes & MRL Truncation Demo.
//
// Mirrors Google's official EmbeddingGemma-2 tutorial (Text & MRL sections):
//   1. Selective Encoder Loading: Text-only mode (~270M params, vision=false, audio=false)
//   2. Asymmetric Retrieval Prompts: encode_query ("task: search result | query: ...")
//      vs encode_document ("title: none | text: ...")
//   3. Matryoshka Representation Learning (MRL): 768 -> 512 -> 256 -> 128 truncation
//
// Usage:
//   example_embed_mm_text [backend: dx|cuda|vk] [model.mmegm] [tokenizer.json]
//   example_embed_mm_text --backend dx --query "What causes the northern lights?" --dim 256

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>

#include "embedding_mm/embed/session.h"
#include "embedding_mm/embed/tokenizer.h"

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
    luisa::string custom_query;
    uint32_t custom_dim = 0u;

    int pos = 0;
    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if ((arg == "--backend" || arg == "-b") && i + 1 < argc) {
            backend = argv[++i];
        } else if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            egm_path = argv[++i];
        } else if ((arg == "--tokenizer" || arg == "-t") && i + 1 < argc) {
            tok_path = argv[++i];
        } else if ((arg == "--query" || arg == "-q") && i + 1 < argc) {
            custom_query = argv[++i];
        } else if ((arg == "--dim" || arg == "-d") && i + 1 < argc) {
            custom_dim = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (!arg.starts_with("-")) {
            if (pos == 0) backend = argv[i];
            else if (pos == 1) egm_path = argv[i];
            else if (pos == 2) tok_path = argv[i];
            pos++;
        }
    }

    egm_path = resolve_path(egm_path);
    tok_path = resolve_path(tok_path);

    std::cout << "================================================================\n"
              << " LuisaCompute EmbeddingGemma-2: Text & MRL Truncation Demo\n"
              << " Selective Encoder Loading: Text-Only (~270M Parameters)\n"
              << " Backend: " << backend << "\n"
              << "================================================================\n"
              << std::endl;

    if (!std::filesystem::exists(egm_path) || !std::filesystem::exists(tok_path)) {
        std::cerr << "Error: Model files ('" << egm_path.string() << "' or '"
                  << tok_path.string() << "') not found.\n";
        return 1;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);

    // Load Text-only encoder (~270M params), skipping vision (~170M) and audio (~300M) towers
    SessionConfig cfg{};
    cfg.enable_vision = false;
    cfg.enable_audio = false;
    auto session = Session::create(device, egm_path, tok_path, cfg);

    // 1. Official Quick Start Example (SearchQuery vs Document prompt prefixes)
    luisa::string_view query = custom_query.empty()
                                   ? luisa::string_view{"What causes the northern lights?"}
                                   : luisa::string_view{custom_query};
    const luisa::string_view documents[] = {
        "The northern lights are caused by charged particles from the sun.",
        "Matryoshka Representation Learning trains embeddings that can be truncated to smaller dimensions.",
        "San Francisco's Golden Gate Bridge is an iconic suspension bridge spanning the Golden Gate strait.",
        "Roses are red, violets are blue, sugar is sweet, and so are you."
    };
    constexpr size_t num_docs = sizeof(documents) / sizeof(documents[0]);

    std::cout << "[1] Asymmetric Retrieval (encode_query vs encode_document)\n"
              << "  Query: \"" << query << "\"\n"
              << "  Candidate Documents:\n";
    for (size_t i = 0; i < num_docs; i++) {
        std::cout << "    [" << i + 1 << "] \"" << documents[i] << "\"\n";
    }
    std::cout << std::endl;

    // 2. Matryoshka Representation Learning (MRL) dimensions: 768, 512, 256, 128
    std::cout << "[2] Matryoshka Representation Learning (MRL) Dimension Truncation\n";
    const uint32_t mrl_dims[] = {768u, 512u, 256u, 128u};

    for (uint32_t dim : mrl_dims) {
        if (custom_dim != 0u && dim != custom_dim) continue;

        EncodeOptions opt{};
        opt.target_dimension = dim;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto q_emb = session->encode_query(query, opt);
        auto d_embs = session->encode_documents(luisa::span{documents, num_docs}, opt);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::printf("  --- MRL Dimension %u (GPU encode latency: %.2f ms) ---\n", dim, ms);
        for (size_t i = 0; i < num_docs; i++) {
            float sim = Session::similarity(q_emb, d_embs[i]);
            std::printf("    Doc [%zu] Similarity: %.4f%s\n",
                        i + 1, sim, (i == 0 ? "  <-- Top-1 Match" : ""));
        }
        std::cout << std::endl;
    }

    std::cout << "Text & MRL demo completed successfully." << std::endl;
    return 0;
}
