// embedding_mm engine -- Official Audio Embedding Demo.
//
// Mirrors Google's official EmbeddingGemma-2 tutorial (Section "Working with Audio"):
//   - Selective Encoder Loading: Text + Audio (~570M params, vision=false, audio=true)
//   - Input: roses-are.wav (preprocessed via AutoProcessor into roses-are.wav.mmfeat)
//   - Candidates:
//     1. "a person speaking English"
//     2. "random static white noise"
//     3. "Roses are red, violets are blue."
//
// Usage:
//   example_embed_mm_audio [backend: dx|cuda|vk] [model.mmegm] [tokenizer.json]
//   example_embed_mm_audio --backend dx --audio path/to/clip.wav.mmfeat --prompt "Transcript of <|audio|>"

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
    std::filesystem::path audio_path = "examples/embedding_mm/data/roses-are.wav.mmfeat";
    luisa::string prompt;
    uint32_t target_dim = 768u;

    int positional_idx = 0;
    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if ((arg == "--backend" || arg == "-b") && i + 1 < argc) {
            backend = argv[++i];
        } else if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            egm_path = argv[++i];
        } else if ((arg == "--tokenizer" || arg == "-t") && i + 1 < argc) {
            tok_path = argv[++i];
        } else if ((arg == "--audio" || arg == "-a") && i + 1 < argc) {
            audio_path = argv[++i];
        } else if ((arg == "--prompt" || arg == "-p") && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((arg == "--dim" || arg == "-d") && i + 1 < argc) {
            target_dim = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (!arg.starts_with("-")) {
            if (positional_idx == 0) backend = argv[i];
            else if (positional_idx == 1) egm_path = argv[i];
            else if (positional_idx == 2) tok_path = argv[i];
            else if (positional_idx == 3) audio_path = argv[i];
            else if (positional_idx == 4) prompt = argv[i];
            positional_idx++;
        }
    }

    egm_path = resolve_path(egm_path);
    tok_path = resolve_path(tok_path);
    audio_path = FeatureLoader::resolve_feat_path(resolve_path(audio_path));

    std::cout << "================================================================\n"
              << " LuisaCompute EmbeddingGemma-2: Working with Audio Demo\n"
              << " Selective Encoder Loading: Text + Audio (~570M Parameters)\n"
              << " Backend: " << backend << "\n"
              << " Audio:   " << audio_path.string() << "\n"
              << "================================================================\n"
              << std::endl;

    if (!std::filesystem::exists(egm_path) || !std::filesystem::exists(tok_path)) {
        std::cerr << "Error: model or tokenizer not found.\n";
        return 1;
    }
    if (!std::filesystem::exists(audio_path)) {
        std::cerr << "Error: preprocessed audio bundle not found at '" << audio_path.string() << "'.\n"
                  << "Run: python examples/embedding_mm/tools/preprocess.py\n";
        return 1;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);

    SessionConfig config{};
    config.enable_vision = false;// Skip Vision Tower (~170M params)
    config.enable_audio = true;  // Load Audio Conformer Tower (~300M params)

    auto session = Session::create(device, egm_path, tok_path, config);
    if (!session->has_audio()) {
        std::cerr << "Error: loaded package does not contain Audio Tower weights.\n";
        return 1;
    }

    EncodeOptions opt{};
    opt.target_dimension = target_dim;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto audio_emb = session->encode_audio(audio_path, opt);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("[1] Audio Encoding (roses-are.wav -> %uD in %.2f ms)\n", audio_emb.dimension, ms);
    std::printf("  First 6 values: [%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, ...]\n\n",
                audio_emb.values[0], audio_emb.values[1], audio_emb.values[2],
                audio_emb.values[3], audio_emb.values[4], audio_emb.values[5]);

    // Official Tutorial Candidate Texts
    const luisa::string_view candidates[] = {
        "a person speaking English",
        "random static white noise",
        "Roses are red, violets are blue."
    };
    auto cand_embs = session->encode_batch(luisa::span{candidates, 3u}, opt);

    std::cout << "[2] Audio-to-Text Cross-Modal Retrieval\n";
    float best_sim = -1.0f;
    size_t best_idx = 0;
    for (size_t i = 0; i < 3; i++) {
        float sim = Session::similarity(audio_emb, cand_embs[i]);
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = i;
        }
    }
    for (size_t i = 0; i < 3; i++) {
        float sim = Session::similarity(audio_emb, cand_embs[i]);
        std::printf("  Candidate [%zu] \"%-33s\" : %.4f%s\n",
                    i + 1, candidates[i].data(), sim,
                    (i == best_idx ? "  <-- Top-1 Match" : ""));
    }
    std::cout << std::endl;

    if (!prompt.empty()) {
        std::cout << "[3] Interleaved Audio + Text Prompt: \"" << prompt << "\"\n";
        auto mm_emb = session->encode_multimodal_audio(audio_path, prompt, opt);
        std::printf("  Cosine Similarity(Audio, Audio+Prompt): %.4f\n\n",
                    Session::similarity(audio_emb, mm_emb));
    }

    std::cout << "Audio demo completed successfully." << std::endl;
    return 0;
}
