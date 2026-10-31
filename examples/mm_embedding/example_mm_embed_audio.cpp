// mm_embedding engine -- interactive multimodal audio & text embedding demo (Phase 3).
//
// Usage:
//   example_mm_embed_audio [backend: dx|cuda|vk] [model.mmegm] [tokenizer.json] [audio.wav] [optional_prompt]
// Or with flags:
//   example_mm_embed_audio [backend] --model <model.mmegm> --tokenizer <tokenizer.json> --audio <audio.wav>
//                          [--prompt <prompt>] [--dim <768|512|256|128>] [--candidates <c1,c2,...>]

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>

#include "mm_embedding/embed/session.h"
#include "mm_embedding/embed/audio_processor.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::mm_embed;

static luisa::vector<luisa::string> split_candidates(const luisa::string &input) {
    luisa::vector<luisa::string> result;
    std::string_view sv{input.data(), input.size()};
    luisa::string current;
    bool in_quote = false;

    for (size_t i = 0; i < sv.size(); i++) {
        char c = sv[i];
        if (c == '"' || c == '\'') {
            in_quote = !in_quote;
        } else if (!in_quote && (c == ';' || c == '|' || c == '\n' || c == ',')) {
            while (!current.empty() && (current.front() == ' ' || current.front() == '\t' || current.front() == '"' || current.front() == '\'')) current.erase(0, 1);
            while (!current.empty() && (current.back() == ' ' || current.back() == '\t' || current.back() == '"' || current.back() == '\'')) current.pop_back();
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(c);
        }
    }
    while (!current.empty() && (current.front() == ' ' || current.front() == '\t' || current.front() == '"' || current.front() == '\'')) current.erase(0, 1);
    while (!current.empty() && (current.back() == ' ' || current.back() == '\t' || current.back() == '"' || current.back() == '\'')) current.pop_back();
    if (!current.empty()) {
        result.push_back(current);
    }
    return result;
}

int main(int argc, char **argv) {
    luisa::string backend = "dx";
    std::filesystem::path egm_path = "model.mmegm";
    std::filesystem::path tok_path = "tokenizer.json";
    std::filesystem::path audio_path = "examples/mm_embedding/data/roses_are_16k.wav";
    luisa::string prompt = "";
    uint32_t target_dim = 768u;
    luisa::string candidates_str = "";

    // Fallback paths if not in current working directory
    if (!std::filesystem::exists(egm_path)) {
        if (std::filesystem::exists("../../model.mmegm")) {
            egm_path = "../../model.mmegm";
        } else if (std::filesystem::exists("../model.mmegm")) {
            egm_path = "../model.mmegm";
        }
    }
    if (!std::filesystem::exists(tok_path)) {
        if (std::filesystem::exists("../../tokenizer.json")) {
            tok_path = "../../tokenizer.json";
        } else if (std::filesystem::exists("../../models/embeddinggemma-2/tokenizer.json")) {
            tok_path = "../../models/embeddinggemma-2/tokenizer.json";
        } else if (std::filesystem::exists("models/embeddinggemma-2/tokenizer.json")) {
            tok_path = "models/embeddinggemma-2/tokenizer.json";
        }
    }
    if (!std::filesystem::exists(audio_path)) {
        if (std::filesystem::exists("../../examples/mm_embedding/data/roses_are_16k.wav")) {
            audio_path = "../../examples/mm_embedding/data/roses_are_16k.wav";
        } else if (std::filesystem::exists("data/roses_are_16k.wav")) {
            audio_path = "data/roses_are_16k.wav";
        } else if (std::filesystem::exists("examples/mm_embedding/data/roses-are.wav")) {
            audio_path = "examples/mm_embedding/data/roses-are.wav";
        } else if (std::filesystem::exists("../../examples/mm_embedding/data/roses-are.wav")) {
            audio_path = "../../examples/mm_embedding/data/roses-are.wav";
        }
    }

    // Argument parsing supporting both flags and positional arguments
    int positional_idx = 0;
    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            backend = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            egm_path = argv[++i];
        } else if (arg == "--tokenizer" && i + 1 < argc) {
            tok_path = argv[++i];
        } else if (arg == "--audio" && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--dim" && i + 1 < argc) {
            target_dim = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--candidates" && i + 1 < argc) {
            candidates_str = argv[++i];
        } else if (!arg.starts_with("--")) {
            if (positional_idx == 0) backend = argv[i];
            else if (positional_idx == 1) egm_path = argv[i];
            else if (positional_idx == 2) tok_path = argv[i];
            else if (positional_idx == 3) audio_path = argv[i];
            else if (positional_idx == 4) prompt = argv[i];
            positional_idx++;
        }
    }

    std::cout << "========================================================\n"
              << " LuisaCompute EmbeddingGemma-2 Audio Demo (Phase 3)\n"
              << " Backend: " << backend << "\n"
              << " Audio:   " << audio_path.string() << "\n"
              << " Target Dim: " << target_dim << "\n"
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
    if (!std::filesystem::exists(audio_path)) {
        std::cerr << "Error: audio file not found at '" << audio_path.string() << "'\n";
        return 1;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);

    SessionConfig config{};
    config.enable_vision = false;
    config.enable_audio = true; // Selective Encoder Loading

    auto session = Session::create(device, egm_path, tok_path, config);
    if (!session->has_audio()) {
        std::cerr << "Error: loaded package does not contain Audio Tower weights.\n";
        return 1;
    }

    std::cout << "\nEncoding audio '" << audio_path.filename().string() << "'..." << std::endl;
    EncodeOptions opt{};
    opt.target_dimension = target_dim;
    auto audio_emb = session->encode_audio(audio_path, opt);

    float norm = 0.0f;
    for (float v : audio_emb.values) norm += v * v;
    norm = std::sqrt(norm);

    std::cout << "  Audio Embedding Dimension: " << audio_emb.dimension << "\n"
              << "  L2 Norm:                   " << norm << "\n"
              << "  First 8 values:            [";
    for (size_t i = 0; i < 8 && i < audio_emb.values.size(); i++) {
        std::cout << (i ? ", " : "") << std::fixed << std::setprecision(5) << audio_emb.values[i];
    }
    std::cout << ", ...]\n" << std::endl;

    // Zero-shot Audio-to-Text retrieval ranking
    luisa::vector<luisa::string> candidate_list;
    if (!candidates_str.empty()) {
        candidate_list = split_candidates(candidates_str);
    } else {
        // Default official sample candidate set
        candidate_list.emplace_back("a person speaking English");
        candidate_list.emplace_back("random static white noise");
        candidate_list.emplace_back("Roses are red, violets are blue.");
    }

    if (!candidate_list.empty()) {
        std::cout << "Evaluating Candidate Texts (" << candidate_list.size() << " candidates):" << std::endl;
        struct CandidateResult {
            luisa::string text;
            float similarity{0.0f};
        };
        luisa::vector<CandidateResult> results;
        results.reserve(candidate_list.size());

        for (const auto &cand : candidate_list) {
            auto cand_emb = session->encode(cand, opt);
            float sim = 0.0f;
            for (uint32_t d = 0; d < target_dim && d < audio_emb.values.size(); d++) {
                sim += audio_emb.values[d] * cand_emb.values[d];
            }
            results.push_back({cand, sim});
        }

        // Sort descending by similarity
        std::sort(results.begin(), results.end(), [](const CandidateResult &a, const CandidateResult &b) {
            return a.similarity > b.similarity;
        });

        for (size_t r = 0; r < results.size(); r++) {
            std::cout << "  [" << (r + 1) << "] Sim: " << std::fixed << std::setprecision(4)
                      << results[r].similarity << "  -> \"" << results[r].text << "\"" << std::endl;
        }
        std::cout << "\n  Top-1 Match: \"" << results.front().text << "\" (Sim: "
                  << std::fixed << std::setprecision(4) << results.front().similarity << ")\n" << std::endl;
    }

    // Multimodal prompt encoding if provided
    if (!prompt.empty()) {
        std::cout << "Encoding multimodal prompt: \"" << prompt << "\"..." << std::endl;
        auto mm_emb = session->encode_multimodal_audio(audio_path, prompt, opt);

        float mm_norm = 0.0f;
        for (float v : mm_emb.values) mm_norm += v * v;
        mm_norm = std::sqrt(mm_norm);

        float sim_with_audio = 0.0f;
        for (uint32_t d = 0; d < target_dim; d++) {
            sim_with_audio += audio_emb.values[d] * mm_emb.values[d];
        }

        std::cout << "  Multimodal Embedding Norm: " << mm_norm << "\n"
                  << "  Cosine Sim(Audio, Multimodal): " << std::fixed << std::setprecision(4)
                  << sim_with_audio << std::endl;
    }

    std::cout << "\nAudio demo completed successfully." << std::endl;
    return 0;
}
