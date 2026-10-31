// embedding_mm engine -- Comprehensive Multimodal GPU Benchmark Suite.
//
// Benchmarks LuisaCompute's native GPU DSL mega-kernels across all EmbeddingGemma-2
// configurations and modalities using the official Google tutorial assets:
//   1. Selective Encoder Loading Cold-Start & Parameter Footprint (270M / 440M / 570M / 740M)
//   2. Text Latency & Batch Throughput (B=1, 4, 8, 16)
//   3. Vision Tower Latency (Default 280 budget vs High-Res 1120 budget)
//   4. Audio Conformer Tower Latency (roses-are.wav, 12-layer USM Conformer)
//   5. Video Spatio-Temporal Latency (16-frame uniform & 29-frame 1fps)
//   6. Interleaved Multimodal Latency (Text + Multi-Image + Audio / Video)
//
// Usage:
//   example_embed_mm_bench [backend: dx|cuda|vk] [model.mmegm] [tokenizer.json] [data_dir]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>

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

template<typename Fn>
static void run_bench_row(const char *label, int warmup_iters, int bench_iters, Fn &&fn) {
    for (int i = 0; i < warmup_iters; i++) {
        fn();
    }
    luisa::vector<double> times;
    times.reserve(bench_iters);
    for (int i = 0; i < bench_iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double p50 = times[times.size() / 2];
    double p95 = times[static_cast<size_t>(times.size() * 0.95)];
    std::printf("  %-46s | Mean: %7.2f ms | P50: %7.2f ms | Min: %7.2f ms | P95: %7.2f ms\n",
                label, mean, p50, times.front(), p95);
}

int main(int argc, char **argv) {
    luisa::string backend = (argc > 1) ? argv[1] : "dx";
    std::filesystem::path egm_path = (argc > 2) ? argv[2] : "model.mmegm";
    std::filesystem::path tok_path = (argc > 3) ? argv[3] : "tokenizer.json";
    std::filesystem::path data_dir = (argc > 4) ? argv[4] : "examples/embedding_mm/data";

    egm_path = resolve_path(egm_path);
    tok_path = resolve_path(tok_path);
    data_dir = resolve_path(data_dir);

    std::cout << "==============================================================================================\n"
              << "             LuisaCompute EmbeddingGemma-2 Multimodal GPU Benchmark Suite                     \n"
              << "==============================================================================================\n"
              << " Backend:   " << backend << "\n"
              << " Model:     " << egm_path.string() << "\n"
              << " Data Dir:  " << data_dir.string() << "\n"
              << "==============================================================================================\n"
              << std::endl;

    if (!std::filesystem::exists(egm_path) || !std::filesystem::exists(tok_path)) {
        std::cerr << "Error: model or tokenizer not found.\n";
        return 1;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);

    // 1. Selective Encoder Loading Cold-Start Benchmark
    std::cout << "[1/3] Selective Encoder Loading Cold-Start Time & Parameter Footprint\n";
    struct ModeSpec {
        const char *name;
        const char *params;
        bool vision;
        bool audio;
    };
    const ModeSpec modes[] = {
        {"Text-Only (Language Model)", "~270M", false, false},
        {"Text + Vision (Image & Video)", "~440M", true, false},
        {"Text + Audio (USM Conformer)", "~570M", false, true},
        {"Full Multimodal (All Encoders)", "~740M", true, true},
    };

    luisa::unique_ptr<Session> session;
    for (const auto &m : modes) {
        SessionConfig cfg{m.vision, m.audio};
        auto t0 = std::chrono::high_resolution_clock::now();
        auto s = Session::create(device, egm_path, tok_path, cfg);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  %-36s (%s) | Init & GPU Upload: %7.1f ms\n", m.name, m.params, ms);
        if (m.vision && m.audio) {
            session = std::move(s);
        }
    }
    std::cout << std::endl;

    // 2. Text Latency & Batch Throughput
    std::cout << "[2/3] Text Embedding Latency & Batch Throughput\n";
    const luisa::string_view sample_doc =
        "The northern lights are caused by charged particles from the sun interacting with Earth's magnetic field.";
    run_bench_row("Text Query (B=1, SearchQuery)", 2, 10, [&]() {
        (void)session->encode_query("What causes the northern lights?");
    });
    for (uint32_t B : {1u, 4u, 8u, 16u}) {
        luisa::vector<luisa::string_view> batch(B, sample_doc);
        char label[64];
        std::snprintf(label, sizeof(label), "Text Document Batch (B=%u)", B);
        run_bench_row(label, 2, 8, [&]() {
            (void)session->encode_documents(batch);
        });
    }
    std::cout << std::endl;

    // 3. Multimodal GPU Mega-Kernel Latency
    std::cout << "[3/3] Multimodal GPU Mega-Kernel Latency (Pre-decoded .mmfeat Tensors)\n";
    auto img_280_path = data_dir / "GoldenGate.png.mmfeat";
    auto img_1120_path = data_dir / "GoldenGate_1120.png.mmfeat";
    auto aud_path = data_dir / "roses-are.wav.mmfeat";
    auto vid_16_path = data_dir / "cat_dozing_16frames.mp4.mmfeat";
    auto vid_def_path = data_dir / "cat_dozing.mp4.mmfeat";
    auto inter_mm_path = data_dir / "interleaved_multimodal.mmfeat";
    auto inter_seq_path = data_dir / "interleaved_media_seq.mmfeat";

    if (std::filesystem::exists(img_280_path)) {
        auto img_280 = FeatureLoader::load_image(img_280_path);
        run_bench_row("Image Default (GoldenGate, 256 soft tokens)", 2, 8, [&]() {
            (void)session->encode_image(img_280);
        });
    }
    if (std::filesystem::exists(img_1120_path)) {
        auto img_1120 = FeatureLoader::load_image(img_1120_path);
        run_bench_row("Image High-Res (GoldenGate, 1089 soft tokens)", 1, 5, [&]() {
            (void)session->encode_image(img_1120);
        });
    }
    if (std::filesystem::exists(aud_path)) {
        auto aud = FeatureLoader::load_audio(aud_path);
        run_bench_row("Audio Conformer (roses-are.wav, 77 tokens)", 2, 10, [&]() {
            (void)session->encode_audio(aud);
        });
    }
    if (std::filesystem::exists(vid_16_path)) {
        auto vid16 = FeatureLoader::load_video(vid_16_path);
        run_bench_row("Video 16-Frame Timestamped (cat_dozing.mp4)", 1, 3, [&]() {
            (void)session->encode_video(vid16);
        });
    }
    if (std::filesystem::exists(vid_def_path)) {
        auto vid29 = FeatureLoader::load_video(vid_def_path);
        run_bench_row("Video Default 1fps 29-Frame (cat_dozing.mp4)", 1, 3, [&]() {
            (void)session->encode_video(vid29);
        });
    }
    if (std::filesystem::exists(inter_mm_path)) {
        auto bundle = FeatureLoader::load_multimodal(inter_mm_path);
        run_bench_row("Interleaved (Text + 2 Images + Audio)", 1, 5, [&]() {
            (void)session->encode_multimodal(bundle);
        });
    }
    if (std::filesystem::exists(inter_seq_path)) {
        auto bundle = FeatureLoader::load_multimodal(inter_seq_path);
        run_bench_row("Interleaved Sequence (<|image|><|audio|><|image|>)", 1, 5, [&]() {
            (void)session->encode_multimodal(bundle);
        });
    }

    std::cout << "\nBenchmark suite completed successfully." << std::endl;
    return 0;
}
