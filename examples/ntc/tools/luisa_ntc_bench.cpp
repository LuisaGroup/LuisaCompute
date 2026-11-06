/**
 * LuisaCompute NTC Benchmark Tool
 * Measures decompression throughput (Mtexels/s), latency, and BCn transcoding performance.
 */

#include "ntc_common.h"
#include "ntex_container.h"
#include "infer_engine.h"
#include "bcn_encoder.h"

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <string>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ntc;

int main(int argc, char *argv[]) {
    luisa::log_level_info();

    std::string ntc_path;
    std::string backend_name = "cuda";
    uint32_t iterations = 50;

    if (const char *env_ntc = std::getenv("NTC_SAMPLE_NTC")) {
        ntc_path = env_ntc;
    } else if (std::filesystem::exists("C:/Users/tomo/Desktop/CVPR2027/demos/output_cli/Tiles135D_py.ntc")) {
        ntc_path = "C:/Users/tomo/Desktop/CVPR2027/demos/output_cli/Tiles135D_py.ntc";
    }

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: luisa_ntc_bench --ntc=<path.ntc> [--backend=cuda|dx|vk] [--iters=50]\n";
            return 0;
        } else if (arg.starts_with("--ntc=")) {
            ntc_path = arg.substr(6);
        } else if (arg.starts_with("--backend=")) {
            backend_name = arg.substr(10);
        } else if (arg.starts_with("--iters=")) {
            iterations = std::stoi(std::string(arg.substr(8)));
        } else if (arg.ends_with(".ntc")) {
            ntc_path = arg;
        }
    }

    if (ntc_path.empty()) {
        std::cout << "Usage: luisa_ntc_bench --ntc=<path.ntc> [--backend=cuda|dx|vk] [--iters=50]\n";
        return 1;
    }

    if (!std::filesystem::exists(ntc_path)) {
        LUISA_ERROR_WITH_LOCATION("File not found: {}", ntc_path);
        return 1;
    }

    NtexContainer container;
    std::string err;
    if (!container.load_from_file(ntc_path, err)) {
        LUISA_ERROR_WITH_LOCATION("Failed to load .ntc file: {}", err);
        return 1;
    }

    Context context{argv[0]};
    Device device = context.create_device(backend_name);
    Stream stream = device.create_stream();

    InferEngine infer(device, stream);
    if (!infer.load_container(container)) {
        LUISA_ERROR_WITH_LOCATION("Failed to load container into InferEngine.");
        return 1;
    }

    std::cout << "\n=======================================================\n";
    std::cout << "  LuisaCompute NTC Inference & Transcoding Benchmark\n";
    std::cout << "  Target: " << ntc_path << "\n";
    std::cout << "  Resolution: " << container.width << "x" << container.height << " (" << container.numChannels << " channels)\n";
    std::cout << "  Backend: " << backend_name << " | Iterations: " << iterations << "\n";
    std::cout << "=======================================================\n\n";

    // Benchmark Decompression Throughput
    std::cout << "--- 1. On-Load Decompression Throughput ---\n";
    std::cout << std::left << std::setw(12) << "Mip Level"
              << std::setw(16) << "Resolution"
              << std::setw(18) << "Throughput (MT/s)"
              << std::setw(16) << "Latency (ms)" << "\n";
    std::cout << "-------------------------------------------------------\n";

    for (uint32_t mip = 0; mip < std::min(4u, container.numColorMips); ++mip) {
        uint32_t mw = std::max(1u, container.width >> mip);
        uint32_t mh = std::max(1u, container.height >> mip);

        double mtexels = infer.benchmark_throughput(mip, mw, mh, iterations);
        double total_texels = static_cast<double>(mw) * mh;
        double latency_ms = (total_texels / (mtexels * 1e6)) * 1000.0;

        std::string res_str = std::to_string(mw) + "x" + std::to_string(mh);
        std::cout << std::left << std::setw(12) << mip
                  << std::setw(16) << res_str
                  << std::fixed << std::setprecision(2) << std::setw(18) << mtexels
                  << std::setw(16) << latency_ms << "\n";
    }

    // Benchmark Pure GPU BCn Transcoding
    std::cout << "\n--- 2. Pure GPU BCn Transcoding Performance ---\n";
    BCnEncoder bcn(device, stream);
    uint32_t w = container.width;
    uint32_t h = container.height;
    Buffer<float> gpu_pixels = infer.decompress_mip_device(0, w, h);

    auto bench_bcn = [&](const std::string &name, BlockCompressedFormat fmt) {
        // warm up
        bcn.encode_to_gpu_buffer(gpu_pixels, w, h, 0, 3, fmt);
        stream << synchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < iterations; ++i) {
            bcn.encode_to_gpu_buffer(gpu_pixels, w, h, 0, 3, fmt);
        }
        stream << synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = (std::chrono::duration<double>(t1 - t0).count() / iterations) * 1000.0;
        double mtex_s = (static_cast<double>(w * h) / (ms * 1e-3)) * 1e-6;

        std::cout << std::left << std::setw(10) << name
                  << "Latency: " << std::fixed << std::setprecision(3) << ms << " ms/texture ("
                  << std::setprecision(1) << mtex_s << " Mtexels/s)\n";
    };

    bench_bcn("BC1 (RGB)", BlockCompressedFormat::BC1);
    bench_bcn("BC3 (RGBA)", BlockCompressedFormat::BC3);
    bench_bcn("BC4 (R)", BlockCompressedFormat::BC4);
    bench_bcn("BC5 (RG)", BlockCompressedFormat::BC5);
    bench_bcn("BC7 (RGBA)", BlockCompressedFormat::BC7);

    std::cout << "\nBenchmark completed successfully!\n";
    return 0;
}
