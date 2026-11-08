/**
 * LuisaCompute NTC CLI Tool
 * Command-line interface Compatible with official NVIDIA ntc-cli.exe.
 */

#include "ntc_common.h"
#include "ntex_container.h"
#include "infer_engine.h"
#include "bcn_encoder.h"
#include "trainer.h"
#include "manifest.h"

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>

#include <stb/stb_image_write.h>

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <string>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ntc;

void print_help() {
    std::cout << R"(
LuisaCompute NTC (Neural Texture Compression) CLI
Usage: luisa_ntc_cli [options...]

Actions:
  -c, --compress                        Perform NTC compression from manifest
  -D, --decompress                      Perform NTC decompression
  -d, --describe                        Describe contents of compressed NTC package
  -h, --help                            Show this help message

File I/O:
  --loadManifest=<str>                  Load JSON manifest file
  --loadCompressed=<str>                Load .ntc container file
  -o, --saveCompressed=<str>            Save compressed .ntc file
  -i, --saveImages=<str>                Output directory for decompressed images
  -B, --bcFormat=<str>                  BCn format: BC1, BC3, BC4, BC5, BC7, none (default: BCn from manifest)
  --backend=<str>                       Backend to use: cuda (default), dx, vk

Training Options:
  -S, --trainingSteps=<int>             Total training step count (default: 10000)
  --networkLearningRate=<flt>           Network learning rate (default: 0.005)
  --gridLearningRate=<flt>              Latent grid learning rate (default: 0.1)
  --kPixelsPerBatch=<int>               Kilopixels per batch (default: 64)
  --gridSizeScale=<int>                 Ratio of image size to feature grid (default: 4)
  --numFeatures=<int>                   Number of features (default: 16)
)" << std::endl;
}

int main(int argc, char *argv[]) {
    luisa::log_level_info();

    bool do_compress = false;
    bool do_decompress = false;
    bool do_describe = false;

    std::string load_manifest;
    std::string load_compressed;
    std::string save_compressed;
    std::string save_images;
    std::string bc_override;
    std::string backend_name = "cuda";

    TrainingSettings train_settings{};

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-c" || arg == "--compress") {
            do_compress = true;
        } else if (arg == "-D" || arg == "--decompress") {
            do_decompress = true;
        } else if (arg == "-d" || arg == "--describe") {
            do_describe = true;
        } else if (arg == "-h" || arg == "--help") {
            print_help();
            return 0;
        } else if (arg.starts_with("--loadManifest=")) {
            load_manifest = arg.substr(15);
        } else if (arg.starts_with("--loadCompressed=")) {
            load_compressed = arg.substr(17);
        } else if (arg.starts_with("-o=") || arg.starts_with("--saveCompressed=")) {
            save_compressed = arg.substr(arg.find('=') + 1);
        } else if (arg == "-o" && i + 1 < argc) {
            save_compressed = argv[++i];
        } else if (arg.starts_with("-i=") || arg.starts_with("--saveImages=")) {
            save_images = arg.substr(arg.find('=') + 1);
        } else if (arg == "-i" && i + 1 < argc) {
            save_images = argv[++i];
        } else if (arg.starts_with("-B=") || arg.starts_with("--bcFormat=")) {
            bc_override = arg.substr(arg.find('=') + 1);
        } else if (arg == "-B" && i + 1 < argc) {
            bc_override = argv[++i];
        } else if (arg.starts_with("-S=") || arg.starts_with("--trainingSteps=")) {
            train_settings.trainingSteps = std::stoi(std::string(arg.substr(arg.find('=') + 1)));
        } else if (arg == "-S" && i + 1 < argc) {
            train_settings.trainingSteps = std::stoi(argv[++i]);
        } else if (arg.starts_with("--backend=")) {
            backend_name = arg.substr(10);
        } else if (arg.ends_with(".ntc")) {
            load_compressed = arg;
        } else if (arg.ends_with(".json")) {
            load_manifest = arg;
        }
    }

    if (!do_compress && !do_decompress && !do_describe) {
        if (!load_compressed.empty()) {
            do_describe = true;
        } else if (!load_manifest.empty()) {
            do_compress = true;
        } else {
            print_help();
            return 1;
        }
    }

    // 1. Describe action
    if (do_describe) {
        if (load_compressed.empty()) {
            LUISA_ERROR_WITH_LOCATION("No .ntc file specified for describe.");
            return 1;
        }
        NtexContainer container;
        std::string err;
        if (!container.load_from_file(load_compressed, err)) {
            LUISA_ERROR_WITH_LOCATION("Failed to load .ntc file: {}", err);
            return 1;
        }

        std::cout << "Dimensions: " << container.width << "x" << container.height
                  << ", " << container.numChannels << " channels, "
                  << container.numColorMips << " mip level(s)\n";

        float bpp = FeatureGridMath::calculate_latent_bits_per_pixel(container.latentShape.gridSizeScale, container.latentShape.numFeatures);
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Base compression rate: --bitsPerPixel " << bpp << "\n";
        std::cout << "Latent shape: --gridSizeScale " << container.latentShape.gridSizeScale
                  << " --numFeatures " << container.latentShape.numFeatures << "\n";

        std::cout << "Inference weights: Int8 [" << (container.has_weight_type(InferenceWeightType::GenericInt8) ? "Y" : "N")
                  << "], FP8 [" << (container.has_weight_type(InferenceWeightType::GenericFP8) ? "Y" : "N") << "]\n";
        std::cout << "Latent compression: None\n";

        std::cout << "Textures:\n";
        for (size_t idx = 0; idx < container.textures.size(); ++idx) {
            const auto &tex = container.textures[idx];
            std::cout << idx << ": " << tex.name << "\n";
            std::cout << "   Channels: " << tex.firstChannel << "-" << (tex.firstChannel + tex.numChannels - 1) << "\n";
            std::cout << "   Channel format: " << (tex.channelFormat ? channel_format_to_string(*tex.channelFormat) : "UNORM8") << "\n";
            std::cout << "   BCn format: " << (tex.bcFormat ? bc_format_to_string(*tex.bcFormat) : "None") << "\n";
            std::cout << "   RGB color space: " << ((tex.rgbColorSpace && *tex.rgbColorSpace == ColorSpace::sRGB) ? "sRGB" : "Linear") << "\n";
        }
        return 0;
    }

    // Initialize LuisaCompute Context and Device
    Context context{argv[0]};
    Device device = context.create_device(backend_name);
    Stream stream = device.create_stream();

    // 2. Compress action
    if (do_compress) {
        if (load_manifest.empty()) {
            LUISA_ERROR_WITH_LOCATION("No manifest file specified for compression.");
            return 1;
        }

        MaterialManifest manifest;
        if (!manifest.load(load_manifest, true)) {
            LUISA_ERROR_WITH_LOCATION("Failed to load material manifest.");
            return 1;
        }

        Trainer trainer(device, stream, train_settings);
        NtexContainer container;

        if (!trainer.train(manifest, container)) {
            LUISA_ERROR_WITH_LOCATION("Training failed.");
            return 1;
        }

        if (save_compressed.empty()) {
            save_compressed = (std::filesystem::path(load_manifest).parent_path() / "output.ntc").string();
        }

        std::string err;
        if (!container.save_to_file(save_compressed, err)) {
            LUISA_ERROR_WITH_LOCATION("Failed to save .ntc container: {}", err);
            return 1;
        }

        LUISA_INFO("Saved compressed NTC package: {} ({} bytes)",
                   save_compressed, container.binaryChunk.size());
    }

    // 3. Decompress action
    if (do_decompress) {
        if (load_compressed.empty()) {
            LUISA_ERROR_WITH_LOCATION("No .ntc file specified for decompression.");
            return 1;
        }

        NtexContainer container;
        std::string err;
        if (!container.load_from_file(load_compressed, err)) {
            LUISA_ERROR_WITH_LOCATION("Failed to load .ntc file: {}", err);
            return 1;
        }

        if (save_images.empty()) {
            save_images = (std::filesystem::path(load_compressed).parent_path() / "decompressed").string();
        }
        std::filesystem::create_directories(save_images);

        InferEngine infer(device, stream);
        if (!infer.load_container(container)) {
            LUISA_ERROR_WITH_LOCATION("Failed to load container into InferEngine.");
            return 1;
        }

        bool save_raw = (bc_override == "none");
        BCnEncoder bcn(device, stream);

        auto get_clean_name = [](std::string_view name) -> std::string {
            size_t pos = name.find('|');
            return std::string((pos == std::string_view::npos) ? name : name.substr(0, pos));
        };

        for (uint32_t mip = 0; mip < 1; ++mip) { // decompress base mip
            uint32_t mw = std::max(1u, container.width >> mip);
            uint32_t mh = std::max(1u, container.height >> mip);

            if (save_raw) {
                auto decomp_textures = infer.decompress_all_textures(container, mip);
                for (const auto &tex : decomp_textures) {
                    std::string clean = get_clean_name(tex.name);
                    auto out_file = std::filesystem::path(save_images) / (clean + ".png");
                    stbi_write_png(out_file.string().c_str(), mw, mh, tex.channels, tex.pixelBytes.data(), mw * tex.channels);
                    LUISA_INFO("Saved image '{}': {}x{} pixels, {} channels, UNORM8.",
                               out_file.string(), mw, mh, tex.channels);
                }
            } else {
                Buffer<float> gpu_pixels = infer.decompress_mip_device(mip, mw, mh);
                for (const auto &tex : container.textures) {
                    BlockCompressedFormat bcf = tex.bcFormat.value_or(BlockCompressedFormat::BC7);
                    if (bcf == BlockCompressedFormat::None) bcf = BlockCompressedFormat::BC7;

                    std::string clean = get_clean_name(tex.name);
                    auto out_file = std::filesystem::path(save_images) / (clean + ".dds");
                    bcn.encode_and_save_dds(gpu_pixels, mw, mh, tex.firstChannel, tex.numChannels, bcf, out_file);
                    LUISA_INFO("Saved image '{}': {}x{} pixels, 1 mips, BCn.", out_file.string(), mw, mh);
                }
            }
        }
    }

    return 0;
}
