// embedding_mm engine -- C++ model package inspection and validation tool.
//
// Usage:
//   example_embed_mm_pack_model <model.mmegm>

#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <iostream>

#include <luisa/core/logging.h>
#include "embedding_mm/embed/package.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <model.mmegm>\n", argv[0]);
        return 1;
    }

    std::filesystem::path egm_path = argv[1];
    if (!std::filesystem::exists(egm_path)) {
        std::fprintf(stderr, "Error: package file '%s' does not exist.\n", egm_path.string().c_str());
        return 1;
    }

    std::cout << "Loading package: " << egm_path << "..." << std::endl;
    auto pkg = luisa::embed_mm::Package::load(egm_path);
    if (!pkg) {
        std::fprintf(stderr, "Error: failed to load package '%s'.\n", egm_path.string().c_str());
        return 1;
    }

    std::cout << "=== Package Metadata ===" << std::endl;
    std::cout << "  Precision:         " << luisa::embed_mm::package_precision_name(pkg->precision()) << std::endl;
    std::cout << "  Layers:            " << pkg->num_layers() << std::endl;
    std::cout << "  Hidden Size:       " << pkg->hidden_size() << std::endl;
    std::cout << "  Intermediate Size: " << pkg->intermediate_size() << std::endl;
    std::cout << "  Num Heads:         " << pkg->num_heads() << std::endl;
    std::cout << "  Num KV Heads:      " << pkg->num_kv_heads() << std::endl;
    std::cout << "  Head Dim:          " << pkg->head_dim() << std::endl;
    std::cout << "  Embedding Dim:     " << pkg->embedding_dim() << std::endl;
    std::cout << "  Vocab Size:        " << pkg->vocab_size() << std::endl;
    std::cout << "  Max Positions:     " << pkg->max_positions() << std::endl;
    std::cout << "  Total Tensors:     " << pkg->tensors().size() << std::endl;

    std::cout << "\n=== Tensor Inventory ===" << std::endl;
    size_t total_bytes = 0;
    for (size_t i = 0; i < pkg->tensors().size(); i++) {
        const auto &t = pkg->tensors()[i];
        total_bytes += t.size_bytes;
        std::cout << "  [" << i << "] rank=" << t.rank << " dims=[";
        for (uint32_t d = 0; d < t.rank; d++) {
            std::cout << (d > 0 ? "x" : "") << t.dims[d];
        }
        std::cout << "] size=" << t.size_bytes << " B" << std::endl;
    }
    std::cout << "Total payload size: " << (total_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;

    return 0;
}
