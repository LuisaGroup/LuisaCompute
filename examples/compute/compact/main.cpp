// GDeflate GPU compression/decompression round-trip test.
//
// Generates a random byte buffer, compresses it on the GPU using a
// GDeflate-compatible tile stream, then decompresses it on the GPU and
// verifies the result matches the original data exactly.

#include "gdeflate.h"

#include <luisa/luisa-compute.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>

#include <cstdlib>
#include <random>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::example::gdeflate;

int main(int argc, char *argv[]) {
    auto executable = argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : "";
    if (argc <= 1 || argv == nullptr || argv[1] == nullptr || argv[1][0] == '\0') {
        LUISA_INFO("Usage: {} <backend> [size]", executable);
        return 1;
    }

    uint32_t size = 1024u * 1024u; // 1 MiB default
    if (argc > 2 && argv[2] != nullptr) {
        size = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
        if (size == 0u) { size = 1024u * 1024u; }
    }

    Context ctx{argv[0]};
    Device device = ctx.create_device(argv[1]);
    Stream stream = device.create_stream();

    // Generate random input data.
    std::mt19937 rng{12345u};
    std::uniform_int_distribution<int> dist{0, 255};
    std::vector<std::byte> host_input(size);
    for (auto &b : host_input) {
        b = static_cast<std::byte>(dist(rng));
    }

    ByteBuffer input = device.create_byte_buffer(size);
    stream << input.copy_from(host_input.data()) << synchronize();

    GDeflateCodec codec{device, stream};

    ByteBuffer compressed = codec.allocate_compressed(size);
    ByteBuffer decompressed = codec.allocate_uncompressed(size);

    Clock clock;
    uint32_t compressed_size = codec.compress(input, compressed, size);
    auto compress_ms = clock.toc();

    codec.decompress(compressed, decompressed, size);
    auto decompress_ms = clock.toc();

    std::vector<std::byte> host_output(size);
    stream << decompressed.copy_to(host_output.data()) << synchronize();

    bool ok = true;
    for (uint32_t i = 0; i < size; ++i) {
        if (host_input[i] != host_output[i]) {
            LUISA_ERROR("Mismatch at byte {}: expected {}, got {}",
                        i, static_cast<int>(host_input[i]), static_cast<int>(host_output[i]));
            ok = false;
            break;
        }
    }

    if (ok) {
        double ratio = static_cast<double>(compressed_size) / static_cast<double>(size);
        LUISA_INFO("GDeflate round-trip: PASSED");
        LUISA_INFO("  input size:      {} bytes", size);
        LUISA_INFO("  compressed size: {} bytes ({:.2f}x)", compressed_size, ratio);
        LUISA_INFO("  compress time:   {} ms", compress_ms);
        LUISA_INFO("  decompress time: {} ms", decompress_ms);
        return 0;
    }

    LUISA_WARNING("GDeflate round-trip: FAILED");
    return 1;
}
