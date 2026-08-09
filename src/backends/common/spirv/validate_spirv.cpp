#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include <spirv-tools/libspirv.hpp>

namespace {

[[nodiscard]] bool validate_file(const char *path) {
    std::ifstream file{path, std::ios::binary | std::ios::ate};
    if (!file) {
        std::cerr << path << ": failed to open SPIR-V module.\n";
        return false;
    }
    auto stream_size = static_cast<std::streamoff>(file.tellg());
    if (stream_size <= 0 ||
        static_cast<uintmax_t>(stream_size) >
            std::numeric_limits<size_t>::max() ||
        stream_size > std::numeric_limits<std::streamsize>::max()) {
        std::cerr << path << ": invalid SPIR-V module size.\n";
        return false;
    }
    auto byte_size = static_cast<size_t>(stream_size);
    if (byte_size % sizeof(uint32_t) != 0u) {
        std::cerr << path << ": SPIR-V size is not word-aligned.\n";
        return false;
    }
    std::vector<uint32_t> words(byte_size / sizeof(uint32_t));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char *>(words.data()),
              static_cast<std::streamsize>(byte_size));
    if (!file) {
        std::cerr << path << ": failed to read SPIR-V module.\n";
        return false;
    }

    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    tools.SetMessageConsumer(
        [path](spv_message_level_t, const char *source,
               const spv_position_t &position, const char *message) {
            std::cerr << path << ':' << position.line << ':'
                      << position.column << ": "
                      << (source == nullptr ? "" : source) << ": "
                      << (message == nullptr ? "" : message) << '\n';
        });
    if (!tools.Validate(words.data(), words.size())) {
        std::cerr << path
                  << ": SPIR-V validation failed for Vulkan 1.2.\n";
        return false;
    }
    return true;
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: luisa_validate_spirv <module.spv> [...]\n";
        return 2;
    }
    auto valid = true;
    for (auto i = 1; i < argc; ++i) {
        valid &= validate_file(argv[i]);
    }
    return valid ? 0 : 1;
}
