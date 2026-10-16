// Isolated native Tile compiler. The existing PTX/OptiX NVRTC helper does not
// link these APIs and keeps its original protocol and options.
#include <nvrtc.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

[[nodiscard]] bool check(nvrtcResult result, const char *operation) noexcept {
    if (result == NVRTC_SUCCESS) { return true; }
    std::fprintf(stderr, "CUDA Tile IR %s: %s\n", operation, nvrtcGetErrorString(result));
    return false;
}

class Program {
public:
    nvrtcProgram handle{};
    Program() noexcept = default;
    Program(const Program &) = delete;
    Program &operator=(const Program &) = delete;
    ~Program() noexcept {
        if (handle != nullptr) { static_cast<void>(nvrtcDestroyProgram(&handle)); }
    }
};

int compile(const std::filesystem::path &input, const std::filesystem::path &output,
            uint32_t architecture, bool debug_info) {
    int major = 0, minor = 0;
    if (!check(nvrtcVersion(&major, &minor), "query compiler version")) { return 1; }
    if (major < 13 || (major == 13 && minor < 4)) {
        std::fprintf(stderr, "CUDA Tile IR requires NVRTC 13.4 or newer (got %d.%d)\n", major, minor);
        return 1;
    }
    std::error_code error;
    auto size = std::filesystem::file_size(input, error);
    constexpr auto maximum_size = uintmax_t{256u * 1024u * 1024u};
    if (error || size == 0u || size > maximum_size) {
        std::fputs("CUDA Tile IR input source is missing or has an invalid size\n", stderr);
        return 1;
    }
    std::string source(static_cast<size_t>(size), '\0');
    std::ifstream input_file{input, std::ios::binary};
    input_file.read(source.data(), static_cast<std::streamsize>(size));
    if (!input_file || source.find('\0') != std::string::npos) {
        std::fputs("CUDA Tile IR could not read its complete source\n", stderr);
        return 1;
    }
    Program program;
    if (!check(nvrtcCreateProgram(&program.handle, source.c_str(), "luisa_tile_kernel.cu", 0, nullptr, nullptr), "create program")) { return 1; }
    auto architecture_option = std::string{"--gpu-architecture=compute_"} + std::to_string(architecture);
    std::vector<const char *> options{
        "--std=c++20", "--enable-tile", "--tile-only", "--ftz=false",
        architecture_option.c_str(),
        "--include-path=" LUISA_CUDA_TILE_IR_INCLUDE_DIR,
        "--include-path=" LUISA_CUDA_TILE_IR_INCLUDE_DIR "/cccl"};
    if (debug_info) { options.emplace_back("-lineinfo"); }
    auto status = nvrtcCompileProgram(program.handle, static_cast<int>(options.size()), options.data());
    size_t log_size = 0u;
    if (check(nvrtcGetProgramLogSize(program.handle, &log_size), "query compile log") && log_size > 1u) {
        std::string log(log_size, '\0');
        if (check(nvrtcGetProgramLog(program.handle, log.data()), "read compile log")) {
            std::fwrite(log.data(), 1u, log_size - 1u, stderr);
            std::fputc('\n', stderr);
        }
    }
    if (!check(status, "compile")) { return 1; }
    size_t binary_size = 0u;
    if (!check(nvrtcGetTileIRSize(program.handle, &binary_size), "query Tile IR size")) { return 1; }
    if (binary_size == 0u || binary_size > maximum_size) {
        std::fputs("CUDA Tile IR compiler returned an invalid binary size\n", stderr);
        return 1;
    }
    std::vector<char> binary(binary_size);
    if (!check(nvrtcGetTileIR(program.handle, binary.data()), "read Tile IR")) { return 1; }
    // File transport is binary on Windows as well; no stdout newline expansion.
    std::ofstream output_file{output, std::ios::binary | std::ios::trunc};
    output_file.write(binary.data(), static_cast<std::streamsize>(binary.size()));
    output_file.close();
    if (!output_file) {
        std::fputs("CUDA Tile IR could not write its complete binary\n", stderr);
        return 1;
    }
    return 0;
}

template<typename Char>
int run(int argc, const Char *const *argv) {
    if (argc != 5) {
        std::fputs("Usage: luisa_cuda_tile_compiler SOURCE TILE_IR ARCHITECTURE DEBUG_INFO_0_OR_1\n", stderr);
        return 2;
    }
    uint32_t architecture = 0u;
    auto text = argv[3];
    if (*text == Char{0}) { return 2; }
    for (; *text != Char{0}; text++) {
        if (*text < Char{'0'} || *text > Char{'9'} || architecture > 999u) { return 2; }
        architecture = architecture * 10u + static_cast<uint32_t>(*text - Char{'0'});
    }
    if (architecture == 0u || (argv[4][0] != Char{'0'} && argv[4][0] != Char{'1'}) ||
        argv[4][1] != Char{0}) { return 2; }
    return compile(std::filesystem::path{argv[1]}, std::filesystem::path{argv[2]}, architecture, argv[4][0] == Char{'1'});
}

}// namespace

#ifdef _WIN32
int wmain(int argc, wchar_t *argv[]) { return run(argc, argv); }
#else
int main(int argc, char *argv[]) { return run(argc, argv); }
#endif
