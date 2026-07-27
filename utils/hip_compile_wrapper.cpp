//
// Offline compiler for HIPRT wrapper functions.
// Uses hiprtc to compile a .hip source file to LLVM bitcode and writes
// the result to a file. This is invoked at build time by CMake so the
// bitcode can be embedded into the backend library with luisa_embed_device_lib.
//

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hiprtc.h>

[[noreturn]] static void print_usage_and_exit(const char *prog, int code) noexcept {
    std::cerr << "Usage: " << prog
              << " -i <input.hip> -o <output.bc> [-a <arch>] [-I <include-dir>] [-D <definition>]...\n"
              << "Options:\n"
              << "  -i, --input  <file>   Input .hip source file\n"
              << "  -o, --output <file>   Output LLVM bitcode file\n"
              << "  -a, --arch   <arch>   Target GPU architecture (e.g., gfx1201)\n"
              << "  -I <dir>              Additional include directory (repeatable)\n"
              << "  -D, --define <def>    Preprocessor definition (repeatable)\n"
              << "      --help            Show this help\n";
    std::exit(code);
}

int main(int argc, char *argv[]) {
    std::filesystem::path input_file;
    std::filesystem::path output_file;
    std::vector<std::string> archs;
    std::vector<std::string> include_dirs;
    std::vector<std::string> definitions;

    for (int i = 1; i < argc; i++) {
        auto opt = std::string_view{argv[i]};
        auto require_next = [&]() -> std::string_view {
            if (i + 1 >= argc) {
                std::cerr << "Error: option " << opt << " requires an argument.\n";
                print_usage_and_exit(argv[0], 1);
            }
            return std::string_view{argv[++i]};
        };
        if (opt == "-i" || opt == "--input") {
            input_file = require_next();
        } else if (opt == "-o" || opt == "--output") {
            output_file = require_next();
        } else if (opt == "-a" || opt == "--arch") {
            archs.emplace_back(require_next());
        } else if (opt == "-I") {
            include_dirs.emplace_back(require_next());
        } else if (opt == "-D" || opt == "--define") {
            definitions.emplace_back(require_next());
        } else if (opt == "--help") {
            print_usage_and_exit(argv[0], 0);
        } else {
            std::cerr << "Error: unknown option: " << opt << "\n";
            print_usage_and_exit(argv[0], 1);
        }
    }

    if (input_file.empty() || output_file.empty()) {
        std::cerr << "Error: both --input and --output are required.\n";
        print_usage_and_exit(argv[0], 1);
    }

    std::ifstream ifs{input_file, std::ios::binary | std::ios::ate};
    if (!ifs) {
        std::cerr << "Error: failed to open input file: " << input_file << "\n";
        return 1;
    }
    auto file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::string source(file_size, '\0');
    if (!ifs.read(source.data(), file_size)) {
        std::cerr << "Error: failed to read input file: " << input_file << "\n";
        return 1;
    }
    ifs.close();

    hiprtcProgram prog{};
    auto file_name = input_file.filename().string();
    auto result = hiprtcCreateProgram(&prog, source.c_str(), file_name.c_str(),
                                      0, nullptr, nullptr);
    if (result != HIPRTC_SUCCESS) {
        std::cerr << "Error: hiprtcCreateProgram failed: "
                  << hiprtcGetErrorString(result) << "\n";
        return 1;
    }

    std::vector<std::string> option_strings;
    option_strings.emplace_back("-fgpu-rdc");
    option_strings.emplace_back("-Xclang");
    option_strings.emplace_back("-disable-llvm-passes");
    option_strings.emplace_back("-Xclang");
    option_strings.emplace_back("-mno-constructor-aliases");
    option_strings.emplace_back("-std=c++17");
    for (auto &a : archs) {
        option_strings.emplace_back("--offload-arch=" + a);
    }
    for (auto &dir : include_dirs) {
        option_strings.emplace_back("-I" + dir);
    }
    for (auto &definition : definitions) {
        option_strings.emplace_back("-D" + definition);
    }

    std::vector<const char *> options;
    options.reserve(option_strings.size());
    for (auto &s : option_strings) {
        options.push_back(s.c_str());
    }

    result = hiprtcCompileProgram(prog, static_cast<int>(options.size()), options.data());
    if (result != HIPRTC_SUCCESS) {
        size_t log_size = 0;
        hiprtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        hiprtcGetProgramLog(prog, log.data());
        hiprtcDestroyProgram(&prog);
        std::cerr << "Error: hiprtcCompileProgram failed: "
                  << hiprtcGetErrorString(result) << "\nLog:\n"
                  << log << "\n";
        return 1;
    }

    size_t bitcode_size = 0;
    result = hiprtcGetBitcodeSize(prog, &bitcode_size);
    if (result != HIPRTC_SUCCESS) {
        hiprtcDestroyProgram(&prog);
        std::cerr << "Error: hiprtcGetBitcodeSize failed: "
                  << hiprtcGetErrorString(result) << "\n";
        return 1;
    }

    std::vector<char> bitcode(bitcode_size);
    result = hiprtcGetBitcode(prog, bitcode.data());
    hiprtcDestroyProgram(&prog);
    if (result != HIPRTC_SUCCESS) {
        std::cerr << "Error: hiprtcGetBitcode failed: "
                  << hiprtcGetErrorString(result) << "\n";
        return 1;
    }

    std::error_code ec;
    std::filesystem::create_directories(output_file.parent_path(), ec);
    std::ofstream ofs{output_file, std::ios::binary | std::ios::trunc};
    if (!ofs) {
        std::cerr << "Error: failed to open output file: " << output_file << "\n";
        return 1;
    }
    ofs.write(bitcode.data(), static_cast<std::streamsize>(bitcode.size()));
    if (!ofs) {
        std::cerr << "Error: failed to write output file: " << output_file << "\n";
        return 1;
    }

    std::cout << "Compiled " << input_file.filename().string()
              << " to " << bitcode_size << " bytes of LLVM bitcode.\n";
    return 0;
}
