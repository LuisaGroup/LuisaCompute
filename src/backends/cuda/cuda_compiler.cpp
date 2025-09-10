#include <luisa/core/clock.h>
#include <luisa/core/binary_io.h>
#include "../common/subprocess.h"
#include "cuda_error.h"
#include "cuda_device.h"
#include "optix_api.h"
#include "cuda_builtin_embedded.h"
#include "cuda_compiler.h"

namespace luisa::compute::cuda {

[[nodiscard]] inline auto read_from_subprocess(reproc::process &p, size_t chunk_size = 4_k) noexcept {
    luisa::vector<std::byte> buffer;
    for (;;) {
        auto current_size = buffer.size();
        buffer.resize(luisa::next_pow2(buffer.size() + chunk_size));
        auto max_read = buffer.size() - current_size;
        auto [read_size, error] = p.read(
            reproc::stream::out, reinterpret_cast<uint8_t *>(buffer.data() + current_size), max_read);
        if (error) {
            buffer.resize(current_size);
            break;
        }
        buffer.resize(current_size + read_size);
    }
    return buffer;
}

[[nodiscard]] inline auto compile_with_standalone_compiler(
    const char *exe_path,
    const luisa::string &src, const luisa::string &src_filename,
    luisa::span<const char *const> options) {

    // prepare the command line
    luisa::vector<const char *> argv;
    argv.reserve(options.size() + 2u);
    argv.emplace_back(exe_path);
    for (auto o : options) { argv.emplace_back(o); }
    argv.emplace_back(nullptr);

    auto temp_file = tmpfile();
    LUISA_ASSERT(temp_file != nullptr,
                 "Failed to create temp file for CUDA compiler.");

    // setup the options
    reproc::options o;
    o.redirect.in.type = reproc::redirect::pipe;
    o.redirect.err.type = reproc::redirect::parent;
    o.redirect.out.type = reproc::redirect::file_;
    o.redirect.out.file = temp_file;

    reproc::process p;
    if (auto error = p.start(reproc::arguments{argv.data()}, o)) {
        LUISA_ERROR_WITH_LOCATION("Failed to start the process: {}.", error.message());
    }

    auto write = [&p](const luisa::string &s) noexcept {
        auto write_data = [&p](const void *data, size_t size) noexcept {
            auto [written_size, error] = p.write(static_cast<const uint8_t *>(data), size);
            LUISA_ASSERT(!error, "Failed to write to the process: {}", error.message());
            if (written_size != size) {
                LUISA_ERROR("Failed to write all data to the process "
                            "({}B written, {}B in total).",
                            written_size, size);
            }
        };
        auto size = s.size() + 1u /* for the null-terminator */;
        auto size_str = luisa::format("{:016x}", size);
        write_data(size_str.data(), size_str.size());
        write_data(s.data(), size);
    };
    write(src_filename);
    write(src);
    using namespace std::chrono_literals;
    if (auto [exit_code, error] = p.wait(1024h /* almost forever */); exit_code || error) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to terminate the process: {} (exit code = {}).",
            error.message(), exit_code);
    }
    if (fseek(temp_file, 0, SEEK_END) != 0) {
        LUISA_ERROR_WITH_LOCATION("Failed to seek temp file end.");
    }
    auto length = ftell(temp_file);
    LUISA_ASSERT(length >= 0, "Failed to tell temp file length.");
    if (fseek(temp_file, 0, SEEK_SET) != 0) {
        LUISA_ERROR_WITH_LOCATION("Failed to seek temp file begin.");
    }
    luisa::vector<std::byte> buffer;
    buffer.resize(length);
    if (fread(buffer.data(), 1, length, temp_file) != length) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to read temp file. "
            "The CUDA kernel might be incomplete.");
    }
    if (fclose(temp_file) != 0) {
        LUISA_WARNING_WITH_LOCATION("Failed to close temp file.");
    }
    return buffer;
}

inline auto find_standalone_nvrtc(const luisa::filesystem::path &runtime_dir) noexcept {
#ifdef LUISA_PLATFORM_WINDOWS
    constexpr auto name = "luisa_nvrtc.exe";
#else
    constexpr auto name = "luisa_nvrtc";
#endif
    if (auto p = runtime_dir / name;
        luisa::filesystem::exists(p)) { return p; }
    if (auto p = luisa::filesystem::canonical(luisa::current_executable_path()) / name;
        luisa::filesystem::exists(p)) { return p; }
    LUISA_ERROR_WITH_LOCATION("Cannot find standalone NVRTC compiler '{}'.", name);
}

inline auto query_nvrtc_version(const char *exe_path) {
    // prepare the command line
    luisa::vector<const char *> argv;
    argv.reserve(3u);
    argv.emplace_back(exe_path);
    argv.emplace_back("--version");
    argv.emplace_back(nullptr);

    // setup the options
    reproc::options o;
    o.redirect.out.type = reproc::redirect::pipe;
    o.redirect.err.type = reproc::redirect::parent;

    reproc::process p;
    if (auto error = p.start(reproc::arguments{argv.data()}, o)) {
        LUISA_ERROR_WITH_LOCATION("Failed to start the process: {}.", error.message());
    }
    auto buffer = read_from_subprocess(p, 16u);
    using namespace std::chrono_literals;
    if (auto [exit_code, error] = p.wait(0ms); exit_code || error) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to terminate the process: {} (exit code = {}).",
            error.message(), exit_code);
    }
    // parse the version
    auto begin = reinterpret_cast<const char *>(buffer.data());
    auto end = begin + buffer.size();
    auto v = std::strtoul(begin, const_cast<char **>(&end), 10);
    LUISA_ASSERT(begin != end, "Failed to parse NVRTC version.");
    constexpr auto required_nvrtc_version = 11u * 10000u + 7u * 100u;
    LUISA_ASSERT(v >= required_nvrtc_version, "NVRTC version too old.");
    return static_cast<uint32_t>(v);
}

luisa::vector<std::byte> CUDACompiler::compile(const luisa::string &src, const luisa::string &src_filename,
                                               luisa::span<const char *const> options,
                                               const CUDAShaderMetadata *metadata) const noexcept {

    Clock clk;

#ifndef NDEBUG
    // in debug mode, we always recompute the hash, so
    // that we can check the hash if metadata is provided
    auto hash = compute_hash(src, options);
    if (metadata) { LUISA_ASSERT(metadata->checksum == hash, "Hash mismatch!"); }
#else
    auto hash = metadata ? metadata->checksum : compute_hash(src, options);
#endif

    if (auto ptx = _cache->fetch(hash)) { return *ptx; }
    auto filename = src_filename.empty() ? "my_kernel.cu" : src_filename.c_str();
    auto ptx = compile_with_standalone_compiler(_nvrtc_path.c_str(), src, filename, options);
    LUISA_VERBOSE("CUDACompiler::compile() took {} ms (output PTX size = {}).", clk.toc(), ptx.size());
    return ptx;
}

size_t CUDACompiler::type_size(const Type *type) noexcept {
    if (type == nullptr) { return 1u; }
    if (!type->is_custom()) { return type->size(); }
    // TODO: support custom types
    if (type->description() == "LC_IndirectKernelDispatch") {
        LUISA_ERROR_WITH_LOCATION("Not implemented.");
    }
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

CUDACompiler::CUDACompiler(const CUDADevice *device) noexcept
    : _device{device},
      _device_library{[] {
          luisa::string device_library;
          auto device_half = luisa::string_view{
              luisa_cuda_builtin_cuda_device_half,
              sizeof(luisa_cuda_builtin_cuda_device_half)};
          auto device_math = luisa::string_view{
              luisa_cuda_builtin_cuda_device_math,
              sizeof(luisa_cuda_builtin_cuda_device_math)};
          auto device_resource = luisa::string_view{
              luisa_cuda_builtin_cuda_device_resource,
              sizeof(luisa_cuda_builtin_cuda_device_resource)};

          device_library.resize(device_half.size() +
                                device_math.size() +
                                device_resource.size());
          std::memcpy(device_library.data(),
                      device_half.data(), device_half.size());
          std::memcpy(device_library.data() + device_half.size(),
                      device_math.data(), device_math.size());
          std::memcpy(device_library.data() + device_half.size() + device_math.size(),
                      device_resource.data(), device_resource.size());
          return device_library;
      }()},
      _cache{Cache::create(max_cache_item_count)},
      _nvrtc_path{luisa::to_string(find_standalone_nvrtc(
          device->context().runtime_directory()))},
      _nvrtc_version{query_nvrtc_version(_nvrtc_path.c_str())} {
    LUISA_VERBOSE("CUDA NVRTC compiler version = {}.", _nvrtc_version);
    LUISA_VERBOSE("CUDA device library size = {} bytes.", _device_library.size());
}

uint64_t CUDACompiler::compute_hash(const string &src,
                                    luisa::span<const char *const> options) const noexcept {
    auto hash = hash_value(src);
    for (auto o : options) { hash = hash_value(o, hash); }
    return hash;
}

}// namespace luisa::compute::cuda
