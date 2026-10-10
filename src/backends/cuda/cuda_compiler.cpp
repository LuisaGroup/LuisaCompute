#include <luisa/core/clock.h>
#include <luisa/core/binary_io.h>
#include "../common/subprocess.h"
#include "cuda_error.h"
#include "cuda_device.h"
#include "optix_api.h"
#include "cuda_builtin_embedded.hpp"
#include "cuda_compiler.h"

#include <cerrno>
#include <cstdio>
#include <system_error>

#ifdef LUISA_PLATFORM_WINDOWS
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#endif

namespace luisa::compute::cuda {

namespace {

[[nodiscard]] FILE *create_compiler_temp_file(std::error_code &error) {
#ifdef LUISA_PLATFORM_WINDOWS
    // The Windows CRT tmpfile() uses a narrow path. Preserve Unicode TEMP/TMP
    // paths while retaining its binary, exclusive, delete-on-close semantics.
    auto directory = luisa::filesystem::temp_directory_path(error);
    if (error) { return nullptr; }
    static std::atomic<uint64_t> sequence{0u};
    auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    for (auto attempt = 0u; attempt < 128u; attempt++) {
        auto name = luisa::format("luisa-cuda-{:x}-{:x}.tmp", stamp,
                                 sequence.fetch_add(1u, std::memory_order_relaxed));
        auto path = directory / name.c_str();
        int descriptor = -1;
        auto result = _wsopen_s(&descriptor, path.c_str(),
                               _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY | _O_TEMPORARY,
                               _SH_DENYNO, _S_IREAD | _S_IWRITE);
        if (result == EEXIST) { continue; }
        if (result != 0) {
            error = std::error_code{result, std::generic_category()};
            return nullptr;
        }
        // On success, fclose() owns descriptor cleanup, including file deletion.
        if (auto file = _fdopen(descriptor, "w+b")) { return file; }
        error = std::error_code{errno, std::generic_category()};
        _close(descriptor);
        return nullptr;
    }
    error = std::make_error_code(std::errc::file_exists);
    return nullptr;
#else
    auto file = std::tmpfile();
    if (file == nullptr) { error = std::error_code{errno, std::generic_category()}; }
    return file;
#endif
}

}// namespace

[[nodiscard]] inline auto read_from_subprocess(reproc::process &p, reproc::stream stream, size_t chunk_size = 4_k) noexcept {
    luisa::vector<std::byte> buffer;
    for (;;) {
        auto current_size = buffer.size();
        buffer.resize(luisa::next_pow2(buffer.size() + chunk_size));
        auto max_read = buffer.size() - current_size;
        auto [read_size, error] = p.read(
            stream, reinterpret_cast<uint8_t *>(buffer.data() + current_size), max_read);
        if (error) {
            buffer.resize(current_size);
            break;
        }
        buffer.resize(current_size + read_size);
    }
    return buffer;
}

namespace {

struct StandaloneNvrtcResult {
    luisa::vector<std::byte> binary;// the PTX/OptiX IR the child wrote to the file
    luisa::string log; // everything the child wrote to stderr (compiler diagnostics)
    int exit_code{0}; // the 32-bit status the child terminated with
    bool wait_failed{false}; // wait() itself failed: `exit_code` is meaningless
    std::error_code wait_error{};
};

}// namespace

[[nodiscard]] static auto compile_with_standalone_compiler(
    const char *exe_path,
    const luisa::string &src, const luisa::string &src_filename,
    luisa::span<const char *const> options) {

    // prepare the command line
    luisa::vector<const char *> argv;
    argv.reserve(options.size() + 2u);
    argv.emplace_back(exe_path);
    for (auto o : options) { argv.emplace_back(o); }
    argv.emplace_back(nullptr);

    std::error_code temp_file_error;
    auto temp_file = create_compiler_temp_file(temp_file_error);
    LUISA_ASSERT(temp_file != nullptr,
                 "Failed to create temp file for CUDA compiler: {}.", temp_file_error.message());

    // setup the options
    reproc::options o;
    o.redirect.in.type = reproc::redirect::pipe;
    // The child's stderr carries the NVRTC diagnostics: capture it so a
    // failed compile reports the compiler's own message instead of a bare
    // "empty PTX". (The child only writes to stderr once, right before it
    // exits, so draining the pipe to EOF before wait() cannot deadlock.)
    o.redirect.err.type = reproc::redirect::pipe;
    o.redirect.out.type = reproc::redirect::file_;
    o.redirect.out.file = temp_file;

    reproc::process p;
    if (auto error = p.start(reproc::arguments{argv.data()}, o)) {
        fclose(temp_file);
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
    auto result = StandaloneNvrtcResult{};
    // Drain stderr to EOF (the child closes it on exit).
    auto log_bytes = read_from_subprocess(p, reproc::stream::err);
    result.log.assign(reinterpret_cast<const char *>(log_bytes.data()),
                      log_bytes.size());
    using namespace std::chrono_literals;
    if (auto [exit_code, error] = p.wait(1024h /* almost forever */); error) {
        // `error` describes a failed wait; `exit_code` is meaningless then.
        result.wait_failed = true;
        result.wait_error = error;
    } else {
        // `exit_code` is the 32-bit status the compiler terminated with, which
        // is an NTSTATUS (e.g. 0xC0000005 for a crash) and therefore negative
        // as an `int`; the caller reports it in hex.
        result.exit_code = exit_code;
    }
    if (fseek(temp_file, 0, SEEK_END) != 0) {
        LUISA_ERROR_WITH_LOCATION("Failed to seek temp file end.");
    }
    auto length = ftell(temp_file);
    LUISA_ASSERT(length >= 0, "Failed to tell temp file length.");
    if (fseek(temp_file, 0, SEEK_SET) != 0) {
        LUISA_ERROR_WITH_LOCATION("Failed to seek temp file begin.");
    }
    result.binary.resize(length);
    if (length > 0 && fread(result.binary.data(), 1, length, temp_file) != length) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to read temp file. "
            "The CUDA kernel might be incomplete.");
    }
    if (fclose(temp_file) != 0) {
        LUISA_WARNING_WITH_LOCATION("Failed to close temp file.");
    }
    return result;
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
    auto buffer = read_from_subprocess(p, reproc::stream::out, 16u);
    using namespace std::chrono_literals;
    if (auto [exit_code, error] = p.wait(0ms); exit_code || error) {
        // `exit_code` is the 32-bit status the compiler terminated with, which is an
        // NTSTATUS (e.g. 0xC0000005 for a crash) and therefore negative as an `int`.
        // Report it in hex; `error` only describes a failed wait.
        LUISA_WARNING_WITH_LOCATION(
            "Failed to terminate the process: {} (exit code = {:#010x}).",
            error.message(), static_cast<uint32_t>(exit_code));
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
                                               const CUDAShaderMetadata *metadata,
                                               luisa::string *error) const noexcept {

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
    auto compiled = compile_with_standalone_compiler(_nvrtc_path.c_str(), src, filename, options);
    // The child always prints its diagnostics (errors, or warnings on success);
    // surface them through the host log so they stay visible now that stderr is
    // captured instead of inherited.
    auto trimmed_log = luisa::string_view{compiled.log};
    while (!trimmed_log.empty() &&
           (trimmed_log.back() == '\n' || trimmed_log.back() == '\r')) {
        trimmed_log = trimmed_log.substr(0u, trimmed_log.size() - 1u);
    }
    if (compiled.wait_failed) {
        auto message = luisa::format(
            "the NVRTC compiler process failed while compiling '{}': {}",
            filename, compiled.wait_error.message());
        if (!trimmed_log.empty()) { message.append("\n").append(trimmed_log.data(), trimmed_log.size()); }
        if (error != nullptr) { *error = message; } else { LUISA_WARNING("{}", message); }
        return {};
    }
    if (compiled.exit_code != 0 || compiled.binary.empty()) {
        // `exit_code` is an NTSTATUS on Windows, so negative as an `int`:
        // report its bits in hex.
        auto message = luisa::format(
            "NVRTC failed to compile '{}' (compiler exit code {:#010x})",
            filename, static_cast<uint32_t>(compiled.exit_code));
        if (trimmed_log.empty()) {
            message.append(": the compiler reported no diagnostics");
        } else {
            message.append(":\n").append(trimmed_log.data(), trimmed_log.size());
        }
        if (error != nullptr) { *error = message; } else { LUISA_WARNING("{}", message); }
        return {};
    }
    if (!trimmed_log.empty()) {
        LUISA_WARNING("NVRTC diagnostics for '{}': {}", filename, trimmed_log);
    }
    auto ptx = std::move(compiled.binary);
    // Fill the in-memory LRU so repeated compile() calls for the same source
    // and options (for example a recreated Tile shader) do not re-run NVRTC.
    _cache->update(hash, ptx);
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
      _cache{Cache::create(max_cache_item_count)},
      _nvrtc_path{luisa::to_string(find_standalone_nvrtc(device->context().runtime_directory()))},
      _nvrtc_version{query_nvrtc_version(_nvrtc_path.c_str())} {
    LUISA_VERBOSE("CUDA NVRTC compiler version = {}.", _nvrtc_version);
    process_builtin(_device_library, reinterpret_cast<const char *>(luisa_compute_cuda_device_half), luisa_compute_cuda_device_half_size);
    process_builtin(_device_library, reinterpret_cast<const char *>(luisa_compute_cuda_device_math), luisa_compute_cuda_device_math_size);
    process_builtin(_device_library, reinterpret_cast<const char *>(luisa_compute_cuda_device_resource), luisa_compute_cuda_device_resource_size);
    process_builtin(_device_library, reinterpret_cast<const char *>(luisa_compute_cuda_device_coop), luisa_compute_cuda_device_coop_size);
    // Kept separate from `_device_library`: only a fallback-mode shader appends
    // it (see CUDADevice::create_shader), so the generated source - and its
    // hash, and the cached PTX - of every hardware-path kernel is unchanged.
    process_builtin(_fallback_rtx_device_library, reinterpret_cast<const char *>(luisa_compute_cuda_device_fallback_rtx), luisa_compute_cuda_device_fallback_rtx_size);
}

void CUDACompiler::process_builtin(luisa::string &result, char const *data, size_t size) noexcept {
    if (size > 0u && data[size - 1] == '\0') { size--; }
    auto n = size - std::count(data, data + size, '\r');
    auto old_size = result.size();
    result.resize(result.size() + n);
    std::copy_if(data, data + size, result.begin() + old_size,
                 [](char c) noexcept { return c != '\r'; });
}

uint64_t CUDACompiler::compute_hash(const string &src, luisa::span<const char *const> options) noexcept {
    auto hash = hash_value(src);
    for (auto o : options) { hash = hash_value(o, hash); }
    return hash;
}

}// namespace luisa::compute::cuda
