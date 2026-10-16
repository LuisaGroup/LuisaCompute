#include "cuda_tile_ir.h"

#ifdef LUISA_CUDA_TILE_IR_ENABLED

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <fstream>
#include <iterator>
#include <system_error>
#include <utility>
#include <cwchar>

#include <reproc++/reproc.hpp>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/vector.h>

#ifdef LUISA_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
#endif

#endif

namespace luisa::compute::cuda {

bool native_tile_ir_compiler_available() noexcept {
#ifdef LUISA_CUDA_TILE_IR_ENABLED
    return true;
#else
    return false;
#endif
}

#ifdef LUISA_CUDA_TILE_IR_ENABLED
namespace {

class TemporaryTileDirectory {
private:
    luisa::filesystem::path _path;

public:
    TemporaryTileDirectory() noexcept = default;
    TemporaryTileDirectory(const TemporaryTileDirectory &) = delete;
    TemporaryTileDirectory &operator=(const TemporaryTileDirectory &) = delete;
    ~TemporaryTileDirectory() noexcept {
        if (!_path.empty()) {
            std::error_code error;
            // _path is assigned only after this process exclusively creates a
            // direct child of the OS temporary directory below.
            luisa::filesystem::remove_all(_path, error);
            if (error) { LUISA_WARNING("CUDA Tile IR temporary cleanup failed: {}", error.message()); }
        }
    }
    [[nodiscard]] const auto &path() const noexcept { return _path; }
    [[nodiscard]] bool create(luisa::string &diagnostic) noexcept {
        std::error_code error;
        auto base = luisa::filesystem::temp_directory_path(error);
        if (error) {
            diagnostic = luisa::format("CUDA Tile IR cannot locate temporary storage: {}", error.message());
            return false;
        }
        base = luisa::filesystem::absolute(base, error);
        if (error) {
            diagnostic = luisa::format("CUDA Tile IR cannot resolve temporary storage: {}", error.message());
            return false;
        }
        static std::atomic<uint64_t> sequence{0u};
        auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (auto attempt = 0u; attempt < 128u; attempt++) {
            auto name = luisa::format("luisa-cuda-tile-{:x}-{:x}", stamp, sequence.fetch_add(1u, std::memory_order_relaxed));
            auto candidate = base / name.c_str();
            if (luisa::filesystem::create_directory(candidate, error)) {
                _path = std::move(candidate);
                return true;
            }
            if (error) {
                diagnostic = luisa::format("CUDA Tile IR cannot create temporary storage: {}", error.message());
                return false;
            }
        }
        diagnostic = "CUDA Tile IR cannot allocate a unique temporary directory";
        return false;
    }
};

[[nodiscard]] luisa::string read_diagnostics(const luisa::filesystem::path &path) noexcept {
    std::error_code error;
    auto length = luisa::filesystem::file_size(path, error);
    if (error) { return {}; }
    constexpr auto limit = uintmax_t{128u * 1024u};
    auto size = static_cast<size_t>(std::min(length, limit));
    std::ifstream file{path, std::ios::binary};
    luisa::string text(size, '\0');
    file.read(text.data(), static_cast<std::streamsize>(size));
    text.resize(static_cast<size_t>(file.gcount()));
    if (length > limit) { text += "\n[compiler diagnostics truncated at 128 KiB]"; }
    return text;
}

[[nodiscard]] bool encode_tile_path(const luisa::filesystem::path &path,
                                    luisa::string &text, luisa::string &diagnostic) noexcept {
    // reproc consumes UTF-8 on all platforms. luisa::to_string(path) preserves
    // the Windows ANSI code page for legacy consumers, so it is unsuitable here.
    try {
        auto utf8 = path.u8string();
        text.assign(reinterpret_cast<const char *>(utf8.data()), utf8.size());
        return true;
    } catch (const std::exception &error) {
        diagnostic = luisa::format("CUDA Tile IR cannot encode a process path as UTF-8: {}", error.what());
        return false;
    }
}

#ifdef LUISA_PLATFORM_WINDOWS
[[nodiscard]] bool assembler_environment(const luisa::filesystem::path &directory,
                                          luisa::vector<luisa::string> &environment,
                                          luisa::string &diagnostic) noexcept {
    luisa::string temporary;
    if (!encode_tile_path(directory, temporary, diagnostic)) { return false; }
    auto ascii = [](luisa::string_view text) noexcept {
        return std::all_of(text.begin(), text.end(), [](unsigned char c) noexcept { return c < 128u; });
    };
    if (!ascii(temporary)) {
        auto size = GetShortPathNameW(directory.c_str(), nullptr, 0u);
        luisa::vector<wchar_t> short_path(size);
        auto written = size == 0u ? 0u : GetShortPathNameW(directory.c_str(), short_path.data(), size);
        if (written == 0u || written >= size ||
            !encode_tile_path(luisa::filesystem::path{short_path.data()}, temporary, diagnostic) || !ascii(temporary)) {
            diagnostic = "CUDA tileiras requires an ASCII temporary path or an ASCII Windows short-path alias; set TEMP to an ASCII directory on volumes without 8.3 aliases";
            return false;
        }
    }
    // reproc's Windows 'extend' concatenates duplicate environment keys. Build
    // an exact replacement block, retaining every unrelated parent setting.
    auto block = GetEnvironmentStringsW();
    if (block == nullptr) {
        diagnostic = "CUDA Tile IR could not read the child compiler environment";
        return false;
    }
    for (auto entry = block; *entry != L'\0'; entry += std::wcslen(entry) + 1u) {
        if (_wcsnicmp(entry, L"TEMP=", 5u) == 0 || _wcsnicmp(entry, L"TMP=", 4u) == 0 ||
            _wcsnicmp(entry, L"TMPDIR=", 7u) == 0) { continue; }
        auto bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, entry, -1, nullptr, 0, nullptr, nullptr);
        if (bytes == 0) {
            FreeEnvironmentStringsW(block);
            diagnostic = "CUDA Tile IR could not encode the child compiler environment";
            return false;
        }
        luisa::string text(static_cast<size_t>(bytes), '\0');
        auto converted = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, entry, -1, text.data(), bytes, nullptr, nullptr);
        if (converted != bytes) {
            FreeEnvironmentStringsW(block);
            diagnostic = "CUDA Tile IR could not encode the complete child compiler environment";
            return false;
        }
        text.pop_back();
        environment.emplace_back(std::move(text));
    }
    FreeEnvironmentStringsW(block);
    for (auto key : {"TEMP=", "TMP=", "TMPDIR="}) { environment.emplace_back(luisa::format("{}{}", key, temporary)); }
    return true;
}
#endif

[[nodiscard]] bool run_tile_tool(luisa::span<const char *const> arguments,
                                 const luisa::filesystem::path &log_path,
                                 luisa::string_view stage, luisa::string &diagnostic,
                                 bool assembler = false) noexcept {
    luisa::string log_name, working_directory;
    if (!encode_tile_path(log_path, log_name, diagnostic) ||
        !encode_tile_path(log_path.parent_path(), working_directory, diagnostic)) { return false; }
    reproc::options options;
    options.working_directory = working_directory.c_str();
#ifdef LUISA_PLATFORM_WINDOWS
    luisa::vector<luisa::string> environment;
    luisa::vector<const char *> environment_pointers;
    if (assembler) {
        if (!assembler_environment(log_path.parent_path(), environment, diagnostic)) { return false; }
        for (auto &&entry : environment) { environment_pointers.emplace_back(entry.c_str()); }
        environment_pointers.emplace_back(nullptr);
        options.env.behavior = reproc::env::empty;
        options.env.extra = reproc::env{environment_pointers.data()};
    }
#else
    static_cast<void>(assembler);
#endif
    options.redirect.in.type = reproc::redirect::discard;
    options.redirect.out.type = reproc::redirect::path_;
    options.redirect.out.path = log_name.c_str();
    options.redirect.err.type = reproc::redirect::stdout_;
    using namespace std::chrono_literals;
    options.stop.first = {reproc::stop::kill, 5s};
    // reproc uses argv directly and STARTF_USESHOWWINDOW/SW_HIDE on Windows.
    reproc::process child;
    if (auto error = child.start(reproc::arguments{arguments.data()}, options)) {
        diagnostic = luisa::format("CUDA Tile IR {} could not start: {}", stage, error.message());
        return false;
    }
    auto [status, error] = child.wait(5min);
    if (error) {
        static_cast<void>(child.kill());
        static_cast<void>(child.wait(5s));
    }
    auto text = read_diagnostics(log_path);
    if (error || status != 0) {
        diagnostic = luisa::format("CUDA Tile IR {} failed (exit {:#010x}, {}):\n{}",
                                   stage, static_cast<uint32_t>(status), error.message(), text);
        return false;
    }
    if (!text.empty()) { LUISA_VERBOSE("CUDA Tile IR {}:\n{}", stage, text); }
    return true;
}

}// namespace
#endif

NativeTileBinary compile_native_tile_ir(const luisa::filesystem::path &runtime_directory,
                                        luisa::string_view source, uint32_t architecture,
                                        bool debug_info) noexcept {
    NativeTileBinary result;
#ifdef LUISA_CUDA_TILE_IR_ENABLED
#ifdef LUISA_PLATFORM_WINDOWS
    constexpr auto helper_name = "luisa_cuda_tile_compiler.exe";
    constexpr auto host_option = "--host-os=windows";
#else
    constexpr auto helper_name = "luisa_cuda_tile_compiler";
    constexpr auto host_option = "--host-os=linux";
#endif
    std::error_code error;
    auto helper = runtime_directory / helper_name;
    if (!luisa::filesystem::is_regular_file(helper, error)) {
        result.error = "CUDA Tile IR standalone compiler is missing from the runtime directory";
        return result;
    }
    const auto assembler = luisa::filesystem::u8path(LUISA_CUDA_TILE_IR_ASSEMBLER);
    error.clear();
    if (!luisa::filesystem::is_regular_file(assembler, error)) {
        result.error = "CUDA Tile IR requires the configured CUDA 13.4+ tileiras executable";
        return result;
    }
    if (source.empty() || architecture == 0u) {
        result.error = "CUDA Tile IR requires nonempty source and a physical device architecture";
        return result;
    }
    TemporaryTileDirectory temporary;
    if (!temporary.create(result.error)) { return result; }
    auto source_path = temporary.path() / "kernel.cu";
    auto ir_path = temporary.path() / "kernel.tilebc";
    auto cubin_path = temporary.path() / "kernel.cubin";
    {
        std::ofstream file{source_path, std::ios::binary | std::ios::trunc};
        file.write(source.data(), static_cast<std::streamsize>(source.size()));
        file.close();
        if (!file) {
            result.error = "CUDA Tile IR could not write its temporary source";
            return result;
        }
    }
    luisa::string helper_text, source_text, ir_text;
    if (!encode_tile_path(helper, helper_text, result.error) ||
        !encode_tile_path(source_path, source_text, result.error) ||
        !encode_tile_path(ir_path, ir_text, result.error)) { return result; }
    auto arch_text = luisa::format("{}", architecture);
    const char *compiler_arguments[]{helper_text.c_str(), source_text.c_str(), ir_text.c_str(),
                                     arch_text.c_str(), debug_info ? "1" : "0", nullptr};
    if (!run_tile_tool(compiler_arguments, temporary.path() / "nvrtc.log", "NVRTC", result.error)) { return result; }

    luisa::string assembler_text;
    if (!encode_tile_path(assembler, assembler_text, result.error)) { return result; }
    auto gpu_option = luisa::format("--gpu-name=sm_{}", architecture);
    // tileiras uses narrow file arguments on Windows. Its child-only working
    // directory is set through reproc's UTF-16 process API, so relative ASCII
    // filenames also work when the OS temporary directory contains Unicode.
    const char *assembler_arguments[]{assembler_text.c_str(), gpu_option.c_str(), host_option,
                                      "kernel.tilebc", "-o", "kernel.cubin", nullptr};
    if (!run_tile_tool(assembler_arguments, temporary.path() / "tileiras.log", "tileiras", result.error, true)) { return result; }

    error.clear();
    auto size = luisa::filesystem::file_size(cubin_path, error);
    constexpr auto maximum_size = uintmax_t{256u * 1024u * 1024u};
    if (error || size < 64u || size > maximum_size) {
        result.error = "CUDA Tile IR assembler returned a missing or invalid-sized cubin";
        return result;
    }
    result.cubin.resize(static_cast<size_t>(size));
    std::ifstream file{cubin_path, std::ios::binary};
    file.read(reinterpret_cast<char *>(result.cubin.data()), static_cast<std::streamsize>(size));
    constexpr std::byte magic[]{std::byte{0x7f}, std::byte{'E'}, std::byte{'L'}, std::byte{'F'}};
    if (!file || !std::equal(std::begin(magic), std::end(magic), result.cubin.begin())) {
        result.cubin.clear();
        result.error = "CUDA Tile IR assembler output is incomplete or is not an ELF cubin";
    }
#else
    static_cast<void>(runtime_directory);
    static_cast<void>(source);
    static_cast<void>(architecture);
    static_cast<void>(debug_info);
    result.error = "CUDA Tile IR is unavailable: configure CMake with CUDA 13.4+ Tile headers and tileiras";
#endif
    return result;
}

}// namespace luisa::compute::cuda
