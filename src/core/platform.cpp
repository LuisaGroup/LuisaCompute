#include <sstream>

#include <luisa/core/clock.h>
#include <luisa/core/platform.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>

static_assert(sizeof(void *) == 8 && sizeof(int) == 4 && sizeof(char) == 1,
              "illegal pointer and integer sizes.");

#if defined(LUISA_PLATFORM_WINDOWS)

#ifndef UNICODE
#define UNICODE 1
#endif

#ifndef NOMINMAX
#define NOMINMAX 1
#endif

#include <windows.h>
#include <DbgHelp.h>

namespace luisa {

void *aligned_alloc(size_t alignment, size_t size) noexcept {
    return _aligned_malloc(size, alignment);
}

void aligned_free(void *p) noexcept {
    _aligned_free(p);
}

namespace detail {

[[nodiscard]] luisa::string win32_last_error_message() {
    // Retrieve the system error message for the last-error code
    void *buffer = nullptr;
    auto err_code = GetLastError();
    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        err_code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&buffer,
        0, nullptr);
    luisa::string err_msg{fmt::format("{} (code = 0x{:x}).", static_cast<char *>(buffer), err_code)};
    LocalFree(buffer);
    return err_msg;
}

}// namespace detail

size_t pagesize() noexcept {
    static thread_local auto page_size = [] {
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        return info.dwPageSize;
    }();
    return page_size;
}
namespace win_detail {

template<typename PathChar>
void set_dll_directory(const PathChar *path) noexcept {
    if constexpr (sizeof(PathChar) == 1) {
        SetDllDirectoryA(path);
    } else {
        SetDllDirectoryW(path);
    }
}

}// namespace win_detail
void *dynamic_module_load(const luisa::filesystem::path &path) noexcept {
    bool has_parent_path = path.has_parent_path();
    using PathType = std::filesystem::path::value_type;
    if (has_parent_path) {
        win_detail::set_dll_directory(path.parent_path().c_str());
    }
    auto path_string = luisa::to_string(path.filename());
    auto module = LoadLibraryA(path_string.c_str());
    if (module == nullptr) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load dynamic module '{}', reason: {}.",
            path_string, detail::win32_last_error_message());
    }
    if (has_parent_path) {
        win_detail::set_dll_directory<PathType>(nullptr);
    }
    return module;
}

void dynamic_module_destroy(void *handle) noexcept {
    if (handle != nullptr) { FreeLibrary(reinterpret_cast<HMODULE>(handle)); }
}

void *dynamic_module_find_symbol(void *handle, luisa::string_view name_view) noexcept {
    static thread_local luisa::string name;
    name = name_view;
    auto symbol = GetProcAddress(reinterpret_cast<HMODULE>(handle), name.c_str());
    if (symbol == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Failed to load symbol '{}', reason: {}.",
                                  name, detail::win32_last_error_message());
    }
    return reinterpret_cast<void *>(symbol);
}

luisa::string dynamic_module_name(luisa::string_view name) noexcept {
    luisa::string s{name};
    s.append(".dll");
    return s;
}

#ifndef NDEBUG
luisa::string demangle(const char *name) noexcept {
    char buffer[256u];
    auto length = UnDecorateSymbolName(name, buffer, 256, 0);
    return {buffer, length};
}

luisa::vector<TraceItem> backtrace() noexcept {

    void *stack[100];
    auto process = GetCurrentProcess();
    SymInitialize(process, nullptr, true);
    auto frame_count = CaptureStackBackTrace(0, 100, stack, nullptr);

    struct Symbol : SYMBOL_INFO {
        char name_storage[1023];
    } symbol{};
    symbol.MaxNameLen = 1024;
    symbol.SizeOfStruct = sizeof(SYMBOL_INFO);
    IMAGEHLP_MODULE64 module{};
    module.SizeOfStruct = sizeof(IMAGEHLP_MODULE64);
    luisa::vector<TraceItem> trace;
    trace.reserve(frame_count - 1u);
    for (auto i = 1u; i < frame_count; i++) {
        auto address = reinterpret_cast<uint64_t>(stack[i]);
        auto displacement = 0ull;
        if (SymFromAddr(process, address, &displacement, &symbol)) {
            TraceItem item{};
            if (SymGetModuleInfo64(process, symbol.ModBase, &module)) {
                item.module = module.ModuleName;
            } else {
                item.module = "???";
            }
            item.symbol = symbol.Name;
            item.address = address;
            item.offset = displacement;
            trace.emplace_back(std::move(item));
        } else {
            LUISA_VERBOSE_WITH_LOCATION(
                "Failed to get stacktrace at 0x{:012}: {}",
                address, detail::win32_last_error_message());
        }
    }
    return trace;
}
#else
luisa::vector<TraceItem> backtrace() noexcept { return {}; }
#endif

luisa::string cpu_name() noexcept {
    int32_t brand[12];
    __cpuid(&brand[0], 0x80000002);
    __cpuid(&brand[4], 0x80000003);
    __cpuid(&brand[8], 0x80000004);
    return reinterpret_cast<const char *>(brand);
}

}// namespace luisa

#elif defined(LUISA_PLATFORM_UNIX)

#include <unistd.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <cxxabi.h>

#ifdef LUISA_ARCH_ARM64
#include <sys/types.h>
#include <sys/sysctl.h>
#else
#include <cpuid.h>
#endif

namespace luisa {

void *aligned_alloc(size_t alignment, size_t size) noexcept { return ::aligned_alloc(alignment, size); }
void aligned_free(void *p) noexcept { free(p); }

size_t pagesize() noexcept {
    static thread_local auto page_size = sysconf(_SC_PAGESIZE);
    return page_size;
}

void *dynamic_module_load(const luisa::filesystem::path &path) noexcept {
    auto p = path;
    for (auto ext : {".so", ".dylib"}) {
        p.replace_extension(ext);
        if (auto module = dlopen(p.c_str(), RTLD_LAZY); module != nullptr) {
            return module;
        }
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load dynamic module '{}', reason: {}.",
            luisa::to_string(p), dlerror());
    }
    return nullptr;
}

void dynamic_module_destroy(void *handle) noexcept {
    if (handle != nullptr) { dlclose(handle); }
}

void *dynamic_module_find_symbol(void *handle, luisa::string_view name_view) noexcept {
    static thread_local luisa::string name;
    name = name_view;
    Clock clock;
    auto symbol = dlsym(handle, name.c_str());
    if (symbol == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Failed to load symbol '{}', reason: {}.",
                                  name, dlerror());
    }
    LUISA_VERBOSE_WITH_LOCATION(
        "Loading dynamic symbol '{}' in {} ms.",
        name, clock.toc());
    return symbol;
}

luisa::string dynamic_module_name(luisa::string_view name) noexcept {
    luisa::string s{"lib"};
    s.append(name).append(".so");
    return s;
}

luisa::string demangle(const char *name) noexcept {
    auto status = 0;
    auto buffer = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    luisa::string demangled{buffer == nullptr ? name : buffer};
    free(buffer);
    return demangled;
}

luisa::vector<TraceItem> backtrace() noexcept {
    void *trace[100u];
    auto count = ::backtrace(trace, 100);
    auto info = ::backtrace_symbols(trace, count);
    luisa::vector<TraceItem> trace_info;
    trace_info.reserve(count - 1u);
    for (auto i = 1 /* skip current frame */; i < count; i++) {
        std::istringstream iss{info[i]};
        auto index = 0;
        char plus = '+';
        TraceItem item{};
        iss >> index >> item.module >> std::hex >> item.address >> item.symbol >> plus >> std::dec >> item.offset;
        item.symbol = demangle(item.symbol.c_str());
        trace_info.emplace_back(std::move(item));
    }
    free(info);
    return trace_info;
}

#ifdef LUISA_ARCH_ARM64
luisa::string cpu_name() noexcept {
    constexpr auto buffer_size = static_cast<size_t>(256u);
    char brand[buffer_size];
    auto size = buffer_size;
    if (sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0) != 0) {
        return "Unknown ARM64";
    }
    return brand;
}
#else
luisa::string cpu_name() noexcept {
    uint32_t brand[12];
    if (!__get_cpuid_max(0x80000004u, nullptr)) { return "Unknown x86_64"; }
    __get_cpuid(0x80000002u, brand + 0x0u, brand + 0x1u, brand + 0x2u, brand + 0x3u);
    __get_cpuid(0x80000003u, brand + 0x4u, brand + 0x5u, brand + 0x6u, brand + 0x7u);
    __get_cpuid(0x80000004u, brand + 0x8u, brand + 0x9u, brand + 0xau, brand + 0xbu);
    return reinterpret_cast<const char *>(brand);
}
#endif

}// namespace luisa

#endif

