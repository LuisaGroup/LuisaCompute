//
// Created by Mike Smith on 2021/3/15.
//

#include <chrono>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <filesystem>
#include <sstream>
#include <iostream>

#include <core/clock.h>
#include <core/platform.h>
#include <core/logging.h>

#if defined(LUISA_PLATFORM_WINDOWS)

#include <windows.h>
#include <dbghelp.h>

namespace luisa {

void *aligned_alloc(size_t alignment, size_t size) noexcept {
    return _aligned_malloc(size, alignment);
}

void aligned_free(void *p) noexcept {
    _aligned_free(p);
}

namespace detail {

[[nodiscard]] std::string win32_last_error_message() {
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
    auto err_msg = fmt::format("{} (code = 0x{:x}).", static_cast<char *>(buffer), err_code);
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

void *dynamic_module_load(const std::filesystem::path &path) noexcept {
    LUISA_INFO("Loading dynamic module: '{}'.", path.string());
    auto module = LoadLibraryA(path.c_str());
    if (module == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load dynamic module '{}', reason: {}.",
            path.string(), detail::win32_last_error_message());
    }
    return module;
}

void dynamic_module_destroy(void *handle) noexcept {
    if (handle != nullptr) { FreeLibrary(reinterpret_cast<HMODULE>(handle)); }
}

void *dynamic_module_find_symbol(void *handle, std::string_view name_view) noexcept {
    static thread_local std::string name;
    name = name_view;
    LUISA_INFO("Loading dynamic symbol: {}.", name);
    auto symbol = GetProcAddress(reinterpret_cast<HMODULE>(handle), name.c_str());
    if (symbol == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Failed to load symbol '{}', reason: {}.",
                                  name, detail::win32_last_error_message());
    }
    return reinterpret_cast<void *>(symbol);
}

std::filesystem::path dynamic_module_path(
    std::string_view name,
    const std::filesystem::path &search_path) noexcept {
    auto decorated_name = fmt::format("{}.dll", name);
    return search_path.empty() ? std::filesystem::path{decorated_name} : search_path / decorated_name;
}

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

}// namespace luisa

#elif defined(LUISA_PLATFORM_UNIX)

#include <unistd.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <cxxabi.h>

namespace luisa {

void *aligned_alloc(size_t alignment, size_t size) noexcept { return ::aligned_alloc(alignment, size); }
void aligned_free(void *p) noexcept { free(p); }

size_t pagesize() noexcept {
    static thread_local auto page_size = sysconf(_SC_PAGESIZE);
    return page_size;
}

void *dynamic_module_load(const std::filesystem::path &path) noexcept {
    Clock clock;
    auto module = dlopen(path.c_str(), RTLD_LAZY);
    if (module == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load dynamic module '{}', reason: {}.",
            path.string(), dlerror());
    }
    LUISA_INFO(
        "Loaded dynamic module '{}' in {} ms.",
        path.string(), clock.toc());
    return module;
}

void dynamic_module_destroy(void *handle) noexcept {
    if (handle != nullptr) { dlclose(handle); }
}

void *dynamic_module_find_symbol(void *handle, std::string_view name_view) noexcept {
    static thread_local std::string name;
    name = name_view;
    Clock clock;
    auto symbol = dlsym(handle, name.c_str());
    if (symbol == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Failed to load symbol '{}', reason: {}.",
                                  name, dlerror());
    }
    LUISA_INFO(
        "Loading dynamic symbol '{}' in {} ms.",
        name, clock.toc());
    return symbol;
}

std::filesystem::path dynamic_module_path(
    std::string_view name,
    const std::filesystem::path &search_path) noexcept {
    auto decorated_name = fmt::format("lib{}.so", name);
    return search_path.empty() ? std::filesystem::path{decorated_name} : search_path / decorated_name;
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

}// namespace luisa

#endif
