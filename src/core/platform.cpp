//
// Created by Mike Smith on 2021/3/15.
//

#include <cstdlib>
#include <string>
#include <type_traits>
#include <filesystem>
#include <fmt/format.h>
#include <core/platform.h>
#include <core/logging.h>

#if defined(LUISA_PLATFORM_WINDOWS)

#include <windows.h>

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
    if (!std::filesystem::exists(path)) {
        LUISA_ERROR_WITH_LOCATION("Dynamic module not found: {}", path.string());
    } else {
        LUISA_INFO("Loading dynamic module: '{}'", path.string());
    }
    auto canonical_path = std::filesystem::canonical(path).string();
    auto module = LoadLibraryA(canonical_path.c_str());
    if (module == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load dynamic module '{}', reason: {}",
            canonical_path, detail::win32_last_error_message());
    }
    return module;
}

void dynamic_module_destroy(void *handle) noexcept {
    if (handle != nullptr) { FreeLibrary(reinterpret_cast<HMODULE>(handle)); }
}

void *dynamic_module_find_symbol(void *handle, std::string_view name_view) noexcept {
    static thread_local std::string name;
    name = name_view;
    LUISA_INFO("Loading dynamic symbol: {}", name);
    auto symbol = GetProcAddress(reinterpret_cast<HMODULE>(handle), name.c_str());
    if (symbol == nullptr) {
        LUISA_ERROR("Failed to load symbol '{}', reason: {}",
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

}// namespace luisa

#elif defined(LUISA_PLATFORM_UNIX)

#include <unistd.h>
#include <dlfcn.h>

namespace luisa {

void *aligned_alloc(size_t alignment, size_t size) noexcept { return ::aligned_alloc(alignment, size); }
void aligned_free(void *p) noexcept { free(p); }

size_t pagesize() noexcept {
    static thread_local auto page_size = sysconf(_SC_PAGESIZE);
    return page_size;
}

void *dynamic_module_load(const std::filesystem::path &path) noexcept {
    if (!std::filesystem::exists(path)) {
        LUISA_ERROR_WITH_LOCATION("Dynamic module not found: {}", path.string());
    } else {
        LUISA_INFO("Loading dynamic module: '{}'", path.string());
    }
    auto canonical_path = std::filesystem::canonical(path).string();
    auto module = dlopen(canonical_path.c_str(), RTLD_LAZY);
    if (module == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load dynamic module '{}', reason: {}",
            canonical_path, dlerror());
    }
    return module;
}

void dynamic_module_destroy(void *handle) noexcept {
    if (handle != nullptr) { dlclose(handle); }
}

void *dynamic_module_find_symbol(void *handle, std::string_view name_view) noexcept {
    static thread_local std::string name;
    name = name_view;
    LUISA_INFO("Loading dynamic symbol: {}", name);
    auto symbol = dlsym(handle, name.c_str());
    if (symbol == nullptr) {
        LUISA_ERROR("Failed to load symbol '{}', reason: {}", name, dlerror());
    }
    return symbol;
}

std::filesystem::path dynamic_module_path(
    std::string_view name,
    const std::filesystem::path &search_path) noexcept {
    auto decorated_name = fmt::format("lib{}.so", name);
    return search_path.empty() ? std::filesystem::path{decorated_name} : search_path / decorated_name;
}

}// namespace luisa

#endif
