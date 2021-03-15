//
// Created by Mike Smith on 2021/3/15.
//

#include <cstdlib>
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
    auto err_msg = fmt::format("{} (code = 0x{:x}).", buffer, err_code);
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

void *dynamic_module_load(const char *path) noexcept {
    if (!std::filesystem::exists(path)) {
        LUISA_ERROR_WITH_LOCATION("Dynamic module not found: {}", path);
    } else {
        LUISA_INFO("Loading dynamic module: '{}'", path);
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

void *dynamic_module_find_symbol(void *handle, const char *name) noexcept {
    LUISA_INFO("Loading dynamic symbol: {}", name);
    auto symbol = GetProcAddress(reinterpret_cast<HMODULE>(handle), name);
    if (symbol == nullptr) {
        LUISA_ERROR("Failed to load symbol '{}', reason: {}",
                    name, detail::win32_last_error_message());
    }
    return reinterpret_cast<void *>(symbol);
}

const char *dynamic_module_prefix() noexcept { return ""; }
const char *dynamic_module_extension() noexcept { return ".dll"; }

}// namespace luisa

#elif defined(LUISA_PLATFORM_UNIX)

#include <unistd.h>
#include <dlfcn.h>

namespace luisa {

void *aligned_alloc(size_t alignment, size_t size) noexcept { return ::aligned_alloc(alignment, size); }
void aligned_free(void *p) noexcept { free(p); }

size_t pagesize() noexcept {
    static thread_local auto page_size = getpagesize();
    return page_size;
}

void *dynamic_module_load(const char *path) noexcept {
    if (!std::filesystem::exists(path)) {
        LUISA_ERROR_WITH_LOCATION("Dynamic module not found: {}", path);
    } else {
        LUISA_INFO("Loading dynamic module: '{}'", path);
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

void *dynamic_module_find_symbol(void *handle, const char *name) noexcept {
    LUISA_INFO("Loading dynamic symbol: {}", name);
    auto symbol = dlsym(handle, name);
    if (symbol == nullptr) {
        LUISA_ERROR("Failed to load symbol '{}', reason: {}", name, dlerror());
    }
    return symbol;
}

const char *dynamic_module_prefix() noexcept { return "lib"; }
const char *dynamic_module_extension() noexcept { return ".so"; }

}// namespace luisa

#endif
