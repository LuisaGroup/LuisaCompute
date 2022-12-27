//
// Created by Mike Smith on 2021/2/11.
//

#pragma once

#include <core/dll_export.h>

#if defined(_WINDOWS) || defined(_WIN32) || defined(_WIN64)
#define LUISA_PLATFORM_WINDOWS
#elif defined(__unix__) || defined(__unix) || defined(__APPLE__)
#define LUISA_PLATFORM_UNIX
#ifdef __APPLE__
#define LUISA_PLATFORM_APPLE
#endif
#endif

#ifdef __cplusplus

#include <core/stl/vector.h>
#include <core/stl/string.h>
#include <core/stl/filesystem.h>

namespace luisa {

[[nodiscard]] LC_CORE_API void *aligned_alloc(size_t alignment, size_t size) noexcept;
LC_CORE_API void aligned_free(void *p) noexcept;
[[nodiscard]] LC_CORE_API size_t pagesize() noexcept;

[[nodiscard]] LC_CORE_API luisa::string_view dynamic_module_prefix() noexcept;
[[nodiscard]] LC_CORE_API luisa::string_view dynamic_module_extension() noexcept;
[[nodiscard]] LC_CORE_API void *dynamic_module_load(const luisa::filesystem::path &path) noexcept;
LC_CORE_API void dynamic_module_destroy(void *handle) noexcept;
[[nodiscard]] LC_CORE_API void *dynamic_module_find_symbol(void *handle, luisa::string_view name) noexcept;
[[nodiscard]] LC_CORE_API luisa::string dynamic_module_name(luisa::string_view name) noexcept;
// [[nodiscard]] LC_CORE_API luisa::string demangle(const char *name) noexcept;

struct TraceItem {
    luisa::string module;
    uint64_t address;
    luisa::string symbol;
    size_t offset;
};

[[nodiscard]] LC_CORE_API LUISA_NEVER_INLINE luisa::vector<TraceItem> backtrace() noexcept;

}// namespace luisa

#endif