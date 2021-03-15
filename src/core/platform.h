//
// Created by Mike Smith on 2021/2/11.
//

#pragma once

#ifdef _MSC_VER
#define LUISA_FORCE_INLINE __forceinline
#define LUISA_EXPORT __declspec(dllexport)
#else
#define LUISA_FORCE_INLINE [[gnu::always_inline, gnu::hot]] inline
#define LUISA_EXPORT [[gnu::visibility("default")]]
#endif

#ifdef _WINDOWS
#define LUISA_PLATFORM_WINDOWS
#define LUISA_DLL_HANDLE HMODULE
#elif defined(__unix__) || defined(__unix) || defined(__APPLE__)
#define LUISA_PLATFORM_UNIX
#define LUISA_DLL_HANDLE void *
#endif

#include <string_view>

namespace luisa {

[[nodiscard]] void *aligned_alloc(size_t alignment, size_t size) noexcept;
void aligned_free(void *p) noexcept;
[[nodiscard]] size_t pagesize() noexcept;

[[nodiscard]] const char *dynamic_module_prefix() noexcept;
[[nodiscard]] const char *dynamic_module_extension() noexcept;
[[nodiscard]] LUISA_DLL_HANDLE dynamic_module_load(const char *path) noexcept;
void dynamic_module_destroy(LUISA_DLL_HANDLE handle) noexcept;
[[nodiscard]] void *dynamic_module_find_symbol(LUISA_DLL_HANDLE handle, const char *name) noexcept;

}// namespace luisa
