#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/intrin.h>

#ifdef __cplusplus

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/optional.h>

namespace luisa {

[[nodiscard]] LUISA_CORE_API void *aligned_alloc(size_t alignment, size_t size) noexcept;
LUISA_CORE_API void aligned_free(void *p) noexcept;
[[nodiscard]] LUISA_CORE_API size_t pagesize() noexcept;

[[nodiscard]] LUISA_CORE_API luisa::string_view dynamic_module_prefix() noexcept;
[[nodiscard]] LUISA_CORE_API luisa::string_view dynamic_module_extension() noexcept;
[[nodiscard]] LUISA_CORE_API void *dynamic_module_load(const luisa::filesystem::path &path) noexcept;
LUISA_CORE_API void dynamic_module_destroy(void *handle) noexcept;
[[nodiscard]] LUISA_CORE_API void *dynamic_module_find_symbol(void *handle, const char *name) noexcept;
[[nodiscard]] LUISA_CORE_API luisa::string dynamic_module_name(luisa::string_view name) noexcept;
// [[nodiscard]] LUISA_CORE_API luisa::string demangle(const char *name) noexcept;

struct TraceItem {
    luisa::string module;
    uint64_t address;
    luisa::string symbol;
    size_t offset;
};

[[nodiscard]] LUISA_CORE_API luisa::string to_string(const TraceItem &item) noexcept;
[[nodiscard]] LUISA_CORE_API LUISA_NEVER_INLINE luisa::vector<TraceItem> backtrace() noexcept;
[[nodiscard]] LUISA_CORE_API luisa::string cpu_name() noexcept;

[[nodiscard]] LUISA_CORE_API luisa::string current_executable_path() noexcept;
[[nodiscard]] LUISA_CORE_API char env_separator() noexcept;
// Reads the current process environment, including changes made after a DLL
// was loaded. The owned result distinguishes an unset variable from an empty one.
[[nodiscard]] LUISA_CORE_API luisa::optional<luisa::string> get_environment_variable(const char *name) noexcept;

}// namespace luisa

#endif
