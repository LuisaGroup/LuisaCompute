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

namespace platform {

// ---------------------------------------------------------------------------
// Narrow <-> wide string conversion primitives over the ANSI code page
// (CP_ACP). Windows-only; POSIX builds keep paths as plain byte strings and
// the owning helpers below never call these.
//
// These primitives NEVER allocate: they only measure or fill caller-provided
// buffers, which makes them safe to export from luisa-core even for modules
// that link a different allocator override (see the note on the owning
// helpers below).
//
// narrow_to_wide_acp:
//   Query:  dst == nullptr -> returns the required wchar_t capacity INCLUDING
//           the null terminator, or 0 when `text` is empty or is not
//           representable in the ANSI code page (MB_ERR_INVALID_CHARS).
//   Fill:   writes up to `capacity` wchar_t (including the null terminator
//           when it fits) and returns the number of wchar_t written. Note:
//           whether that count includes the null terminator differs between
//           Windows builds (documented as excluded; observed as included), so
//           callers should size the buffer with the query result and strip
//           trailing nulls from the string instead of trusting the count.
// wide_to_narrow_acp:
//   Same contract over bytes. `strict` adds WC_ERR_INVALID_CHARS |
//   WC_NO_BEST_FIT_CHARS so an unrepresentable character fails (0 for
//   non-empty input) instead of becoming the default replacement character.
// ---------------------------------------------------------------------------
#if defined(LUISA_PLATFORM_WINDOWS) || defined(_WIN32) || defined(_WIN64)
[[nodiscard]] LUISA_CORE_API size_t narrow_to_wide_acp(luisa::string_view text, wchar_t *dst, size_t capacity) noexcept;
[[nodiscard]] LUISA_CORE_API size_t wide_to_narrow_acp(luisa::wstring_view text, char *dst, size_t capacity, bool strict) noexcept;
#endif

// Owning path <-> narrow-string conversions with the same semantics as
// luisa::path_from_narrow() and luisa::to_string(const filesystem::path&):
// the ANSI code page, never throwing (empty or undecodable input becomes an
// empty result), and path_to_narrow() degrades lossily (replacement
// characters) when the strict conversion fails.
//
// THESE HELPERS ARE INLINE ON PURPOSE. Every module in this project links its
// own allocator override - luisa-core uses mimalloc, luisa-backend-cuda uses
// the rpmalloc bundled in LLVMSupport.lib, other modules use the plain CRT -
// so an owning std::filesystem::path / luisa::string must be allocated AND
// freed inside the same module. A core-exported factory handing such an
// object to another module transfers a block across allocators and crashes at
// destruction (observed: access violation inside rpmalloc's free on the CUDA
// backend's first shader-cache lookup). Being inline gives every caller
// module its own copy; the allocation-free primitives above do the heavy
// lifting and are the only part that crosses the module boundary.
[[nodiscard]] inline luisa::filesystem::path path_from_narrow(luisa::string_view text) noexcept {
    luisa::filesystem::path result;
#if defined(LUISA_PLATFORM_WINDOWS) || defined(_WIN32) || defined(_WIN64)
    if (const size_t needed = narrow_to_wide_acp(text, nullptr, 0u); needed > 1u) {
        luisa::wstring wide(needed, L'\0');
        if (narrow_to_wide_acp(text, &wide[0], needed) != 0u) {
            while (!wide.empty() && wide.back() == L'\0') { wide.pop_back(); }
            result = luisa::filesystem::path{wide};
        }
    }
#else
    // POSIX paths are plain byte strings; construction cannot fail.
    result = luisa::filesystem::path{text};
#endif
    return result;
}

[[nodiscard]] inline luisa::string path_to_narrow(const luisa::filesystem::path &path) noexcept {
#if defined(LUISA_PLATFORM_WINDOWS) || defined(_WIN32) || defined(_WIN64)
    const std::wstring_view wide{path.native()};
    if (wide.empty()) { return {}; }
    // Strict CP_ACP first (byte-identical to path.string<char>() for every
    // representable path); lossy CP_ACP second, mirroring how the rest of
    // Windows resolves unrepresentable names.
    for (const bool strict : {true, false}) {
        if (const size_t needed = wide_to_narrow_acp(wide, nullptr, 0u, strict); needed > 1u) {
            luisa::string out(needed, '\0');
            if (wide_to_narrow_acp(wide, &out[0], needed, strict) != 0u) {
                while (!out.empty() && out.back() == '\0') { out.pop_back(); }
                return out;
            }
        }
    }
    return {};
#else
    return luisa::string{path.native()};
#endif
}

}// namespace platform
}// namespace luisa

#endif
