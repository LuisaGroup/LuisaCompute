// Dependency-light PTX `.version` patcher used by the CUDA backend to make
// newer PTX text loadable on old drivers (CUDA_ERROR_UNSUPPORTED_PTX_VERSION).
//
// Kept as a pure string/byte edit so the transform can be host-unit-tested
// without linking the whole CUDA backend or including CUDA/NVTX headers.
// The heuristic rewrites only the major field of the first `.version`
// directive (e.g. ".version 8.3" -> ".version 8.0") exactly like the
// builtin-kernel fallback path: newer minor PTX versions are forward
// compatible within the same major, so clamping to "<major>.0" is enough for
// drivers that reject only the too-new minor.

#pragma once

#include <cctype>
#include <cstddef>
#include <cstring>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::cuda {

// Rewrites the version of the first `.version` directive in PTX text in
// place. Missing or malformed directives leave the text unchanged (with a
// warning) so callers can still attempt a real module load and surface the
// driver's own error if that is the genuine failure.
inline void patch_cuda_ptx_version(luisa::string &ptx) noexcept {

    LUISA_WARNING_WITH_LOCATION(
        "The PTX version is not supported by the installed CUDA driver. "
        "Trying to patch the PTX to make it compatible with the driver. "
        "This might cause unexpected behavior. "
        "Please consider upgrading your CUDA driver. "
        "After upgrading, please clear the cache and recompile.");

    // For users with newer CUDA and older driver,
    // the generated PTX might be reported invalid.
    // We have to patch the ".version 7.x" instruction.
    using namespace std::string_view_literals;
    static constexpr auto pattern = ".version"sv;
    auto p = ptx.find(pattern);
    if (p == luisa::string::npos) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to patch PTX version. "
            "The PTX might be invalid.");
        return;
    }
    auto remaining = luisa::string_view{ptx}.substr(p + pattern.size());
    auto version_begin = 0ull;
    while (remaining[version_begin] && isblank(remaining[version_begin])) { version_begin++; }
    auto version_end = version_begin;
    auto is_digit_or_dot = [](char c) noexcept { return isdigit(c) || c == '.'; };
    while (remaining[version_end] && is_digit_or_dot(remaining[version_end])) { version_end++; }
    auto version = remaining.substr(version_begin, version_end - version_begin);
    if (version.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to patch PTX version. "
            "The PTX might be invalid.");
        return;
    }
    // get the major version
    auto sep = version.find('.');
    if (sep == luisa::string_view::npos || version.size() < sep + 2) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to patch PTX version. "
            "The PTX might be invalid.");
        return;
    }
    auto patched_version = luisa::format("{}.0", version.substr(0, sep));
    // now lets contrust the new ptx
    std::memcpy(ptx.data() + (remaining.data() - ptx.data() + version_begin),
                patched_version.data(), patched_version.size());
    ptx.erase(remaining.data() - ptx.data() + version_begin + patched_version.size(),
              version.size() - patched_version.size());
}

// Byte-vector wrapper mirroring the old CUDAShader::_patch_ptx_version
// NUL-termination handling: the trailing NUL (when present) is excluded from
// the string edit and exactly one trailing NUL is retained afterwards.
inline void patch_cuda_ptx_version_bytes(luisa::vector<std::byte> &ptx) noexcept {
    auto trailing_null = !ptx.empty() && ptx.back() == std::byte{0};
    luisa::string text{reinterpret_cast<const char *>(ptx.data()),
                       trailing_null ? ptx.size() - 1u : ptx.size()};
    patch_cuda_ptx_version(text);
    ptx.resize(text.size() + 1u);
    std::memcpy(ptx.data(), text.data(), text.size() + 1u);
}

}// namespace luisa::compute::cuda
