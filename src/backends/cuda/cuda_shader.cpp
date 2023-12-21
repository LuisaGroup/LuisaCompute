#include <cstdlib>
#include <nvtx3/nvToolsExtCuda.h>

#include <luisa/core/logging.h>

#include "cuda_shader_printer.h"
#include "cuda_shader.h"

namespace luisa::compute::cuda {

void CUDAShader::_patch_ptx_version(luisa::string &ptx) noexcept {

    LUISA_WARNING_WITH_LOCATION(
        "The PTX version is not supported by the installed CUDA driver. "
        "Trying to patch the PTX to make it compatible with the driver. "
        "This might cause unexpected behavior. "
        "Please consider upgrading your CUDA driver.");

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

CUDAShader::CUDAShader(luisa::unique_ptr<CUDAShaderPrinter> printer,
                       luisa::vector<Usage> arg_usages) noexcept
    : _printer{std::move(printer)},
      _argument_usages{std::move(arg_usages)} {}

CUDAShader::~CUDAShader() noexcept = default;

Usage CUDAShader::argument_usage(size_t i) const noexcept {
    LUISA_ASSERT(i < _argument_usages.size(),
                 "Invalid argument index {} for shader with {} argument(s).",
                 i, _argument_usages.size());
    return _argument_usages[i];
}

void CUDAShader::set_name(luisa::string &&name) noexcept {
    std::scoped_lock lock{_name_mutex};
    _name = std::move(name);
}

void CUDAShader::launch(CUDACommandEncoder &encoder,
                        ShaderDispatchCommand *command) const noexcept {

    // check if this is an empty launch
    auto report_empty_launch = [&]() noexcept {
#ifndef NDEBUG
        LUISA_WARNING_WITH_LOCATION(
            "Empty launch detected. "
            "This might be caused by a shader dispatch command with all dispatch sizes set to zero. "
            "The command will be ignored.");
#endif
    };
    if (command->is_indirect()) {
        auto indirect = command->indirect_dispatch();
        if (indirect.max_dispatch_size == 0u) {
            report_empty_launch();
            return;
        }
    } else if (command->is_multiple_dispatch()) {
        auto dispatch_sizes = command->dispatch_sizes();
        if (std::all_of(dispatch_sizes.begin(), dispatch_sizes.end(),
                        [](auto size) noexcept { return any(size == make_uint3(0u)); })) {
            report_empty_launch();
            return;
        }
    } else {
        auto dispatch_size = command->dispatch_size();
        if (any(dispatch_size == make_uint3(0u))) {
            report_empty_launch();
            return;
        }
    }

    auto name = [this] {
        std::scoped_lock lock{_name_mutex};
        return _name;
    }();
    if (!name.empty()) { nvtxRangePushA(name.c_str()); }
    _launch(encoder, command);
    if (!name.empty()) { nvtxRangePop(); }
}

}// namespace luisa::compute::cuda
