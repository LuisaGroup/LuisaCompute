#include <cstdlib>
#include <nvtx3/nvToolsExtCuda.h>

#include <luisa/core/logging.h>

#include "cuda_ptx_version.h"
#include "cuda_shader_metadata.h"
#include "cuda_shader_printer.h"
#include "cuda_shader.h"

namespace luisa::compute::cuda {

void CUDAShader::_patch_ptx_version(luisa::vector<std::byte> &ptx) noexcept {
    patch_cuda_ptx_version_bytes(ptx);
}

CUDAShader::CUDAShader(luisa::unique_ptr<CUDAShaderPrinter> printer,
                       luisa::vector<Usage> arg_usages,
                       luisa::vector<uint32_t> texture_sampler_masks) noexcept
    : _printer{std::move(printer)},
      _argument_usages{std::move(arg_usages)},
      _texture_sampler_masks{std::move(texture_sampler_masks)} {}

CUDAShader::~CUDAShader() noexcept = default;

Usage CUDAShader::argument_usage(size_t i) const noexcept {
    LUISA_ASSERT(i < _argument_usages.size(),
                 "Invalid argument index {} for shader with {} argument(s).",
                 i, _argument_usages.size());
    return _argument_usages[i];
}

uint32_t CUDAShader::texture_sampler_mask(size_t i) const noexcept {
    return i < _texture_sampler_masks.size() ?
               _texture_sampler_masks[i] :
               CUDAShaderMetadata::texture_uses_surface_bit;
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
