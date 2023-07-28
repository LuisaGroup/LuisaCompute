#include <cstdlib>
#include <nvtx3/nvToolsExtCuda.h>

#include <luisa/core/logging.h>
#include "cuda_shader.h"

namespace luisa::compute::cuda {

CUDAShader::CUDAShader(luisa::vector<Usage> arg_usages) noexcept
    : _argument_usages{std::move(arg_usages)} {}

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
    auto name = [this] {
        std::scoped_lock lock{_name_mutex};
        return _name;
    }();
    if (!name.empty()) { nvtxRangePushA(name.c_str()); }
    _launch(encoder, command);
    if (!name.empty()) { nvtxRangePop(); }
}

}// namespace luisa::compute::cuda

