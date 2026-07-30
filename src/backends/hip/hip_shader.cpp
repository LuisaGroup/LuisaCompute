#include <luisa/runtime/rhi/command.h>
#include <luisa/core/logging.h>
#include "hip_shader_printer.h"
#include "hip_shader.h"
#include "hip_command_encoder.h"

namespace luisa::compute::hip {

HIPShader::HIPShader(luisa::unique_ptr<HIPShaderPrinter> printer,
                     luisa::vector<Usage> arg_usages) noexcept
    : _printer{std::move(printer)},
      _argument_usages{std::move(arg_usages)} {}

HIPShader::~HIPShader() noexcept = default;

Usage HIPShader::argument_usage(size_t i) const noexcept {
    LUISA_ASSERT(i < _argument_usages.size(),
                 "Invalid argument index {} for shader with {} argument(s).",
                 i, _argument_usages.size());
    return _argument_usages[i];
}

void HIPShader::set_name(luisa::string &&name) noexcept {
    std::scoped_lock lock{_name_mutex};
    _name = std::move(name);
}

void HIPShader::launch(HIPCommandEncoder &encoder,
                       ShaderDispatchCommand *command) const noexcept {
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
    _launch(encoder, command);
}

}// namespace luisa::compute::hip
