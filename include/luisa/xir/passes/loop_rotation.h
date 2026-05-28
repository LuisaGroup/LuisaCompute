#pragma once

#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

struct LoopRotationInfo {
    size_t rotated_loop_count{0u};
};

[[nodiscard]] LUISA_XIR_API LoopRotationInfo loop_rotation_pass_run_on_function(FunctionDefinition *def) noexcept;
[[nodiscard]] LUISA_XIR_API LoopRotationInfo loop_rotation_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
