#pragma once

#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

// Unstructured-CFG-only loop rotation pass. Callers must run an explicit CFG
// destructuring pass first. Structured control flow is rejected and left
// unchanged; see LoopRotationInfo::structured_cfg_error_count. Rotation is
// limited to verifier-valid, innermost, top-tested natural loops whose header
// is pure, whose exit has no PHIs, and whose loop values do not escape.
struct LoopRotationInfo {
    size_t rotated_loop_count{0u};
    size_t structured_cfg_error_count{0u};
    [[nodiscard]] bool changed() const noexcept { return rotated_loop_count != 0u; }
    [[nodiscard]] bool succeeded() const noexcept { return structured_cfg_error_count == 0u; }
};

[[nodiscard]] LUISA_XIR_API LoopRotationInfo loop_rotation_pass_run_on_function(FunctionDefinition *def) noexcept;
[[nodiscard]] LUISA_XIR_API LoopRotationInfo loop_rotation_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
