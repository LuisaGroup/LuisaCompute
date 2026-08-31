#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/passes/pass_verification.h>

namespace luisa::compute::xir {

class PassReport;

class Module;
class Function;

struct DestructureCFGInfo {
    size_t destructured_if_count{0u};
    size_t destructured_switch_count{0u};
    size_t destructured_loop_count{0u};
    size_t destructured_simple_loop_count{0u};
    size_t destructured_break_count{0u};
    size_t destructured_continue_count{0u};
    size_t destructured_early_return_count{0u};
    size_t leaked_block_count{0u};
    size_t error_count{0u};
    // Full generic verifier invocations owned by this public pass call. Local
    // transform-specific validation is not included.
    size_t boundary_verifier_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return destructured_if_count != 0u ||
               destructured_switch_count != 0u ||
               destructured_loop_count != 0u ||
               destructured_simple_loop_count != 0u ||
               destructured_break_count != 0u ||
               destructured_continue_count != 0u ||
               destructured_early_return_count != 0u;
    }
    [[nodiscard]] bool succeeded() const noexcept {
        return error_count == 0u && leaked_block_count == 0u;
    }
};

struct DestructureCFGOptions {
    const XIRPassVerificationTransaction *verification_transaction{nullptr};
};

// Explicitly lowers structured IF/SWITCH/LOOP/SIMPLE_LOOP/BREAK/CONTINUE
// constructs in every block owned by the function, including disconnected
// regions. SwitchInst is converted to the native raw-CFG IndexedBranchInst,
// preserving its selector, case labels, case targets, and default target.
// restructure_cfg converts it back and reconstructs the structured merge.
// Malformed constructs or unterminated owned blocks reject the complete
// function/module before mutation; leaked_block_count reports the latter.
// Declaration-like callables with no body own no CFG and are successful
// no-ops, including when mixed with definitions in a module.
[[nodiscard]] LUISA_XIR_API DestructureCFGInfo destructure_cfg_pass_run_on_function(
    Function *function,
    const DestructureCFGOptions &options = {}) noexcept;
[[nodiscard]] LUISA_XIR_API DestructureCFGInfo destructure_cfg_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;
[[nodiscard]] LUISA_XIR_API DestructureCFGInfo destructure_cfg_pass_preflight_function(Function *function) noexcept;

}// namespace luisa::compute::xir
