#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::xir {

class Module;
class Function;

struct AutodiffInfo {
    size_t transformed_scope_count{0u};
    size_t removed_instruction_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return transformed_scope_count != 0u ||
               removed_instruction_count != 0u;
    }
};

struct AutodiffOptions {
    bool run_forward{true};
    bool run_backward{true};
};

// Reverse-mode loop handling is a private semantic expansion with a hard
// 64-iteration guard; it is not the removed generic XIR loop-unroll
// optimization. Structured scope/loop terminators and cloned region blocks
// preserve instruction-local metadata on their explicit replacements.
LUISA_XIR_API AutodiffInfo autodiff_pass_run_on_function(Function *function, const AutodiffOptions &options = {}) noexcept;
LUISA_XIR_API AutodiffInfo autodiff_pass_run_on_module(Module *module, const AutodiffOptions &options = {}) noexcept;

}// namespace luisa::compute::xir
